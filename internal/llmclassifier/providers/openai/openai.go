// Package openai implements the classifier provider against an OpenAI-compatible chat completions API.
//
// TwoStageOff: single JSON-mode call. The prompt asks the model to emit the classifier schema.
//
// TwoStageBoth/Fast: stage 1 is a quick XML call with max_tokens and `</block>` stop_sequences asking for
// `<block>yes/no</block>`. Allow verdicts return immediately. In Both, block verdicts escalate to a chain-of-thought
// stage 2 for the authoritative reason; in Fast, stage 1's reason is final.
//
// TwoStageThinking: single non-streaming XML chain-of-thought call.
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"claude-auto-permission/internal/llmclassifier/prompt"
	"claude-auto-permission/internal/llmclassifier/providers"
)

// ToolName matches the system prompt's `classify_result` reference. Used only for logging / metadata symmetry with
// the Bedrock provider.
const ToolName = "classify_result"

// xmlS1BothSuffix runs stage 1 as a conservative prefilter; stage 2 applies user intent and ALLOW exceptions.
const xmlS1BothSuffix = "\nErr on the side of blocking. Stage 1 does NOT apply user intent or ALLOW exceptions — stage 2 will handle those. Judge the action by its full effect — what it runs, sends, publishes, or enables — not its surface form. Block if ANY rule could apply. <block> immediately."

// xmlS1FastSuffix nudges an immediate final `<block>` verdict from fast-only mode.
const xmlS1FastSuffix = "\nErr on the side of blocking. <block> immediately."

// xmlS2Suffix elicits chain-of-thought before the stage-2 verdict.
const xmlS2Suffix = "\nReview the classification process and follow it carefully, making sure you deny actions that should be blocked. As a reminder, explicit (not suggestive or implicit) user confirmation is required to override blocks. Use <thinking> before responding with <block>. Think longer on ambiguous or borderline actions; keep reasoning brief for clear-cut ones."

const stage1StopSequence = "</block>"

// xmlBlockRegex tolerates a missing closing tag because some backends strip the stop sequence from the response.
var xmlBlockRegex = regexp.MustCompile(`(?i)<block>\s*(yes|no)\b\s*(?:</block>)?`)

// xmlReasonRegex extracts the optional one-sentence reason.
var xmlReasonRegex = regexp.MustCompile(`(?s)<reason>\s*(.*?)\s*</reason>`)

// Strip thinking blocks (and unclosed tails) before parsing so tags inside chain-of-thought aren't mistaken for the
// verdict.
var xmlThinkingRegex = regexp.MustCompile(`(?s)<thinking>.*?</thinking>`)
var xmlPartialThinkingRegex = regexp.MustCompile(`(?s)<thinking>.*$`)

func stripThinking(s string) string {
	s = xmlThinkingRegex.ReplaceAllString(s, "")
	s = xmlPartialThinkingRegex.ReplaceAllString(s, "")
	return s
}

// parseXMLBlock returns (true, true) for `<block>yes…`, (false, true) for `<block>no…`, and (false, false) when no
// parseable verdict.
func parseXMLBlock(s string) (block bool, ok bool) {
	m := xmlBlockRegex.FindStringSubmatch(stripThinking(s))
	if m == nil {
		return false, false
	}
	return strings.EqualFold(m[1], "yes"), true
}

func parseXMLReason(s string) string {
	m := xmlReasonRegex.FindStringSubmatch(stripThinking(s))
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

// Provider is the OpenAI-native classifier implementation.
type Provider struct {
	cfg    Config
	client *client
}

// New constructs a Provider from cfg. AWS credentials are not required.
func New(ctx context.Context, cfg Config) (*Provider, error) {
	if cfg.Model == "" {
		return nil, errors.New("openai: Model is required")
	}
	c, err := newClient(cfg.BaseURL, cfg.APIKey)
	if err != nil {
		return nil, err
	}
	return &Provider{
		cfg:    cfg,
		client: c,
	}, nil
}

func (p *Provider) Name() string { return "openai" }

func (p *Provider) Model() string { return p.cfg.Model }

// Classify dispatches to the single- or two-stage path per cfg.TwoStage.
func (p *Provider) Classify(ctx context.Context, req providers.Request) (providers.Result, error) {
	if p.cfg.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, p.cfg.Timeout)
		defer cancel()
	}

	switch p.cfg.TwoStage {
	case TwoStageOff:
		return p.classifyJSON(ctx, req)
	case TwoStageFast, TwoStageBoth, TwoStageUnspecified:
		return p.classifyTwoStage(ctx, req)
	case TwoStageThinking:
		return p.classifyXML(ctx, req, xmlS2Suffix, p.cfg.MaxTokens, nil)
	default:
		return p.classifyTwoStage(ctx, req)
	}
}

// classifyJSON: one chat completion with response_format={"type":"json_object"}, forcing the model to emit the
// classifier schema.
func (p *Provider) classifyJSON(ctx context.Context, req providers.Request) (providers.Result, error) {
	system := rewriteSystemForJSON(req.SystemPrompt)
	body := req.UserPrompt

	messages := chatMessages(req.UserPrefix, body)
	if system != "" {
		messages = append([]chatMessage{{Role: "system", Content: system}}, messages...)
	}

	completionReq := chatCompletionRequest{
		Model:          p.cfg.Model,
		Messages:       messages,
		MaxTokens:      p.cfg.MaxTokens,
		ResponseFormat: &responseFmt{Type: "json_object"},
	}

	start := time.Now()
	text, err := p.client.chat(ctx, completionReq)
	latency := int(time.Since(start) / time.Millisecond)
	if err != nil {
		return providers.Result{LatencyMs: latency}, classifyErr(ctx, err, latency)
	}

	res, parseErr := parseJSONVerdict(text)
	if parseErr != nil {
		return providers.Result{
			LatencyMs:   latency,
			RawResponse: []byte(text),
		}, &providers.Error{
			Reason: providers.UnavailableParse,
			Err:    parseErr,
		}
	}
	res.LatencyMs = latency
	res.RawResponse = []byte(text)
	return res, nil
}

// rewriteSystemForJSON keeps JSON-native prompts unchanged, upgrades older XML prompts, and appends the JSON contract
// only when the caller supplied a custom prompt with no output-format instruction.
func rewriteSystemForJSON(s string) string {
	if strings.Contains(s, prompt.ForcedToolResultLine) {
		return s
	}
	if strings.Contains(s, prompt.XMLOutputFormat) {
		return strings.Replace(s, prompt.XMLOutputFormat, jsonOutputFormatLine, 1)
	}
	return strings.TrimRight(s, "\n") + "\n\n" + jsonOutputFormatLine
}

const jsonOutputFormatLine = "Respond with a single JSON object matching the provided schema, containing `thinking`, `shouldBlock`, and `reason` fields. Do not include any prose outside the JSON object."

// parseJSONVerdict parses the model's JSON output into a Result. Failures return a structured parse error so callers
// can fall back safely.
func parseJSONVerdict(text string) (providers.Result, error) {
	var parsed struct {
		Thinking    *string `json:"thinking"`
		ShouldBlock *bool   `json:"shouldBlock"`
		Reason      string  `json:"reason"`
	}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return providers.Result{}, fmt.Errorf("decode verdict JSON: %w", err)
	}
	if parsed.ShouldBlock == nil {
		// Fail closed on malformed output, matching the Bedrock provider and Claude Code's reference impl.
		return providers.Result{
			ShouldBlock: true,
			Reason:      "model omitted shouldBlock — blocking for safety",
		}, nil
	}
	return providers.Result{
		ShouldBlock: *parsed.ShouldBlock,
		Reason:      parsed.Reason,
	}, nil
}

// classifyTwoStage runs stage 1 and either short-circuits on allow, finalizes on block in Fast mode, or escalates to
// stage 2 in Both mode.
func (p *Provider) classifyTwoStage(ctx context.Context, req providers.Request) (providers.Result, error) {
	system := rewriteSystemForXml(req.SystemPrompt)
	stage1Body := wrapTranscript(req.UserPrompt, xmlS1BothSuffix)

	stage1MaxTokens := p.cfg.Stage1MaxTokensBoth
	stopSequences := []string{stage1StopSequence}
	if p.cfg.TwoStage == TwoStageFast {
		stage1Body = wrapTranscript(req.UserPrompt, xmlS1FastSuffix)
		stage1MaxTokens = p.cfg.Stage1MaxTokensFast
		stopSequences = nil
	}

	messages := chatMessages(req.UserPrefix, stage1Body)
	if system != "" {
		messages = append([]chatMessage{{Role: "system", Content: system}}, messages...)
	}

	stage1Req := chatCompletionRequest{
		Model:     p.cfg.Model,
		Messages:  messages,
		MaxTokens: stage1MaxTokens,
		Stop:      stopSequences,
	}

	stage1Start := time.Now()
	stage1Text, err := p.client.chat(ctx, stage1Req)
	stage1Ms := int(time.Since(stage1Start) / time.Millisecond)
	if err != nil {
		return providers.Result{LatencyMs: stage1Ms}, classifyErr(ctx, err, stage1Ms)
	}

	block, ok := parseXMLBlock(stage1Text)
	switch {
	case ok && !block:
		return providers.Result{
			ShouldBlock: false,
			RawResponse: []byte(stage1Text),
			LatencyMs:   stage1Ms,
		}, nil

	case ok && block && p.cfg.TwoStage == TwoStageFast:
		reason := parseXMLReason(stage1Text)
		if reason == "" {
			reason = "Blocked by fast classifier"
		}
		return providers.Result{
			ShouldBlock: true,
			Reason:      reason,
			RawResponse: []byte(stage1Text),
			LatencyMs:   stage1Ms,
		}, nil

	case !ok && p.cfg.TwoStage == TwoStageFast:
		return providers.Result{
			ShouldBlock: true,
			Reason:      "Classifier stage 1 unparseable - blocking for safety",
			RawResponse: []byte(stage1Text),
			LatencyMs:   stage1Ms,
		}, nil
	}

	stage2Result, stage2Err := p.classifyXML(ctx, req, xmlS2Suffix, p.cfg.MaxTokens, nil)
	stage2Result.LatencyMs += stage1Ms
	return stage2Result, stage2Err
}

// classifyXML runs one XML-format chat completion: TwoStageThinking (single call) or TwoStageBoth stage 2 (escalation).
func (p *Provider) classifyXML(
	ctx context.Context,
	req providers.Request,
	suffix string,
	maxTokens int32,
	stopSequences []string,
) (providers.Result, error) {
	system := rewriteSystemForXml(req.SystemPrompt)
	body := wrapTranscript(req.UserPrompt, suffix)

	messages := chatMessages(req.UserPrefix, body)
	if system != "" {
		messages = append([]chatMessage{{Role: "system", Content: system}}, messages...)
	}

	completionReq := chatCompletionRequest{
		Model:     p.cfg.Model,
		Messages:  messages,
		MaxTokens: maxTokens,
		Stop:      stopSequences,
	}

	start := time.Now()
	text, err := p.client.chat(ctx, completionReq)
	latency := int(time.Since(start) / time.Millisecond)
	if err != nil {
		return providers.Result{LatencyMs: latency}, classifyErr(ctx, err, latency)
	}

	block, ok := parseXMLBlock(text)
	if !ok {
		return providers.Result{
			ShouldBlock: true,
			Reason:      "Classifier stage 2 unparseable - blocking for safety",
			RawResponse: []byte(text),
			LatencyMs:   latency,
		}, nil
	}
	reason := parseXMLReason(text)
	if block && reason == "" {
		reason = "Blocked by classifier"
	}
	return providers.Result{
		ShouldBlock: block,
		Reason:      reason,
		RawResponse: []byte(text),
		LatencyMs:   latency,
	}, nil
}

// rewriteSystemForXml keeps XML-native prompts unchanged, upgrades older JSON prompts, and appends the XML contract
// only when the caller supplied a custom prompt with no output-format instruction.
func rewriteSystemForXml(s string) string {
	if strings.Contains(s, prompt.XMLOutputFormat) {
		return s
	}
	if strings.Contains(s, prompt.ForcedToolResultLine) {
		return strings.Replace(s, prompt.ForcedToolResultLine, prompt.XMLOutputFormat, 1)
	}
	return strings.TrimRight(s, "\n") + "\n\n" + prompt.XMLOutputFormat
}

// wrapTranscript wraps the body in `<transcript>` and appends suffix.
func wrapTranscript(body, suffix string) string {
	return "<transcript>\n" + body + "\n</transcript>\n" + suffix
}

// chatMessages emits CLAUDE.md (when present) as a separate user-role message before the transcript.
func chatMessages(prefix, body string) []chatMessage {
	var msgs []chatMessage
	if prefix != "" {
		msgs = append(msgs, chatMessage{Role: "user", Content: prefix})
	}
	msgs = append(msgs, chatMessage{Role: "user", Content: body})
	return msgs
}

// classifyErr maps a client error to a tagged providers.Error so the orchestrator can distinguish timeout,
// prompt-too-long, and generic outage.
func classifyErr(ctx context.Context, err error, latency int) error {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return &providers.Error{
			Reason: providers.UnavailableTimeout,
			Err:    fmt.Errorf("after %dms", latency),
		}
	}
	if isPromptTooLong(err.Error()) {
		return &providers.Error{
			Reason: providers.UnavailableTooLong,
			Err:    err,
		}
	}
	return &providers.Error{
		Reason: providers.UnavailableProviderErr,
		Err:    err,
	}
}

// isPromptTooLong recognizes prompt-too-long errors across common backend phrasings.
func isPromptTooLong(s string) bool {
	lower := strings.ToLower(s)
	return strings.Contains(lower, "input is too long") ||
		strings.Contains(lower, "too many tokens") ||
		strings.Contains(lower, "prompt is too long") ||
		strings.Contains(lower, "exceed") && strings.Contains(lower, "context")
}
