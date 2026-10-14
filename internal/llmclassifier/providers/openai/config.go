// Package openai implements the classifier provider against an OpenAI-compatible chat completions API.
//
// It targets endpoints like the CSCS Inference API (https://api.inference.cscs.ch/v1) but is generic enough to work
// with any OpenAI-compatible provider. Authentication uses a bearer token read from an environment variable by
// default; an inline api_key can be configured but is not recommended.
package openai

import "time"

// TwoStageMode picks single- vs two-stage classification. Values match OpenAIProvider.TwoStageMode in the proto so
// callers can translate by simple cast. The proto-zero `TwoStageUnspecified` is a sentinel; the production default
// (`TwoStageBoth`) is applied via the schema's `(opts).default` annotation in `loader.FillDefaults`.
type TwoStageMode int

const (
	// TwoStageUnspecified is the proto-zero sentinel. Treated as `TwoStageBoth` by [Provider.Classify] for defense in
	// depth, but the schema default should remap it before this code sees it.
	TwoStageUnspecified TwoStageMode = iota
	// TwoStageOff: single JSON-mode call with the classifier schema.
	TwoStageOff
	// TwoStageFast: stage 1 XML only. Verdict and reason both from the XML; no stop sequences.
	TwoStageFast
	// TwoStageBoth: production default. Stage 1 XML with stop sequence `</block>`; on block, stage 2 chain-of-thought.
	TwoStageBoth
	// TwoStageThinking: single XML chain-of-thought call.
	TwoStageThinking
)

// Config holds the per-invocation knobs.
type Config struct {
	// BaseURL is the OpenAI-compatible endpoint root, e.g. "https://api.inference.cscs.ch/v1".
	BaseURL string

	// APIKey is the bearer token. It may be empty if APIKeyEnv is set and resolves at construction time.
	APIKey string

	// Model is the model identifier to request, e.g. "swiss-ai/Apertus-v1.5-70B".
	Model string

	// MaxTokens caps output for the authoritative classifier call.
	MaxTokens int32

	// Timeout bounds the whole round-trip (both stages in two-stage modes).
	Timeout time.Duration

	// TwoStage selects single- vs two-stage classification.
	TwoStage TwoStageMode

	// Stage-1 max_tokens budgets.
	Stage1MaxTokensBoth int32
	Stage1MaxTokensFast int32
}
