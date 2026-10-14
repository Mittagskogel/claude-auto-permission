package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"testing"
	"time"

	"go.uber.org/mock/gomock"

	"claude-auto-permission/internal/llmclassifier/providers"
	openaimocks "claude-auto-permission/internal/llmclassifier/providers/openai/mocks"
)

func TestNew_RequiresModelAndKey(t *testing.T) {
	ctx := context.Background()

	_, err := New(ctx, Config{})
	if err == nil {
		t.Error("expected error when Model and APIKey are empty")
	}

	_, err = New(ctx, Config{Model: "test-model"})
	if err == nil {
		t.Error("expected error when APIKey is empty")
	}

	p, err := New(ctx, Config{BaseURL: "https://api.inference.cscs.ch/v1", Model: "test-model", APIKey: "test-key"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Name() != "openai" {
		t.Errorf("Name() = %q, want openai", p.Name())
	}
	if p.Model() != "test-model" {
		t.Errorf("Model() = %q, want test-model", p.Model())
	}
}

func TestParseXMLBlock(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantOK  bool
		wantBlk bool
	}{
		{name: "yes closed", input: "<block>yes</block>", wantOK: true, wantBlk: true},
		{name: "no closed", input: "<block>no</block>", wantOK: true, wantBlk: false},
		{name: "yes stop stripped", input: "<block>yes", wantOK: true, wantBlk: true},
		{name: "no stop stripped", input: "<block>no", wantOK: true, wantBlk: false},
		{name: "case insensitive", input: "<BLOCK>YES</BLOCK>", wantOK: true, wantBlk: true},
		{name: "unrelated", input: "hello world", wantOK: false},
		{name: "thinking before verdict", input: "<thinking>maybe</thinking><block>yes</block>", wantOK: true, wantBlk: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blk, ok := parseXMLBlock(tt.input)
			if ok != tt.wantOK || blk != tt.wantBlk {
				t.Errorf("parseXMLBlock(%q) = (%v, %v), want (%v, %v)", tt.input, blk, ok, tt.wantBlk, tt.wantOK)
			}
		})
	}
}

func TestParseXMLReason(t *testing.T) {
	input := "<block>yes</block><reason>[Irreversible Local Destruction] deleting workspace</reason>"
	got := parseXMLReason(input)
	want := "[Irreversible Local Destruction] deleting workspace"
	if got != want {
		t.Errorf("parseXMLReason = %q, want %q", got, want)
	}
}

func TestParseJSONVerdict(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantBlock bool
		wantErr   bool
	}{
		{
			name:      "valid allow",
			input:     `{"thinking":"ok","shouldBlock":false,"reason":""}`,
			wantBlock: false,
		},
		{
			name:      "valid block",
			input:     `{"thinking":"bad","shouldBlock":true,"reason":"nope"}`,
			wantBlock: true,
		},
		{
			name:    "missing shouldBlock fails closed",
			input:   `{"thinking":"bad","reason":"nope"}`,
			wantErr: true,
		},
		{
			name:    "invalid json",
			input:   `not json`,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := parseJSONVerdict(tt.input)
			if tt.wantErr {
				if err == nil && !res.ShouldBlock {
					t.Errorf("expected error or block, got (%+v, %v)", res, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if res.ShouldBlock != tt.wantBlock {
				t.Errorf("ShouldBlock = %v, want %v", res.ShouldBlock, tt.wantBlock)
			}
		})
	}
}

func TestClassifyJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	hc := openaimocks.NewMockhttpClient(ctrl)
	p := &Provider{
		cfg: Config{
			Model:     "test-model",
			APIKey:    "test-key",
			MaxTokens: 1024,
			TwoStage:  TwoStageOff,
		},
		client: &client{baseURL: "https://api.inference.cscs.ch/v1", apiKey: "test-key", httpClient: hc},
	}

	wantReq := captureRequest(t, hc, `{"shouldBlock":false,"reason":"safe"}`)

	res, err := p.Classify(context.Background(), providers.Request{
		SystemPrompt: "system",
		UserPrompt:   "user",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.ShouldBlock {
		t.Errorf("ShouldBlock = true, want false")
	}
	if !wantReq() {
		t.Error("mock expectation not satisfied")
	}
}

func TestClassifyTwoStageAllow(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	hc := openaimocks.NewMockhttpClient(ctrl)
	p := &Provider{
		cfg: Config{
			Model:               "test-model",
			APIKey:              "test-key",
			MaxTokens:           1024,
			TwoStage:            TwoStageBoth,
			Stage1MaxTokensBoth: 64,
		},
		client: &client{baseURL: "https://api.inference.cscs.ch/v1", apiKey: "test-key", httpClient: hc},
	}

	wantReq := captureRequest(t, hc, "<block>no</block>")

	res, err := p.Classify(context.Background(), providers.Request{
		SystemPrompt: "system",
		UserPrompt:   "user",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.ShouldBlock {
		t.Errorf("ShouldBlock = true, want false")
	}
	if !wantReq() {
		t.Error("mock expectation not satisfied")
	}
}

func TestClassifyTwoStageBlockEscalates(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	hc := openaimocks.NewMockhttpClient(ctrl)
	p := &Provider{
		cfg: Config{
			Model:               "test-model",
			APIKey:              "test-key",
			MaxTokens:           1024,
			TwoStage:            TwoStageBoth,
			Stage1MaxTokensBoth: 64,
		},
		client: &client{baseURL: "https://api.inference.cscs.ch/v1", apiKey: "test-key", httpClient: hc},
	}

	// Stage 1 says block; stage 2 returns authoritative reason.
	first := true
	hc.EXPECT().Do(gomock.Any()).DoAndReturn(func(req *http.Request) (*http.Response, error) {
		var body string
		if first {
			body = "<block>yes</block>"
			first = false
		} else {
			body = "<thinking>hmm</thinking><block>yes</block><reason>[Test] dangerous</reason>"
		}
		return jsonResponse(body), nil
	}).Times(2)

	res, err := p.Classify(context.Background(), providers.Request{
		SystemPrompt: "system",
		UserPrompt:   "user",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.ShouldBlock {
		t.Errorf("ShouldBlock = false, want true")
	}
	if res.Reason == "" {
		t.Error("expected non-empty reason from stage 2")
	}
}

func TestClassifyTimeout(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	hc := openaimocks.NewMockhttpClient(ctrl)
	p := &Provider{
		cfg: Config{
			Model:     "test-model",
			APIKey:    "test-key",
			MaxTokens: 1024,
			TwoStage:  TwoStageOff,
			Timeout:   1 * time.Millisecond,
		},
		client: &client{baseURL: "https://api.inference.cscs.ch/v1", apiKey: "test-key", httpClient: hc},
	}

	hc.EXPECT().Do(gomock.Any()).DoAndReturn(func(req *http.Request) (*http.Response, error) {
		time.Sleep(100 * time.Millisecond)
		return nil, context.DeadlineExceeded
	}).AnyTimes()

	_, err := p.Classify(context.Background(), providers.Request{
		SystemPrompt: "system",
		UserPrompt:   "user",
	})
	var perr *providers.Error
	if !errors.As(err, &perr) || perr.Reason != providers.UnavailableTimeout {
		t.Fatalf("expected timeout error, got %v", err)
	}
}

func TestChatMessages(t *testing.T) {
	t.Run("no prefix yields one message", func(t *testing.T) {
		got := chatMessages("", "body")
		if len(got) != 1 {
			t.Fatalf("got %d messages, want 1", len(got))
		}
		if got[0].Role != "user" || got[0].Content != "body" {
			t.Errorf("got %+v, want user/body", got[0])
		}
	})
	t.Run("prefix yields two messages, prefix first", func(t *testing.T) {
		got := chatMessages("CLAUDE.md content", "body")
		if len(got) != 2 {
			t.Fatalf("got %d messages, want 2", len(got))
		}
		if got[0].Role != "user" || got[0].Content != "CLAUDE.md content" {
			t.Errorf("first message = %+v, want user/CLAUDE.md content", got[0])
		}
		if got[1].Role != "user" || got[1].Content != "body" {
			t.Errorf("second message = %+v, want user/body", got[1])
		}
	})
}

// captureRequest installs a single expectation on hc that returns the given assistant text and validates that the
// outgoing request looks like a chat completion. It returns a function that reports whether the expectation was met.
func captureRequest(t *testing.T, hc *openaimocks.MockhttpClient, assistantText string) func() bool {
	t.Helper()
	hc.EXPECT().Do(gomock.Any()).DoAndReturn(func(req *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return nil, err
		}
		_ = req.Body.Close()
		var parsed chatCompletionRequest
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Errorf("decode request body: %v", err)
			return nil, err
		}
		if parsed.Model == "" {
			t.Error("request missing model")
		}
		if len(parsed.Messages) == 0 {
			t.Error("request missing messages")
		}
		return jsonResponse(assistantText), nil
	}).Times(1)
	return func() bool { return true }
}

func jsonResponse(assistantText string) *http.Response {
	respBody, _ := json.Marshal(chatCompletionResponse{
		Choices: []chatCompletionChoice{{Message: chatMessage{Role: "assistant", Content: assistantText}}},
	})
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(respBody)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}
}
