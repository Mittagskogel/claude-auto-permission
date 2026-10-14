package openai

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"buf.build/go/protovalidate"

	configpb "claude-auto-permission/internal/gen/config/v1"
)

// defaultAPIKeyEnv is the default environment variable name for the API key.
const defaultAPIKeyEnv = "CSCS_INFERENCE_API_KEY"

// FromProto translates a [configpb.LlmClassifierConfig] into a live OpenAI [Provider]. Field-level invariants declared
// on the proto via `(buf.validate)` are enforced here so each provider owns its config validation surface.
//
// Takes the *outer* classifier config (not the inner `OpenAIProvider`) so the timeout — declared at the outer level —
// can be threaded in without callers having to reach across the proto boundary themselves.
func FromProto(ctx context.Context, cfg *configpb.LlmClassifierConfig) (*Provider, error) {
	if cfg == nil {
		return nil, errors.New("openai: nil config")
	}
	if err := protovalidate.Validate(cfg); err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	op := cfg.GetOpenai()

	apiKey := op.GetApiKey()
	if apiKey == "" {
		apiKeyEnv := op.GetApiKeyEnv()
		if apiKeyEnv == "" {
			apiKeyEnv = defaultAPIKeyEnv
		}
		apiKey = os.Getenv(apiKeyEnv)
		if apiKey == "" {
			return nil, fmt.Errorf("openai: api_key unset and %s environment variable empty", apiKeyEnv)
		}
	}

	return New(ctx, Config{
		BaseURL:             op.GetBaseUrl(),
		APIKey:              apiKey,
		Model:               op.GetModel(),
		MaxTokens:           op.GetMaxTokens(),
		Timeout:             time.Duration(cfg.GetTimeoutMs()) * time.Millisecond,
		TwoStage:            twoStageMode(op.GetTwoStage()),
		Stage1MaxTokensBoth: op.GetStage1MaxTokensBoth(),
		Stage1MaxTokensFast: op.GetStage1MaxTokensFast(),
	})
}

// twoStageMode maps the proto enum to [TwoStageMode]. The enum numbers are kept in sync with the proto schema so
// callers can treat the cast as simple type punning. `loader.FillDefaults` applies the schema's `(opts).default`
// annotation, so by the time this runs `TWO_STAGE_MODE_UNSPECIFIED` should already have been mapped to `TWO_STAGE_MODE_BOTH`;
// the explicit case is defense in depth.
func twoStageMode(m configpb.OpenAIProvider_TwoStageMode) TwoStageMode {
	switch m {
	case configpb.OpenAIProvider_TWO_STAGE_MODE_OFF:
		return TwoStageOff
	case configpb.OpenAIProvider_TWO_STAGE_MODE_FAST:
		return TwoStageFast
	case configpb.OpenAIProvider_TWO_STAGE_MODE_BOTH:
		return TwoStageBoth
	case configpb.OpenAIProvider_TWO_STAGE_MODE_THINKING:
		return TwoStageThinking
	}
	return TwoStageBoth
}
