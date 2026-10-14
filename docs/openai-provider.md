# OpenAI-compatible provider for claude-auto-permission

This branch adds an OpenAI-compatible classifier provider to `claude-auto-permission`, enabling use with inference endpoints like the **CSCS Inference API** (`https://api.inference.cscs.ch/v1`) in addition to the existing AWS Bedrock provider.

## What changed

- New provider package: `internal/llmclassifier/providers/openai/`
  - `config.go` — provider configuration
  - `client.go` — minimal OpenAI chat-completions HTTP client
  - `openai.go` — `providers.Provider` implementation
  - `from_proto.go` — config translation and validation
  - `openai_test.go` — unit tests
  - `mocks/client_mock.go` — generated HTTP client mock
- Proto schema update in `proto/config/v1/config.proto`
  - Added `OpenAIProvider` message
  - Added `openai` field to `LlmClassifierConfig`
  - Relaxed `bedrock` from `required` to one of multiple provider options
- Factory dispatch in `internal/llmclassifier/factory.go`
  - Added `cfg.GetOpenai()` case
- Regenerated Go protobuf bindings (`internal/gen/config/v1/config.pb.go`)
- New starter config: `scripts/starter-config-openai.txtpb`
- Factory tests extended to cover the new provider

## Supported classification modes

The OpenAI provider mirrors Bedrock's four strategies:

| Mode | Behavior |
|---|---|
| `TWO_STAGE_MODE_BOTH` (default) | Stage 1 XML prefilter with `</block>` stop sequence; stage 2 chain-of-thought on block |
| `TWO_STAGE_MODE_FAST` | Single XML verdict |
| `TWO_STAGE_MODE_OFF` | Single JSON-mode call using the classifier schema |
| `TWO_STAGE_MODE_THINKING` | Single XML chain-of-thought call |

> Note: OpenAI-compatible endpoints do not support Bedrock's native prompt cache points, so stage-2 calls do not benefit from prefix caching. Latency is otherwise equivalent.

## Configuration example

```textproto
projects {
  path_patterns: "/**"
  static_bash_rules {
    allow_write_patterns: "/tmp/**"
    use_default_rules {}
  }
  llm_classifier {
    enabled: true
    mode: CLASSIFIER_MODE_BLOCK_ONLY
    openai {
      base_url: "https://api.inference.cscs.ch/v1"
      # Gemma 4 31B with JSON-mode structured output gives the best speed/reliability
      # trade-off on CSCS for the classifier prompt.
      model: "google/gemma-4-31B-it"
      two_stage: TWO_STAGE_MODE_OFF
    }
    timeout_ms: 20000
    log_decisions: true
  }
}

runtime {
  log_decisions: true
}
```

Authentication resolves from the `CSCS_INFERENCE_API_KEY` environment variable by default. You can override the env var name with `api_key_env`, or set `api_key` inline (not recommended).

> **Recommended ramp:** start with `CLASSIFIER_MODE_BLOCK_ONLY` and monitor the decision log for false positives. Once you trust the model's verdicts, switch to `CLASSIFIER_MODE_FULL_AUTO` to reduce permission prompts on harmless non-edit commands.

## Hook registration

Pass the API key through to the hook subprocess in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "CSCS_INFERENCE_API_KEY=$CSCS_INFERENCE_API_KEY claude-auto-permission"
          }
        ]
      }
    ]
  }
}
```

## Build and test

```bash
ml go
export PATH=$HOME/go/bin:$PATH
make proto   # only needed if you edit proto/config/v1/config.proto
go test ./...
```

## Choosing a model

The classifier prompt is tuned for instruction-following models that can emit XML or JSON reliably. On CSCS, the best starting point is:

- **`google/gemma-4-31B-it`** with `two_stage: TWO_STAGE_MODE_OFF` — best speed/reliability trade-off in testing.

Other models available on the endpoint may work, but `swiss-ai/Apertus-v1.5-70B` (the 70B non-instruct Apertus model) in our testing produced XML/stop-sequence artifacts and hallucinated false positives such as blocking `wget example.com` with a reason referencing `rm -rf /`. `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` is larger and slower. Smaller models (`swiss-ai/Apertus-8B-*`) may be too unreliable for safety classification.

Test with `runtime.dump_llm_classifier: true` and `runtime.log_decisions: true` to evaluate verdict quality before trusting auto-approval.
