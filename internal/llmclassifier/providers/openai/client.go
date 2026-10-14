package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// chatMessage is the wire format for the OpenAI chat completions messages array.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatCompletionRequest is the request body for POST /v1/chat/completions.
type chatCompletionRequest struct {
	Model          string        `json:"model"`
	Messages       []chatMessage `json:"messages"`
	MaxTokens      int32         `json:"max_tokens,omitempty"`
	Stop           []string      `json:"stop,omitempty"`
	Temperature    *float32      `json:"temperature,omitempty"`
	ResponseFormat *responseFmt  `json:"response_format,omitempty"`
}

// responseFmt selects plain text vs JSON output mode.
type responseFmt struct {
	Type string `json:"type"`
}

// chatCompletionChoice is one element of the choices array in the response.
type chatCompletionChoice struct {
	Message chatMessage `json:"message"`
}

// chatCompletionResponse is the response body from POST /v1/chat/completions.
type chatCompletionResponse struct {
	Choices []chatCompletionChoice `json:"choices"`
	Error   *apiError              `json:"error,omitempty"`
}

// apiError mirrors the standard OpenAI error envelope.
type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func (e *apiError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: %s", e.Code, e.Message)
	}
	return e.Message
}

// httpClient is the slice of the underlying HTTP transport used by [Provider]. Extracted as an interface so tests can
// swap in a mock. The real *http.Client satisfies it.
//
//go:generate go run go.uber.org/mock/mockgen@v0.6.0 -typed -source=client.go -destination=mocks/client_mock.go -package=mocks
type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// client is a tiny OpenAI chat-completions HTTP client.
type client struct {
	baseURL    string
	apiKey     string
	httpClient httpClient
}

// newClient constructs a client. It validates that baseURL and apiKey are non-empty.
func newClient(baseURL, apiKey string) (*client, error) {
	if baseURL == "" {
		return nil, fmt.Errorf("openai: base_url is required")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("openai: api_key is required (set %s or configure api_key inline)", defaultAPIKeyEnv)
	}
	return &client{
		baseURL:    baseURL,
		apiKey:     apiKey,
		httpClient: &http.Client{Timeout: 0}, // per-call timeout handled by context
	}, nil
}

// chat sends one chat-completion request and returns the assistant's text content.
func (c *client) chat(ctx context.Context, req chatCompletionRequest) (string, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("request failed after %dms: %w", int(time.Since(start)/time.Millisecond), err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Try to parse the standard error envelope, but fall back to the raw body.
		var errResp struct {
			Error *apiError `json:"error"`
		}
		if json.Unmarshal(respBody, &errResp) == nil && errResp.Error != nil {
			return "", fmt.Errorf("API error (status %d): %s", resp.StatusCode, errResp.Error.Error())
		}
		return "", fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var completion chatCompletionResponse
	if err := json.Unmarshal(respBody, &completion); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}
	if apiErr := completion.Error; apiErr != nil {
		return "", fmt.Errorf("API error: %s", apiErr.Error())
	}
	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("empty choices in response")
	}
	return completion.Choices[0].Message.Content, nil
}
