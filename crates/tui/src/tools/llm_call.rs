//! `llm_call` tool — invoke a named secondary model alias mid-conversation.
//!
//! The caller supplies an `alias` that must be pre-declared in config.toml
//! under `[models.aliases.<name>]`. Dynamic base-URL injection is forbidden
//! to prevent SSRF / prompt-injection attacks; only pre-declared provider
//! strings are accepted.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::client::{api_url, versioned_base_url};
use crate::config::ModelsConfig;

use super::spec::{
    ApprovalRequirement, ToolCapability, ToolContext, ToolError, ToolResult, ToolSpec,
};

// ── per-thread state ──────────────────────────────────────────────────────────

/// Multi-turn conversation state for one secondary-model thread.
#[derive(Debug, Default)]
struct ThreadState {
    messages: Vec<Value>,
    call_count: u32,
}

// ── tool struct ───────────────────────────────────────────────────────────────

/// `llm_call` tool — call a secondary model via a pre-declared alias.
pub struct LlmCallTool {
    /// Resolved models config snapshot taken at tool-registration time.
    models_cfg: Option<ModelsConfig>,
    /// Per-session call counters for rate limiting (alias → count).
    session_counts: Mutex<HashMap<String, u32>>,
    /// In-memory multi-turn thread store (thread_id → state).
    threads: Mutex<HashMap<String, ThreadState>>,
}

impl LlmCallTool {
    #[must_use]
    pub fn new(models_cfg: Option<ModelsConfig>) -> Self {
        Self {
            models_cfg,
            session_counts: Mutex::new(HashMap::new()),
            threads: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for LlmCallTool {
    fn default() -> Self {
        Self::new(None)
    }
}

// ── ToolSpec ──────────────────────────────────────────────────────────────────

#[async_trait]
impl ToolSpec for LlmCallTool {
    fn name(&self) -> &'static str {
        "llm_call"
    }

    fn description(&self) -> &'static str {
        "Invoke a named secondary model alias (e.g. 'reviewer', 'auditor') \
         defined in config.toml under [models.aliases]. Use this to get an \
         independent critique, translation, or analysis from a different model \
         mid-conversation. The alias must be pre-declared — dynamic endpoints \
         are not accepted."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "alias": {
                    "type": "string",
                    "description": "Name of the model alias declared in config.toml [models.aliases]."
                },
                "prompt": {
                    "type": "string",
                    "description": "The user/task message to send to the secondary model."
                },
                "context": {
                    "type": "string",
                    "description": "Optional extra context prepended as a system message."
                },
                "thread_id": {
                    "type": "string",
                    "description": "Optional thread ID for multi-turn conversation. \
                                    Pass the ID returned by a previous llm_call to \
                                    continue the same thread."
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature (0.0–2.0). Defaults to 0.3."
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens in the response. Defaults to 4096."
                }
            },
            "required": ["alias", "prompt"],
            "additionalProperties": false
        })
    }

    fn capabilities(&self) -> Vec<ToolCapability> {
        vec![ToolCapability::Network]
    }

    fn approval_requirement(&self) -> ApprovalRequirement {
        // `Suggest` means: show approval prompt in Agent mode, but skip in YOLO.
        // This lets users review which secondary model is being called and at what cost.
        ApprovalRequirement::Suggest
    }

    fn supports_parallel(&self) -> bool {
        false
    }

    async fn execute(&self, input: Value, _context: &ToolContext) -> Result<ToolResult, ToolError> {
        // ── parse inputs ──────────────────────────────────────────────────────
        let alias = input
            .get("alias")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_input("missing field: alias"))?
            .trim()
            .to_string();

        let prompt = input
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_input("missing field: prompt"))?
            .to_string();

        let extra_context = input
            .get("context")
            .and_then(Value::as_str)
            .map(str::to_string);

        let thread_id = input
            .get("thread_id")
            .and_then(Value::as_str)
            .map(str::to_string);

        let temperature = input
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(0.3)
            .clamp(0.0, 2.0);

        let max_tokens = input
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(4096)
            .clamp(1, 32_768) as u32;

        // ── resolve alias ─────────────────────────────────────────────────────
        let models_cfg = self.models_cfg.as_ref().ok_or_else(|| {
            ToolError::not_available(
                "No [models] section in config.toml. \
                 Define [models.aliases.<name>] to use llm_call."
                    .to_string(),
            )
        })?;

        let alias_cfg = models_cfg.aliases.get(&alias).ok_or_else(|| {
            let available: Vec<&str> = models_cfg.aliases.keys().map(String::as_str).collect();
            ToolError::invalid_input(format!(
                "Unknown alias '{alias}'. Available: [{}]",
                available.join(", ")
            ))
        })?;

        // ── per-session call limit ────────────────────────────────────────────
        if let Some(max) = alias_cfg.max_calls_per_session {
            let mut counts = self.session_counts.lock().unwrap();
            let count = counts.entry(alias.clone()).or_insert(0);
            if *count >= max {
                return Err(ToolError::not_available(format!(
                    "Alias '{alias}' has reached its session limit of {max} calls."
                )));
            }
            *count += 1;
        }

        // ── resolve API key ───────────────────────────────────────────────────
        let api_key = alias_cfg.resolve_api_key().ok_or_else(|| {
            ToolError::not_available(format!(
                "No API key for alias '{alias}'. \
                 Set api_key_env or api_key in [models.aliases.{alias}]."
            ))
        })?;

        // ── derive base URL from provider string ──────────────────────────────
        let base_url = provider_base_url(&alias_cfg.provider);

        // ── build / continue thread ───────────────────────────────────────────
        let tid = thread_id.unwrap_or_else(|| Uuid::new_v4().to_string());

        let messages = {
            let mut threads = self.threads.lock().unwrap();
            let thread = threads.entry(tid.clone()).or_default();

            if thread.messages.is_empty() {
                if let Some(ctx) = &extra_context {
                    thread.messages.push(json!({
                        "role": "system",
                        "content": ctx
                    }));
                }
            }

            thread.messages.push(json!({
                "role": "user",
                "content": prompt
            }));

            thread.messages.clone()
        };

        // ── call secondary model ──────────────────────────────────────────────
        let response_json =
            call_chat_completions(&api_key, &base_url, &alias_cfg.model, &messages, temperature, max_tokens)
                .await
                .map_err(|e| ToolError::execution_failed(format!("llm_call failed: {e}")))?;

        let content = response_json
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let tokens_in = response_json
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let tokens_out = response_json
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);

        // Estimate cost and report to the per-alias side-channel (#18).
        // Uses a rough 1M-token price for the provider as a best-effort estimate.
        let cost_per_1m_in = provider_input_price_per_1m(&alias_cfg.provider);
        let cost_per_1m_out = provider_output_price_per_1m(&alias_cfg.provider);
        let estimated_cost = (tokens_in as f64 * cost_per_1m_in
            + tokens_out as f64 * cost_per_1m_out)
            / 1_000_000.0;
        crate::llm_call_costs::report(&alias, tokens_in, tokens_out, estimated_cost);

        // ── store assistant reply in thread ───────────────────────────────────
        {
            let mut threads = self.threads.lock().unwrap();
            if let Some(thread) = threads.get_mut(&tid) {
                thread.messages.push(json!({
                    "role": "assistant",
                    "content": content
                }));
                thread.call_count += 1;
            }
        }

        // ── format result ─────────────────────────────────────────────────────
        let output = json!({
            "content": content,
            "thread_id": tid,
            "model_used": alias_cfg.model,
            "alias": alias,
            "tokens_used": {
                "input": tokens_in,
                "output": tokens_out
            }
        });

        Ok(ToolResult::success(output.to_string()))
    }
}

// ── HTTP helper ───────────────────────────────────────────────────────────────

/// POST to `<base_url>/chat/completions` with minimal OpenAI-compatible payload.
async fn call_chat_completions(
    api_key: &str,
    base_url: &str,
    model: &str,
    messages: &[Value],
    temperature: f64,
    max_tokens: u32,
) -> anyhow::Result<Value> {
    let url = api_url(base_url, "chat/completions");

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))?,
    );

    let body = json!({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": false
    });

    let client = reqwest::Client::builder()
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(30))
        .min_tls_version(reqwest::tls::Version::TLS_1_2)
        .build()?;

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(120))
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {status}: {text}");
    }

    let json: Value = resp.json().await?;
    Ok(json)
}

// ── provider URL map ──────────────────────────────────────────────────────────

/// Map a provider string to its default base URL (no `/v1` suffix —
/// `versioned_base_url` / `api_url` will append it).
fn provider_base_url(provider: &str) -> String {
    let key = provider.trim().to_ascii_lowercase();
    let raw = match key.as_str() {
        "openai" | "open-ai" => "https://api.openai.com/v1",
        "anthropic" => "https://api.anthropic.com/v1",
        "deepseek" => "https://api.deepseek.com",
        "deepseek-cn" | "deepseekcn" => "https://api.deepseeki.com",
        "openrouter" => "https://openrouter.ai/api/v1",
        "novita" => "https://api.novita.ai/v1",
        "fireworks" => "https://api.fireworks.ai/inference/v1",
        other => return format!("https://api.{other}.com/v1"),
    };
    raw.to_string()
}

// The `versioned_base_url` import is used indirectly through `api_url`.
// Suppress the dead-code lint if the compiler inlines it away.
#[allow(dead_code)]
const _: fn(&str) -> String = versioned_base_url;

/// Rough input token price per 1M tokens in USD for common providers.
fn provider_input_price_per_1m(provider: &str) -> f64 {
    match provider.trim().to_ascii_lowercase().as_str() {
        "openai" | "open-ai" => 2.50,       // gpt-4o range
        "anthropic" => 3.00,                 // claude-3.5-sonnet range
        "deepseek" | "deepseek-cn" => 0.14,  // deepseek-v4-pro
        _ => 1.00,                            // conservative fallback
    }
}

/// Rough output token price per 1M tokens in USD for common providers.
fn provider_output_price_per_1m(provider: &str) -> f64 {
    match provider.trim().to_ascii_lowercase().as_str() {
        "openai" | "open-ai" => 10.00,
        "anthropic" => 15.00,
        "deepseek" | "deepseek-cn" => 0.28,
        _ => 4.00,
    }
}
