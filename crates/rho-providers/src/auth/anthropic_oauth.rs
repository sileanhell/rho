//! Claude Pro/Max subscription OAuth (PKCE), mirroring Claude Code's public
//! client flow.
//!
//! Anthropic's OAuth client redirects to `https://console.anthropic.com/oauth/code/callback`,
//! a page that displays a `code#state` pair, so there is no loopback listener:
//! the login shows the authorize URL and the user pastes the pair back.
//! Refresh tokens are single-use; a successful refresh must be persisted
//! before the old pair is replayed as `invalid_grant`.

use std::time::Duration;

use serde::Deserialize;

use crate::{
    credentials::AnthropicTokens,
    model::{ModelError, TransportError},
};

use super::loopback::{pkce_challenge, random_token};

pub const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const SCOPES: &str = "org:create_api_key user:profile user:inference";
/// platform.claude.com is the live token host; console.anthropic.com 404s but
/// is kept as a fallback, matching Claude Code.
const TOKEN_URLS: &[&str] = &[
    "https://platform.claude.com/v1/oauth/token",
    "https://console.anthropic.com/v1/oauth/token",
];
/// The token endpoints 429 requests whose UA starts with `claude-code/` (or
/// Mozilla); Claude Code sends bare axios there. Inference requests still
/// need the `claude-code/` UA.
const TOKEN_USER_AGENT: &str = "axios/1.7.9";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// OAuth `error` codes after which replaying the same refresh token can never
/// succeed; only a fresh login recovers.
const GRANT_DEAD_CODES: &[&str] = &["invalid_grant", "invalid_token", "refresh_token_reused"];

/// True for Anthropic OAuth/setup tokens (`sk-ant-oat*`, `eyJ` JWTs, `cc-`);
/// false for the Console API-key families `sk-ant-api*`, `sk-ant-usr*`,
/// `sk-ant-svc*` and `sk-ant-admin*` — they authenticate as `x-api-key`, so the
/// Claude Code identity must not ride along. Unknown `sk-ant-*` prefixes
/// classify as OAuth so prefixes Anthropic ships later keep working.
pub fn is_oauth_token(key: &str) -> bool {
    if key.is_empty()
        || key.starts_with("sk-ant-api")
        || key.starts_with("sk-ant-usr")
        || key.starts_with("sk-ant-svc")
        || key.starts_with("sk-ant-admin")
    {
        return false;
    }
    key.starts_with("sk-ant-") || key.starts_with("eyJ") || key.starts_with("cc-")
}

/// True when `base_url` is exactly `api.anthropic.com` (an empty base is the
/// native default).
pub fn is_native_anthropic_host(base_url: &str) -> bool {
    let text = base_url.trim().trim_end_matches('/').to_lowercase();
    if text.is_empty() {
        return true;
    }
    url::Url::parse(&text)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| host.eq_ignore_ascii_case("api.anthropic.com"))
}

/// Claude Code OAuth identity applies only on the native Anthropic host or
/// the `anthropic` provider itself. Third-party Anthropic-protocol endpoints
/// never get it (the headers 401/403 there).
pub fn anthropic_route_is_oauth(base_url: &str, credential: &str) -> bool {
    is_native_anthropic_host(base_url) && is_oauth_token(credential)
}

#[derive(Debug, thiserror::Error)]
pub enum AnthropicOAuthError {
    /// The token endpoint rejected the grant. `relogin_required` marks codes
    /// after which the same refresh token can never succeed.
    #[error("Anthropic token {what} failed: HTTP {status} {code}{detail}")]
    TokenEndpoint {
        status: http::StatusCode,
        code: String,
        detail: String,
        what: &'static str,
    },
    #[error("the Anthropic OAuth token request failed: {0}")]
    Request(#[source] TransportError),
    #[error("the pasted authorization code is invalid: {0}")]
    InvalidCode(String),
    #[error("the Anthropic OAuth token response was missing access_token")]
    MissingAccessToken,
}

impl AnthropicOAuthError {
    /// True when retrying the same grant cannot succeed (dead grant).
    pub fn relogin_required(&self) -> bool {
        match self {
            Self::TokenEndpoint { status, code, .. } => {
                matches!(status.as_u16(), 400 | 401) && GRANT_DEAD_CODES.contains(&code.as_str())
            }
            _ => false,
        }
    }
}

impl From<reqwest::Error> for AnthropicOAuthError {
    fn from(error: reqwest::Error) -> Self {
        Self::Request(TransportError::from_reqwest(error))
    }
}

/// A pending paste-code login. The authorize URL is ready immediately.
pub struct AnthropicBrowserLogin {
    pub authorize_url: String,
    verifier: String,
    state: String,
}

impl std::fmt::Debug for AnthropicBrowserLogin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicBrowserLogin")
            .field("authorize_url", &"[REDACTED]")
            .field("verifier", &"[REDACTED]")
            .field("state", &"[REDACTED]")
            .finish()
    }
}

pub fn start_anthropic_browser_login() -> AnthropicBrowserLogin {
    let verifier = random_token(64);
    let state = random_token(32);
    let challenge = pkce_challenge(&verifier);
    let mut url = url::Url::parse(AUTHORIZE_URL).expect("Anthropic authorize URL must be valid");
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", &state);
    AnthropicBrowserLogin {
        authorize_url: url.to_string(),
        verifier,
        state,
    }
}

/// Exchange the pasted `code#state` pair for tokens. A bare code (no `#`)
/// is accepted when the state cannot be verified against the page output,
/// matching how other CLIs accept the code alone.
pub async fn complete_anthropic_browser_login(
    login: AnthropicBrowserLogin,
    pasted: &str,
) -> Result<AnthropicTokens, AnthropicOAuthError> {
    let pasted = pasted.trim();
    let (code, state) = match pasted.split_once('#') {
        Some((code, state)) => (code.trim(), state.trim()),
        None => (pasted, ""),
    };
    if code.is_empty() {
        return Err(AnthropicOAuthError::InvalidCode("no code entered".into()));
    }
    if !state.is_empty() && state != login.state {
        return Err(AnthropicOAuthError::InvalidCode(
            "state mismatch; copy the full code shown after authorizing".into(),
        ));
    }
    let client = token_client()?;
    let result = post_token_request(
        &client,
        &TokenRequest::AuthorizationCode {
            code,
            state,
            verifier: &login.verifier,
        },
        "exchange",
    )
    .await?;
    token_state(result, "")
}

enum TokenRequest<'a> {
    AuthorizationCode {
        code: &'a str,
        state: &'a str,
        verifier: &'a str,
    },
    RefreshToken {
        refresh_token: &'a str,
    },
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
    error: Option<String>,
    error_description: Option<String>,
}

fn token_client() -> Result<reqwest::Client, AnthropicOAuthError> {
    crate::reqwest_client_builder()
        .timeout(REQUEST_TIMEOUT)
        .user_agent(TOKEN_USER_AGENT)
        .build()
        .map_err(AnthropicOAuthError::from)
}

/// POST to the token endpoints in order; raise the last error if all fail.
/// A dead grant is dead at every endpoint and is not replayed.
async fn post_token_request(
    client: &reqwest::Client,
    request: &TokenRequest<'_>,
    what: &'static str,
) -> Result<TokenResponse, AnthropicOAuthError> {
    let mut last_error = None;
    for endpoint in TOKEN_URLS {
        let request = with_fixed_fields(client.post(*endpoint), request);
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(AnthropicOAuthError::from(error));
                continue;
            }
        };
        let status = response.status();
        let body = response
            .json::<TokenResponse>()
            .await
            .unwrap_or(TokenResponse {
                access_token: None,
                refresh_token: None,
                expires_in: None,
                error: None,
                error_description: None,
            });
        if status.is_success() {
            return Ok(body);
        }
        let code = body.error.unwrap_or_default();
        let description = body.error_description.unwrap_or_default();
        let error = AnthropicOAuthError::TokenEndpoint {
            status,
            code,
            detail: if description.is_empty() {
                String::new()
            } else {
                format!(" ({description})")
            },
            what,
        };
        if error.relogin_required() {
            return Err(error);
        }
        last_error = Some(error);
    }
    Err(last_error.unwrap_or(AnthropicOAuthError::MissingAccessToken))
}

fn with_fixed_fields(
    builder: reqwest::RequestBuilder,
    request: &TokenRequest<'_>,
) -> reqwest::RequestBuilder {
    match request {
        TokenRequest::AuthorizationCode {
            code,
            state,
            verifier,
        } => builder.json(&serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        })),
        TokenRequest::RefreshToken { refresh_token } => builder.form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", CLIENT_ID),
        ]),
    }
}

fn token_state(
    result: TokenResponse,
    fallback_refresh_token: &str,
) -> Result<AnthropicTokens, AnthropicOAuthError> {
    let access_token = result
        .access_token
        .filter(|token| !token.is_empty())
        .ok_or(AnthropicOAuthError::MissingAccessToken)?;
    let refresh_token = result.refresh_token.or_else(|| {
        (!fallback_refresh_token.is_empty()).then(|| fallback_refresh_token.to_string())
    });
    let expires_at_ms = result
        .expires_in
        .map(|expires_in| now_unix_ms().saturating_add(expires_in.saturating_mul(1_000)));
    Ok(AnthropicTokens {
        access_token,
        refresh_token,
        expires_at_ms,
    })
}

/// Refresh an Anthropic OAuth token without mutating any credential store.
/// The returned refresh token falls back to the one presented when the
/// endpoint omits a replacement.
pub async fn refresh_anthropic_tokens(
    client: &reqwest::Client,
    refresh_token: &str,
) -> Result<AnthropicTokens, AnthropicOAuthError> {
    if refresh_token.is_empty() {
        return Err(AnthropicOAuthError::InvalidCode(
            "refresh_token is required".into(),
        ));
    }
    let result = post_token_request(
        client,
        &TokenRequest::RefreshToken { refresh_token },
        "refresh",
    )
    .await?;
    token_state(result, refresh_token)
}

pub(crate) fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .unwrap_or_default()
}

/// Claude Code's reported version for the OAuth `user-agent` header.
///
/// Anthropic rejects OAuth requests whose `claude-code/` version is too far
/// behind the real release, so the version is fetched live from the releases
/// endpoint on every process run; the pinned fallback serves only before the
/// first [`ensure_claude_code_version`] resolves.
pub fn claude_code_version() -> &'static str {
    const FALLBACK: &str = "2.1.74";
    REMOTE_VERSION
        .get()
        .and_then(|result| result.as_ref().ok())
        .map(String::as_str)
        .unwrap_or(FALLBACK)
}

/// The fetched `claude-code/` version, or the fetch error that disables the
/// OAuth provider for this process. A failure is cached too: retrying a dead
/// network every request would stall each turn instead of failing fast.
static REMOTE_VERSION: tokio::sync::OnceCell<Result<String, String>> =
    tokio::sync::OnceCell::const_new();

/// Gate every OAuth activation on the releases endpoint.
///
/// The fetch runs once per process (every program start re-fetches) and is
/// awaited inline so the caller — startup model refresh or the first
/// `/v1/messages` send — blocks until the version is known. Failure disables
/// the OAuth provider for the rest of the run.
pub(crate) async fn ensure_claude_code_version(
    client: &reqwest::Client,
) -> Result<&'static str, ModelError> {
    let result = REMOTE_VERSION
        .get_or_init(|| fetch_claude_code_version(client.clone()))
        .await;
    match result {
        Ok(version) => Ok(version.as_str()),
        Err(error) => Err(ModelError::InvalidResponse(format!(
            "could not fetch the Claude Code release version, Anthropic OAuth is unavailable: {error}"
        ))),
    }
}

async fn fetch_claude_code_version(client: reqwest::Client) -> Result<String, String> {
    const LATEST_URL: &str = "https://downloads.claude.ai/claude-code-releases/latest";
    let text = client
        .get(LATEST_URL)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .text()
        .await
        .map_err(|error| error.to_string())?;
    let version = text.split_whitespace().next().unwrap_or_default();
    if !version
        .chars()
        .next()
        .is_some_and(|first| first.is_ascii_digit())
    {
        return Err(format!("unexpected version payload from {LATEST_URL}"));
    }
    Ok(version.to_string())
}
