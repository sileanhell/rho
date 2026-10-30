use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::{credentials::GitHubCopilotTokens, model::TransportError};

use super::device_code::{
    poll_device_token, standard_device_poll_step, DeviceCodeResponse, DevicePollHttp,
    DevicePollOutcome, DevicePollRequest, DevicePollStep, FirstPoll,
};

const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const DEFAULT_SCOPE: &str = "read:user";
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct GitHubCopilotDeviceLogin {
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: Option<String>,
    pub expires_in: Duration,
    device_code: String,
    interval: Duration,
}

impl std::fmt::Debug for GitHubCopilotDeviceLogin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GitHubCopilotDeviceLogin")
            .field("user_code", &"[REDACTED]")
            .field("verification_uri", &self.verification_uri)
            .field("verification_uri_complete", &"[REDACTED]")
            .field("expires_in", &self.expires_in)
            .field("device_code", &"[REDACTED]")
            .field("interval", &self.interval)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GitHubCopilotDeviceError {
    #[error("GitHub Copilot device login setup failed: {0}")]
    Setup(String),
    #[error("GitHub Copilot device login failed: {0}")]
    OAuthDenied(String),
    #[error("GitHub Copilot device login request failed: {0}")]
    Request(#[source] TransportError),
    #[error("GitHub Copilot device login response was missing {0}")]
    MissingField(&'static str),
    #[error("timed out waiting for GitHub Copilot device login")]
    Timeout,
}

impl From<reqwest::Error> for GitHubCopilotDeviceError {
    fn from(error: reqwest::Error) -> Self {
        Self::Request(TransportError::from_reqwest(error))
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
    error: Option<String>,
    error_description: Option<String>,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct GitHubCopilotGithubToken {
    pub(crate) access_token: String,
    pub(crate) refresh_token: Option<String>,
    pub(crate) expires_at_unix: Option<i64>,
}

impl std::fmt::Debug for GitHubCopilotGithubToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GitHubCopilotGithubToken")
            .field("credentials", &"[REDACTED]")
            .field("expires_at_unix", &self.expires_at_unix)
            .finish()
    }
}

pub async fn start_github_copilot_device_login(
) -> Result<GitHubCopilotDeviceLogin, GitHubCopilotDeviceError> {
    start_github_copilot_device_login_with_endpoint(&http_client()?, DEVICE_CODE_URL).await
}

fn http_client() -> Result<reqwest::Client, GitHubCopilotDeviceError> {
    crate::reqwest_client_builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(GitHubCopilotDeviceError::from)
}

async fn start_github_copilot_device_login_with_endpoint(
    client: &reqwest::Client,
    endpoint: &str,
) -> Result<GitHubCopilotDeviceLogin, GitHubCopilotDeviceError> {
    let response: DeviceCodeResponse = client
        .post(endpoint)
        .header("Accept", "application/json")
        .header("User-Agent", crate::rho_user_agent())
        .form(&[("client_id", CLIENT_ID), ("scope", DEFAULT_SCOPE)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    if let Some(error) = response.error {
        return Err(GitHubCopilotDeviceError::Setup(
            response.error_description.unwrap_or(error),
        ));
    }

    Ok(GitHubCopilotDeviceLogin {
        device_code: required(response.device_code, "device_code")?,
        user_code: required(response.user_code, "user_code")?,
        verification_uri: required(response.verification_uri, "verification_uri")?,
        verification_uri_complete: response.verification_uri_complete,
        expires_in: Duration::from_secs(required(response.expires_in, "expires_in")?),
        interval: response
            .interval
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_POLL_INTERVAL),
    })
}

fn required<T>(value: Option<T>, field: &'static str) -> Result<T, GitHubCopilotDeviceError> {
    value.ok_or(GitHubCopilotDeviceError::MissingField(field))
}

pub async fn complete_github_copilot_device_login(
    login: GitHubCopilotDeviceLogin,
) -> Result<GitHubCopilotTokens, GitHubCopilotDeviceError> {
    complete_github_copilot_device_login_with_endpoint(&http_client()?, login, TOKEN_URL).await
}

async fn complete_github_copilot_device_login_with_endpoint(
    client: &reqwest::Client,
    login: GitHubCopilotDeviceLogin,
    endpoint: &str,
) -> Result<GitHubCopilotTokens, GitHubCopilotDeviceError> {
    let form = [
        ("client_id", CLIENT_ID),
        ("device_code", login.device_code.as_str()),
        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
    ];
    let user_agent = crate::rho_user_agent();
    let extra_headers = [
        ("Accept", "application/json"),
        ("User-Agent", user_agent.as_str()),
    ];
    match poll_device_token(
        DevicePollRequest {
            client,
            endpoint,
            form: &form,
            extra_headers: &extra_headers,
            expires_in: login.expires_in,
            interval: login.interval,
            first_poll: FirstPoll::AfterInterval,
            http: DevicePollHttp::ErrorForStatus,
        },
        |_status, body: TokenResponse| {
            if body.access_token.is_some() {
                return DevicePollStep::Tokens(body);
            }
            if let Some(step) = standard_device_poll_step(body.error.as_deref()) {
                return step;
            }
            match body.error.as_deref() {
                Some("access_denied") => DevicePollStep::Denied {
                    description: body
                        .error_description
                        .unwrap_or_else(|| "access denied".into()),
                },
                Some(error) => DevicePollStep::Denied {
                    description: body.error_description.unwrap_or_else(|| error.into()),
                },
                None => DevicePollStep::Fatal {
                    error: "access_token".into(),
                },
            }
        },
    )
    .await?
    {
        DevicePollOutcome::Tokens(response) => Ok(GitHubCopilotTokens {
            github_access_token: response
                .access_token
                .ok_or(GitHubCopilotDeviceError::MissingField("access_token"))?,
            github_refresh_token: response.refresh_token,
            github_expires_at_unix: response
                .expires_in
                .map(|seconds| now_unix_seconds() + seconds),
            copilot_token: None,
            copilot_expires_at_unix: None,
            copilot_refresh_after_unix: None,
            copilot_token_endpoint: None,
            copilot_chat_endpoint: None,
            copilot_models_endpoint: None,
        }),
        DevicePollOutcome::Denied { description } => {
            Err(GitHubCopilotDeviceError::OAuthDenied(description))
        }
        DevicePollOutcome::Timeout => Err(GitHubCopilotDeviceError::Timeout),
        DevicePollOutcome::Fatal { error } if error == "access_token" => {
            Err(GitHubCopilotDeviceError::MissingField("access_token"))
        }
        DevicePollOutcome::Fatal { error } => Err(GitHubCopilotDeviceError::OAuthDenied(error)),
    }
}

pub(crate) async fn refresh_github_copilot_github_token(
    client: &reqwest::Client,
    refresh_token: &str,
) -> Result<GitHubCopilotGithubToken, GitHubCopilotDeviceError> {
    refresh_github_copilot_github_token_with_endpoint(client, refresh_token, TOKEN_URL).await
}

async fn refresh_github_copilot_github_token_with_endpoint(
    client: &reqwest::Client,
    refresh_token: &str,
    endpoint: &str,
) -> Result<GitHubCopilotGithubToken, GitHubCopilotDeviceError> {
    let response: TokenResponse = client
        .post(endpoint)
        .header("Accept", "application/json")
        .header("User-Agent", crate::rho_user_agent())
        .form(&[
            ("client_id", CLIENT_ID),
            ("refresh_token", refresh_token),
            ("grant_type", "refresh_token"),
        ])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    if let Some(error) = response.error {
        return Err(GitHubCopilotDeviceError::OAuthDenied(
            response.error_description.unwrap_or(error),
        ));
    }

    Ok(GitHubCopilotGithubToken {
        access_token: required(response.access_token, "access_token")?,
        refresh_token: response.refresh_token,
        expires_at_unix: response
            .expires_in
            .map(|seconds| now_unix_seconds() + seconds),
    })
}

fn now_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}
