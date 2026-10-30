//! External CLI session adapter over the shared run-artifact sink.
//!
//! Translates stream-json effects into the generic status/attachment contract.

use std::path::PathBuf;

use tokio::sync::watch;

use crate::{
    run_artifacts::{AttachmentEvent, LiveRunTitle, RunArtifactIdentity, RunArtifactSink},
    subagent::RunStatus,
};

use super::stream_effect::{StreamEffect, TerminalResult};
use super::stream_format::apply_status_patch;

/// Starting activity, program name, and model-facing copy for a CLI runtime sink.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RuntimeLabel {
    pub(crate) starting_activity: &'static str,
    pub(crate) program: &'static str,
    /// Binary invoked for `--resume <session-id>`. `None` when the recorded
    /// session id cannot be resumed (ACP session ids are not CLI resume ids).
    pub(crate) resume_command: Option<&'static str>,
    /// Prefix for session metadata, e.g. `claude session`.
    pub(crate) session_label: &'static str,
    /// Cost line label, e.g. `claude cost`.
    pub(crate) cost_label: &'static str,
}

/// Thin CLI-facing handle around [`RunArtifactSink`].
pub(crate) struct StatusSink {
    inner: RunArtifactSink,
}

impl StatusSink {
    pub(crate) fn new(
        path: PathBuf,
        identity: &RunArtifactIdentity,
        prompt: &str,
        status_tx: Option<watch::Sender<RunStatus>>,
        label: RuntimeLabel,
    ) -> anyhow::Result<Self> {
        let mut inner = RunArtifactSink::open(path, identity, prompt, status_tx)?;
        inner.status.last_activity = Some(label.starting_activity.into());
        inner.publish();
        Ok(Self { inner })
    }

    /// Resume after the executor already wrote the Starting boundary.
    pub(crate) fn continue_from(
        path: PathBuf,
        mut status: RunStatus,
        prompt: &str,
        status_tx: Option<watch::Sender<RunStatus>>,
        live_title: Option<LiveRunTitle>,
        label: RuntimeLabel,
    ) -> anyhow::Result<Self> {
        status.last_activity = Some(label.starting_activity.into());
        let mut inner =
            RunArtifactSink::continue_from(path, status, prompt, status_tx, live_title)?;
        inner.publish();
        Ok(Self { inner })
    }

    pub(crate) fn mark_running(&mut self) {
        self.inner.mark_running("running");
    }

    pub(crate) fn apply_effect(&mut self, effect: StreamEffect) {
        match effect {
            StreamEffect::Attachment(event) => {
                // Answer text reaches `last_text` through its status patch.
                // Reasoning has no text patch, so mirror only that here.
                if let AttachmentEvent::ReasoningDelta(text) = &event {
                    if !text.is_empty() {
                        self.inner.append_last_text(text);
                    }
                }
                self.inner.record_attachment(event);
            }
            StreamEffect::Status(patch) => {
                apply_status_patch(&mut self.inner.status, patch);
                self.inner.publish();
            }
        }
    }

    pub(crate) async fn fail(&mut self, error: impl Into<String>) {
        self.inner.finish_error(error);
    }

    pub(crate) async fn stop(&mut self, reason: &str, pending: Option<&TerminalResult>) {
        if let Some(terminal) = pending {
            apply_terminal_metadata(&mut self.inner.status, terminal);
        }
        self.inner.finish_stopped(reason);
    }

    pub(crate) async fn finalize_success_from_stream(&mut self, terminal: &TerminalResult) {
        apply_terminal_metadata(&mut self.inner.status, terminal);
        self.inner.finish_ok(terminal.result_text.clone());
    }

    pub(crate) async fn finalize_failure_from_stream(
        &mut self,
        terminal: Option<&TerminalResult>,
        detail: String,
        prefer_detail: bool,
    ) {
        if let Some(terminal) = terminal {
            apply_terminal_metadata(&mut self.inner.status, terminal);
        }
        let error = if prefer_detail {
            detail
        } else {
            terminal
                .and_then(|terminal| {
                    terminal
                        .error
                        .clone()
                        .filter(|text| !text.is_empty())
                        .or_else(|| terminal.result_text.clone())
                })
                .unwrap_or(detail)
        };
        self.inner.finish_error(error);
    }
}

fn apply_terminal_metadata(status: &mut RunStatus, terminal: &TerminalResult) {
    if let Some(session_id) = terminal.session_id.clone() {
        status.claude_session_id = Some(session_id);
    }
    if let Some(turns) = terminal.num_turns {
        status.turns = turns;
    }
    if let Some(usage) = &terminal.usage {
        if let Some(tokens) = usage.inclusive_prompt_tokens() {
            status.input_tokens = Some(tokens);
        }
        if let Some(tokens) = usage.output_tokens {
            status.output_tokens = Some(tokens);
        }
    }
    if let Some(cost) = terminal.total_cost_usd {
        status.total_cost_usd = Some(cost);
    }
    if let Some(result) = terminal.result_text.clone() {
        status.result = Some(result);
    }
    if let Some(error) = terminal.error.clone() {
        status.error = Some(error);
    }
}
