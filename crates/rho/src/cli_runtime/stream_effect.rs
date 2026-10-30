//! Stream-json effect types shared by CLI mappers and the session sink.
//!
//! Protocol-neutral presentation and terminal contracts.

use rho_sdk::model::{ContextUsage, ModelUsage};

use crate::{run_artifacts::AttachmentEvent, subagent::RunState};

/// Soft cap for a single assistant text or reasoning delta retained/displayed.
pub(crate) const MAX_TEXT_DELTA_CHARS: usize = 32 * 1024;

/// Soft cap for terminal `result` text retained on status/attachments.
pub(crate) const MAX_RESULT_CHARS: usize = 64 * 1024;

/// Soft cap for tool input/result payload text shown in display lines.
pub(crate) const MAX_TOOL_PAYLOAD_CHARS: usize = 16 * 1024;

/// One presentation or status update produced from a stream-json line.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum StreamEffect {
    Attachment(AttachmentEvent),
    Status(StatusPatch),
}

/// Incremental status fields. Terminal Ok/Error is not applied here from a
/// `result` message; session combines [`TerminalResult`] with process exit.
/// `last_activity` may say "result received" while the run is still pending.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct StatusPatch {
    pub(crate) state: Option<RunState>,
    pub(crate) turns: Option<u64>,
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) last_activity: Option<String>,
    pub(crate) append_text: Option<String>,
    pub(crate) result: Option<String>,
    pub(crate) error: Option<String>,
    // NEXT_MAJOR(result.json): rename claude_session_id/claude_model to runtime_session_id/runtime_model; readers branch on runtime.
    pub(crate) claude_session_id: Option<String>,
    /// Concrete model Claude reported running, from the `init` frame. Rho
    /// passes `--model` through untouched, so this is the only report of what
    /// an alias such as `opus` actually resolved to.
    // NEXT_MAJOR(result.json): rename claude_session_id/claude_model to runtime_session_id/runtime_model; readers branch on runtime.
    pub(crate) claude_model: Option<String>,
    pub(crate) total_cost_usd: Option<f64>,
}

/// Explicit terminal classification. Missing `subtype` / `is_error` is never
/// treated as success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TerminalClassification {
    Success {
        subtype: String,
    },
    Failure {
        subtype: String,
        is_error: bool,
    },
    /// Schema drift or incomplete terminal fields.
    Invalid {
        reason: String,
    },
}

#[cfg(test)]
impl TerminalClassification {
    pub(crate) fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    pub(crate) fn is_failure(&self) -> bool {
        matches!(self, Self::Failure { .. })
    }

    pub(crate) fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid { .. })
    }
}

/// Terminal `result` payload for the session layer to combine with child exit.
///
/// Session must still combine this with process exit and must not treat stream
/// success alone as final run success when exit fails.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TerminalResult {
    pub(crate) classification: TerminalClassification,
    pub(crate) result_text: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) num_turns: Option<u64>,
    pub(crate) usage: Option<ModelUsage>,
    pub(crate) context: Option<ContextUsage>,
    pub(crate) total_cost_usd: Option<f64>,
    pub(crate) permission_denials: Vec<String>,
    pub(crate) stop_reason: Option<String>,
}
