//! Per-conversation sticky request downgrades, learned from 400 responses.
//!
//! Hosted Anthropic Messages adapters and proxies reject optional fields a
//! first-party host accepts: `cache_control`, its `ttl`/`scope` leaves,
//! mid-conversation `role: system` markers, or named `anthropic-beta` values.
//! Retrying the same payload pays a second failure, so the first classified
//! refusal is latched for the conversation and every later request starts out
//! degraded — the same policy Claude Code applies through its resend ladder.
//!
//! State is keyed by the request's `prompt_cache_key` (the session identity),
//! capped like [`super::per_message_effort`]'s LRU. Requests without a key
//! share one bucket: a host refusal describes the host, not the conversation.

use std::collections::{BTreeSet, VecDeque};

/// Retained conversations per provider instance; see `per_message_effort`.
const MAX_TRACKED_CONVERSATIONS: usize = 1_024;

#[derive(Debug, Default)]
pub(super) struct ConversationLatches {
    conversations: VecDeque<(String, Latches)>,
    keyless: Latches,
}

/// Downgrades latched for one conversation.
#[derive(Clone, Debug, Default)]
pub(super) struct Latches {
    /// `cache_control` rejected outright: send no breakpoints at all.
    pub(super) cache_control_refused: bool,
    /// `cache_control.ttl` rejected: keep breakpoints, drop the TTL leaf.
    pub(super) ttl_refused: bool,
    /// `cache_control.scope` rejected: keep breakpoints, drop the scope leaf.
    pub(super) scope_refused: bool,
    /// Mid-conversation `role: system` rejected: fall back to top-level
    /// `output_config` effort (one prefix reset, then stable again).
    pub(super) mid_conv_system_refused: bool,
    /// Named `anthropic-beta` values the host refused; dropped from the header.
    pub(super) refused_betas: BTreeSet<&'static str>,
}

impl ConversationLatches {
    /// Latches for `conversation`, LRU-touched when keyed.
    pub(super) fn latches(&mut self, conversation: Option<&str>) -> &mut Latches {
        let Some(key) = conversation else {
            return &mut self.keyless;
        };
        if let Some(index) = self
            .conversations
            .iter()
            .position(|(existing, _)| existing == key)
        {
            let entry = self.conversations.remove(index).expect("positioned");
            self.conversations.push_back(entry);
        } else {
            if self.conversations.len() >= MAX_TRACKED_CONVERSATIONS {
                self.conversations.pop_front();
            }
            self.conversations
                .push_back((key.to_owned(), Latches::default()));
        }
        &mut self.conversations.back_mut().expect("just positioned").1
    }

    /// Applies `degrade`; returns `true` only when it newly latched. A `false`
    /// means the request already ran degraded and the error is unrelated to
    /// the latched field, so the caller surfaces it instead of retrying.
    pub(super) fn latch(&mut self, conversation: Option<&str>, degrade: Degrade) -> bool {
        let latches = self.latches(conversation);
        match degrade {
            Degrade::StripCacheControl => latch_once(&mut latches.cache_control_refused),
            Degrade::DropTtl => latch_once(&mut latches.ttl_refused),
            Degrade::DropScope => latch_once(&mut latches.scope_refused),
            Degrade::MidConvSystem => latch_once(&mut latches.mid_conv_system_refused),
            Degrade::DropBeta(beta) => {
                let mut changed = latches.refused_betas.insert(beta);
                if beta == super::per_message_effort::BETA {
                    // The effort markers are dead weight without their beta.
                    changed |= latch_once(&mut latches.mid_conv_system_refused);
                }
                changed
            }
        }
    }
}

fn latch_once(flag: &mut bool) -> bool {
    let fresh = !*flag;
    *flag = true;
    fresh
}

/// A downgrade a classified 400 asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Degrade {
    StripCacheControl,
    DropTtl,
    DropScope,
    MidConvSystem,
    DropBeta(&'static str),
}

/// Words a proxy or gateway uses when it refuses an unknown field. Kept loose
/// on purpose: a false negative just costs one more rejected request, while a
/// false positive permanently drops a feature on this conversation.
const REFUSAL_WORDS: &[&str] = &[
    "not supported",
    "unsupported",
    "unknown",
    "unrecognized",
    "unrecognised",
    "unexpected",
    "not permitted",
    "not allowed",
    "cannot be set",
    "extra field",
    "additional propert",
    "extraneous",
    "invalid",
    "must be",
];

fn mentions_refusal(body: &str, term: &str) -> bool {
    body.contains(term) && REFUSAL_WORDS.iter().any(|word| body.contains(word))
}

/// Maps a 4xx error body to the downgrade it asks for. `sent_betas` are the
/// `anthropic-beta` values the failed attempt carried, so a named refusal can
/// drop exactly that beta.
///
/// Order matters: `ttl`/`scope` refusals also name `cache_control`, so the
/// more specific leaf is matched before the generic strip. The
/// mid-conversation `role: system` refusal is checked before `cache_control`
/// only because its message never mentions the field.
pub(super) fn classify_bad_request(body: &str, sent_betas: &[&'static str]) -> Option<Degrade> {
    let body = body.to_ascii_lowercase();
    if mentions_refusal(&body, "cache_control") || mentions_refusal(&body, "cache-control") {
        if mentions_refusal(&body, "ttl") || body.contains("\"1h\"") {
            return Some(Degrade::DropTtl);
        }
        if mentions_refusal(&body, "scope") {
            return Some(Degrade::DropScope);
        }
        return Some(Degrade::StripCacheControl);
    }
    // Proxies that strip the field name from the message still tend to echo
    // the offending value: a `"1h"` complaint is the TTL leaf.
    if (body.contains("\"1h\"") || body.contains("ttl"))
        && REFUSAL_WORDS.iter().any(|word| body.contains(word))
    {
        return Some(Degrade::DropTtl);
    }
    // Gateways built for OpenAI-shaped payloads reject `role: "system"` inside
    // `messages` or the `output_config` leaf the effort markers carry. The
    // message need not quote the role value: "role must be user or assistant"
    // still names `role` and `system` next to a refusal word.
    if mentions_refusal(&body, "output_config")
        || mentions_refusal(&body, "mid-conversation")
        || (body.contains("role")
            && body.contains("system")
            && REFUSAL_WORDS.iter().any(|word| body.contains(word)))
    {
        return Some(Degrade::MidConvSystem);
    }
    // A refusal naming a sent beta drops exactly that one; an `anthropic-beta`
    // complaint naming nothing drops every optional beta sent this attempt.
    if mentions_refusal(&body, "beta") || mentions_refusal(&body, "anthropic-beta") {
        let named = sent_betas
            .iter()
            .copied()
            .filter(|beta| body.contains(beta))
            .next();
        if let Some(beta) = named {
            return Some(Degrade::DropBeta(beta));
        }
        return sent_betas
            .iter()
            .copied()
            .find(|beta| OPTIONAL_BETAS.contains(beta))
            .map(Degrade::DropBeta);
    }
    None
}

/// Betas the request may safely lose: dropping them degrades a feature, while
/// [`super::COMMON_BETAS`] and [`super::OAUTH_BETAS`] are structural.
const OPTIONAL_BETAS: &[&str] = &[super::FAST_MODE_BETA, super::per_message_effort::BETA];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_cache_control_leaves() {
        assert_eq!(
            classify_bad_request("unexpected field ttl in cache_control", &[]),
            Some(Degrade::DropTtl)
        );
        assert_eq!(
            classify_bad_request("cache_control scope is not permitted", &[]),
            Some(Degrade::DropScope)
        );
        assert_eq!(
            classify_bad_request("unknown field cache_control", &[]),
            Some(Degrade::StripCacheControl)
        );
        // A bare "1h" complaint without the field name is still the TTL leaf.
        assert_eq!(
            classify_bad_request("invalid enum value \"1h\"", &[]),
            Some(Degrade::DropTtl)
        );
    }

    #[test]
    fn classifies_mid_conversation_system() {
        assert_eq!(
            classify_bad_request(
                "messages[3].role: role must be user or assistant, not system",
                &[]
            ),
            Some(Degrade::MidConvSystem)
        );
        assert_eq!(
            classify_bad_request("unsupported field output_config", &[]),
            Some(Degrade::MidConvSystem)
        );
    }

    #[test]
    fn classifies_beta_refusals() {
        assert_eq!(
            classify_bad_request(
                "anthropic-beta value fast-mode-2026-02-01 is not supported",
                &[
                    super::super::FAST_MODE_BETA,
                    "interleaved-thinking-2025-05-14"
                ],
            ),
            Some(Degrade::DropBeta(super::super::FAST_MODE_BETA))
        );
        // No beta named: drop the first optional one actually sent.
        assert_eq!(
            classify_bad_request(
                "unsupported anthropic-beta header",
                &[
                    "interleaved-thinking-2025-05-14",
                    super::super::per_message_effort::BETA
                ],
            ),
            Some(Degrade::DropBeta(super::super::per_message_effort::BETA))
        );
        // Structural betas are never dropped: the retry would fail the same way.
        assert_eq!(
            classify_bad_request(
                "unsupported anthropic-beta header",
                &["claude-code-20250219"],
            ),
            None
        );
    }

    #[test]
    fn unrelated_errors_do_not_degrade() {
        assert_eq!(classify_bad_request("invalid api key", &[]), None);
        assert_eq!(classify_bad_request("credit balance too low", &[]), None);
    }

    #[test]
    fn latch_is_sticky_per_conversation() {
        let mut latches = ConversationLatches::default();
        assert!(latches.latch(Some("a"), Degrade::StripCacheControl));
        // A second refusal of the same field is not fresh: the caller must
        // surface it rather than retry forever.
        assert!(!latches.latch(Some("a"), Degrade::StripCacheControl));
        assert!(latches.latches(Some("a")).cache_control_refused);
        assert!(!latches.latches(Some("b")).cache_control_refused);
    }

    #[test]
    fn dropping_the_effort_beta_also_latches_markers() {
        let mut latches = ConversationLatches::default();
        assert!(latches.latch(
            None,
            Degrade::DropBeta(super::super::per_message_effort::BETA)
        ));
        let latches = latches.latches(None);
        assert!(latches.mid_conv_system_refused);
        assert!(latches
            .refused_betas
            .contains(super::super::per_message_effort::BETA));
    }
}
