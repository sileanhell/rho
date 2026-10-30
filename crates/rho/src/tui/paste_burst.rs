use std::time::{Duration, Instant};

const PASTE_BURST_GAP: Duration = Duration::from_millis(12);
const PASTE_ENTER_SUPPRESSION: Duration = Duration::from_millis(120);
const PASTE_BURST_MIN_CHARS: usize = 2;
// Keep short multiline pastes editable in the composer; only larger pastes
// become atomic `[ pasted: N lines ]` markers.
pub(super) const PASTE_COLLAPSE_MIN_LINES: usize = 5;
const PASTE_COLLAPSE_MIN_CHARS: usize = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PasteBurstEnter {
    Buffered,
    InsertNewline,
    NotPaste,
}

#[derive(Default)]
pub(super) struct PasteBurst {
    pending_text: String,
    last_event_at: Option<Instant>,
    plain_char_count: usize,
    suppress_enter_until: Option<Instant>,
}

impl PasteBurst {
    pub(super) fn has_pending(&self) -> bool {
        !self.pending_text.is_empty()
    }

    pub(super) fn can_continue(&self, now: Instant) -> bool {
        if !self.has_pending() {
            return true;
        }

        self.last_event_at
            .is_some_and(|last| now.saturating_duration_since(last) <= PASTE_BURST_GAP)
    }

    pub(super) fn push_plain_char(&mut self, ch: char, now: Instant) {
        if !self.has_pending() {
            self.plain_char_count = 0;
            self.suppress_enter_until = None;
        }

        self.pending_text.push(ch);
        self.last_event_at = Some(now);
        self.plain_char_count = self.plain_char_count.saturating_add(1);
        if self.plain_char_count >= PASTE_BURST_MIN_CHARS {
            self.suppress_enter_until = now.checked_add(PASTE_ENTER_SUPPRESSION);
        }
    }

    pub(super) fn push_enter_if_paste(&mut self, now: Instant) -> PasteBurstEnter {
        let follows_pending_burst = self
            .last_event_at
            .is_some_and(|last| now.saturating_duration_since(last) <= PASTE_BURST_GAP)
            && self.plain_char_count > 0;
        let follows_plain_text_burst = self.suppresses_enter_at(now);
        if !follows_pending_burst && !follows_plain_text_burst {
            return PasteBurstEnter::NotPaste;
        }

        self.suppress_enter_until = now.checked_add(PASTE_ENTER_SUPPRESSION);
        if self.has_pending() {
            self.pending_text.push('\n');
            self.last_event_at = Some(now);
            PasteBurstEnter::Buffered
        } else {
            PasteBurstEnter::InsertNewline
        }
    }

    pub(super) fn is_due(&self, now: Instant) -> bool {
        self.deadline().is_some_and(|deadline| now >= deadline)
    }

    pub(super) fn poll_timeout(&self, now: Instant, idle_timeout: Duration) -> Duration {
        let Some(deadline) = self.deadline() else {
            return idle_timeout;
        };

        deadline
            .checked_duration_since(now)
            .unwrap_or_default()
            .min(idle_timeout)
    }

    pub(super) fn deadline(&self) -> Option<Instant> {
        self.last_event_at
            .and_then(|last| last.checked_add(PASTE_BURST_GAP))
    }

    pub(super) fn take_pending(&mut self) -> Option<String> {
        if self.pending_text.is_empty() {
            self.clear_pending_text();
            return None;
        }

        let text = std::mem::take(&mut self.pending_text);
        self.clear_pending_text();
        Some(text)
    }

    pub(super) fn clear(&mut self) {
        self.pending_text.clear();
        self.last_event_at = None;
        self.plain_char_count = 0;
        self.suppress_enter_until = None;
    }

    fn clear_pending_text(&mut self) {
        self.pending_text.clear();
        self.last_event_at = None;
        self.plain_char_count = 0;
    }

    fn suppresses_enter_at(&self, now: Instant) -> bool {
        self.suppress_enter_until
            .is_some_and(|deadline| now <= deadline)
    }
}

pub(super) fn previous_word_boundary(input: &str, cursor: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let mut index = cursor.min(chars.len());
    while index > 0 && chars[index - 1].is_whitespace() {
        index -= 1;
    }
    while index > 0 && !chars[index - 1].is_whitespace() {
        index -= 1;
    }
    index
}

pub(super) fn next_word_boundary(input: &str, cursor: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let mut index = cursor.min(chars.len());
    while index < chars.len() && chars[index].is_whitespace() {
        index += 1;
    }
    while index < chars.len() && !chars[index].is_whitespace() {
        index += 1;
    }
    index
}

/// Character range of the word (or whitespace run) under `index`.
///
/// Double-click selection uses the same whitespace split as arrow-word moves:
/// a contiguous non-whitespace token, or a contiguous whitespace run when the
/// pointer lands on space. An empty input or past-end empty buffer yields `0..0`.
pub(super) fn word_range_at(input: &str, index: usize) -> std::ops::Range<usize> {
    let chars: Vec<char> = input.chars().collect();
    if chars.is_empty() {
        return 0..0;
    }
    let len = chars.len();
    let probe = if index >= len { len - 1 } else { index };
    let class_is_whitespace = chars[probe].is_whitespace();
    let mut start = probe;
    while start > 0 && chars[start - 1].is_whitespace() == class_is_whitespace {
        start -= 1;
    }
    let mut end = probe + 1;
    while end < len && chars[end].is_whitespace() == class_is_whitespace {
        end += 1;
    }
    start..end
}

pub(super) fn normalize_paste(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// A paste large enough to collapse into an atomic composer marker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CollapsedPaste {
    count: usize,
    unit: &'static str,
}

impl CollapsedPaste {
    /// Marker text that replaces the paste content in the composer.
    pub(super) fn marker(&self) -> String {
        format!("[ pasted: {} {} ]", self.count, self.unit)
    }

    /// Toast confirming the collapse, since the content itself is hidden.
    pub(super) fn toast(&self) -> String {
        format!("pasted {} {}", self.count, self.unit)
    }
}

pub(super) fn collapsed_paste_for(text: &str) -> Option<CollapsedPaste> {
    let line_count = text.split('\n').count();
    if line_count >= PASTE_COLLAPSE_MIN_LINES {
        return Some(CollapsedPaste {
            count: line_count,
            unit: "lines",
        });
    }
    let char_count = text.chars().count();
    if char_count > PASTE_COLLAPSE_MIN_CHARS {
        return Some(CollapsedPaste {
            count: char_count,
            unit: "chars",
        });
    }
    None
}

pub(super) fn expand_paste_segments(input: &str, segments: &[super::PasteSegment]) -> String {
    if segments.is_empty() {
        return input.to_string();
    }

    let mut result = String::new();
    let mut cursor = 0;
    for segment in segments {
        if cursor > segment.start || segment.end() > input.chars().count() {
            continue;
        }
        result.extend(input.chars().skip(cursor).take(segment.start - cursor));
        result.push_str(&segment.content);
        cursor = segment.end();
    }
    result.extend(input.chars().skip(cursor));
    result
}
