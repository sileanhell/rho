//! Optional event-loop timing dump, enabled by `RHO_PERF=<path>`.
//!
//! Off by default; when set, each loop iteration appends one line of
//! `<tag>=<millis>` fields so a lagging session shows which phase ate the
//! frame budget. Used for diagnosing intermittent startup stalls.

use std::{fs::File, io::Write, time::Instant};

pub(super) struct PerfLog {
    file: Option<File>,
    start: Instant,
}

/// Accumulates one iteration's phase timings; flushed at the loop bottom.
pub(super) struct IterTiming {
    fields: String,
    t0: Instant,
    last: Instant,
    enabled: bool,
}

impl PerfLog {
    pub(super) fn from_env() -> Self {
        let file = std::env::var_os("RHO_PERF")
            .filter(|path| !path.is_empty())
            .and_then(|path| File::create(path).ok());
        Self {
            file,
            start: Instant::now(),
        }
    }

    pub(super) fn begin_iter(&self) -> IterTiming {
        IterTiming {
            fields: String::new(),
            t0: Instant::now(),
            last: Instant::now(),
            enabled: self.file.is_some(),
        }
    }

    fn write_line(&self, text: &str) {
        if let Some(file) = self.file.as_ref() {
            let mut file = file;
            let _ = writeln!(file, "{:>9} {text}", self.start.elapsed().as_millis());
            let _ = file.flush();
        }
    }

    pub(super) fn finish_iter(&self, iter: IterTiming) {
        if !iter.enabled {
            return;
        }
        self.write_line(&format!(
            "iter total={} {}",
            iter.t0.elapsed().as_millis(),
            iter.fields
        ));
    }

    pub(super) fn finish_named(&self, tag: &str, iter: IterTiming) {
        if !iter.enabled {
            return;
        }
        self.write_line(&format!(
            "{tag} total={} {}",
            iter.t0.elapsed().as_millis(),
            iter.fields
        ));
    }
}

impl IterTiming {
    /// Millis since the previous mark (or iteration start).
    pub(super) fn mark(&mut self, tag: &str) {
        if !self.enabled {
            return;
        }
        let elapsed = self.last.elapsed().as_millis();
        self.last = Instant::now();
        self.fields.push_str(tag);
        self.fields.push('=');
        self.fields.push_str(&elapsed.to_string());
        self.fields.push(' ');
    }

    pub(super) fn field(&mut self, tag: &str, value: impl std::fmt::Display) {
        if !self.enabled {
            return;
        }
        self.fields.push_str(tag);
        self.fields.push('=');
        self.fields.push_str(&value.to_string());
        self.fields.push(' ');
    }
}
