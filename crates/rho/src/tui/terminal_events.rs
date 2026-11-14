use std::io;

use crossterm::event::{Event, EventStream};
use futures_util::StreamExt;

#[cfg(windows)]
#[path = "windows_input.rs"]
mod windows;
#[cfg(windows)]
pub(super) use windows::VtInput;

#[cfg(any(windows, test))]
#[path = "windows_input_parser.rs"]
mod parser;
#[cfg(all(test, not(windows)))]
#[path = "windows_input_records.rs"]
mod windows_records;

pub(super) struct TerminalEvents {
    #[cfg(not(windows))]
    stream: EventStream,
    #[cfg(windows)]
    stream: Option<WindowsStream>,
}

#[cfg(windows)]
enum WindowsStream {
    Vt(windows::Input),
    Legacy(EventStream),
}

impl TerminalEvents {
    pub(super) fn new() -> Self {
        Self {
            #[cfg(not(windows))]
            stream: EventStream::new(),
            #[cfg(windows)]
            stream: None,
        }
    }

    /// Events decoded but not yet consumed by the event loop. Used by
    /// `RHO_PERF` diagnostics to spot input backlogs.
    pub(super) fn queued(&self) -> usize {
        #[cfg(not(windows))]
        {
            0
        }
        #[cfg(windows)]
        {
            match self.stream.as_ref() {
                Some(WindowsStream::Vt(input)) => input.queued(),
                _ => 0,
            }
        }
    }

    /// An already-decoded event without waiting, if one is buffered.
    ///
    /// Lets the event loop drain bursts (mouse tracking, paste) in one pass
    /// instead of paying a full poll/draw cycle per record.
    pub(super) fn try_next(&mut self) -> Option<io::Result<Event>> {
        use futures_util::FutureExt;
        #[cfg(not(windows))]
        {
            self.stream.next().now_or_never().flatten()
        }
        #[cfg(windows)]
        {
            match self.stream.as_mut() {
                Some(WindowsStream::Vt(input)) => input.try_next(),
                Some(WindowsStream::Legacy(stream)) => stream.next().now_or_never().flatten(),
                None => None,
            }
        }
    }

    pub(super) async fn next(&mut self) -> io::Result<Event> {
        #[cfg(not(windows))]
        {
            event_result(self.stream.next().await)
        }
        #[cfg(windows)]
        {
            // Keyboard modes may be acquired after this object is constructed.
            // Do not start either console reader until the first request.
            if self.stream.is_none() {
                self.stream = Some(match windows::Input::start()? {
                    Some(input) => WindowsStream::Vt(input),
                    None => WindowsStream::Legacy(EventStream::new()),
                });
            }
            match self.stream.as_mut().expect("initialized above") {
                WindowsStream::Vt(input) => input.next().await,
                WindowsStream::Legacy(stream) => event_result(stream.next().await),
            }
        }
    }
}

fn event_result(event: Option<io::Result<Event>>) -> io::Result<Event> {
    event.unwrap_or_else(|| Err(stream_ended_error()))
}

pub(super) fn stream_ended_error() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "terminal event stream ended")
}
