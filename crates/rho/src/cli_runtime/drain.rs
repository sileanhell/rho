//! Stream-json line decode error formatting for NDJSON transports.

use super::line_decoder::LineDecodeError;

pub(crate) fn format_line_error(error: &LineDecodeError, program_label: &str) -> String {
    match error {
        LineDecodeError::InvalidUtf8(_) => {
            format!("{program_label}: malformed UTF-8 on stream-json stdout: {error}")
        }
        LineDecodeError::LineTooLong { .. } => {
            format!("{program_label}: oversize stream-json line: {error}")
        }
    }
}
