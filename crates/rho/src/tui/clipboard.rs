use std::{io, path::PathBuf};

use rho_providers::model::{image_summary, ImageContent};

use crate::clipboard::{
    paste_text_as_file_path, path_has_supported_image_magic, read_clipboard_image, read_image_file,
};
pub(super) use crate::clipboard::{CopyOutcome, SystemClipboard};

use super::{
    feed_image::{DecodedFeedImage, FeedImage},
    media_attach::{MediaAttachOutcome, MediaAttachTask},
    App, ChatMedia, ChatTextDocument, ComposerMode, MediaAttachId, PendingAttachmentSource,
};

enum PastedImageOutcome {
    NotImage,
    Image(ImageContent),
    Failed { kind: &'static str, message: String },
}

/// Reads and writes the user's clipboard synchronously.
///
/// `copy` must preserve the supplied text and report whether the destination confirmed the write.
/// `paste` returns the current clipboard text; an empty clipboard is `Ok("")`. Errors mean that no
/// available backend accepted the request.
pub(super) trait Clipboard {
    fn copy(&mut self, text: &str) -> io::Result<CopyOutcome>;
    fn paste(&mut self) -> io::Result<String>;
}

impl Clipboard for SystemClipboard {
    fn copy(&mut self, text: &str) -> io::Result<CopyOutcome> {
        self.copy_text(text)
    }

    fn paste(&mut self) -> io::Result<String> {
        self.paste_text()
    }
}

/// Whether the host terminal acts on right-click itself even while Rho has
/// mouse reporting on.
///
/// Any terminal that names itself in `TERM_PROGRAM` (VS Code and its forks,
/// iTerm2, WezTerm, mintty, tmux, ...) is trusted to own right-click; VS Code,
/// for one, pastes per `terminal.integrated.rightClickBehavior` and still
/// forwards the click, so a Rho paste would insert the clipboard twice. Plain
/// console hosts such as conhost (`cmd.exe`) and Windows Terminal do not set
/// it, so Rho keeps its own right-click paste there.
pub(super) fn host_handles_right_click(get_var: impl Fn(&str) -> Option<String>) -> bool {
    get_var("TERM_PROGRAM").is_some_and(|value| !value.trim().is_empty())
}

impl App {
    /// Right-click paste, unless the host terminal already pastes on right-click.
    pub(super) fn paste_clipboard_on_right_click(&mut self) {
        if !self.host_handles_right_click {
            self.paste_clipboard_text();
        }
    }

    /// Pastes host clipboard text through the same composer path as bracketed paste.
    pub(super) fn paste_clipboard_text(&mut self) {
        match self.clipboard.paste() {
            Ok(text) if text.is_empty() => {}
            Ok(text) => self.apply_external_paste(&text),
            Err(err) => {
                self.notify_status(format!("could not paste clipboard: {err}"));
            }
        }
    }

    pub(super) fn paste_clipboard_image(&mut self) {
        if self.is_ui_busy() {
            self.notify_status("image paste is unavailable while a model turn is running");
            return;
        }
        if !matches!(self.input_ui.composer(), ComposerMode::Input) {
            self.notify_status("image paste is only available in the message box");
            return;
        }
        match read_clipboard_image() {
            Ok(image) => self.attach_ready_image(image),
            Err(err) => {
                self.notify_status(format!("image paste failed: {err}"));
            }
        }
    }

    /// Starts background extraction when the paste is a path to a regular file.
    pub(super) fn start_pasted_media_path(&mut self, text: &str) -> bool {
        if self.is_ui_busy() || !matches!(self.input_ui.composer(), ComposerMode::Input) {
            return false;
        }
        let Some(path) = paste_text_as_file_path(text, &self.info.runtime.cwd) else {
            return false;
        };
        let original_text = text.to_owned();
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let id = MediaAttachId::new();
        let task = tokio::spawn(classify_pasted_path(path, original_text));
        self.media_attach_tasks.push(MediaAttachTask { id, task });
        self.input_ui
            .push_pending_attachment(id, PendingAttachmentSource::File, name.clone());
        self.notify_status(format!("extracting {name}"));
        true
    }

    fn attach_ready_image(&mut self, image: ImageContent) {
        let summary = image_summary(&image);
        let id = MediaAttachId::new();
        self.input_ui
            .push_pending_attachment(id, PendingAttachmentSource::Image, summary.clone());
        let wants_preview = self.image_picker.is_some();
        let task = tokio::spawn(async move {
            let decoded_preview = if wants_preview {
                FeedImage::decode_base64_preview_async(image.data.clone()).await
            } else {
                None
            };
            MediaAttachOutcome::ready_image(image, decoded_preview)
        });
        self.media_attach_tasks.push(MediaAttachTask { id, task });
        self.notify_status(format!("decoding image ({summary})"));
    }

    pub(super) fn finish_pending_image(
        &mut self,
        id: MediaAttachId,
        image: ImageContent,
        decoded_preview: Option<DecodedFeedImage>,
    ) {
        let summary = image_summary(&image);
        let preview = decoded_preview.and_then(|decoded| {
            self.image_picker
                .as_ref()
                .map(|picker| decoded.to_feed_image(picker))
        });
        if let Some(index) =
            self.input_ui
                .replace_pending_attachment(id, ChatMedia::Image(image), preview)
        {
            self.notify_status(format!("attached image {} ({summary})", index + 1));
        }
    }
}

async fn classify_pasted_path(path: PathBuf, original_text: String) -> MediaAttachOutcome {
    let path = match tokio::task::spawn_blocking(move || {
        path.canonicalize().ok().filter(|path| path.is_file())
    })
    .await
    {
        Ok(Some(path)) => path,
        Ok(None) | Err(_) => return MediaAttachOutcome::Unsupported { original_text },
    };
    let image_path = path.clone();
    let image_outcome =
        tokio::task::spawn_blocking(move || classify_pasted_image(image_path)).await;
    match image_outcome {
        Ok(PastedImageOutcome::Image(image)) => {
            let decoded_preview = FeedImage::decode_base64_preview_async(image.data.clone()).await;
            MediaAttachOutcome::ready_image(image, decoded_preview)
        }
        Ok(PastedImageOutcome::Failed { kind, message }) => {
            MediaAttachOutcome::Failed { kind, message }
        }
        Ok(PastedImageOutcome::NotImage) => {
            match rho_tools::document::extract_document_from_path_async(path).await {
                Ok(document) => MediaAttachOutcome::ready(ChatMedia::TextDocument(
                    ChatTextDocument::from(document),
                )),
                Err(rho_tools::document::DocumentExtractionError::UnsupportedFormat { .. }) => {
                    MediaAttachOutcome::Unsupported { original_text }
                }
                Err(error) => MediaAttachOutcome::Failed {
                    kind: "document paste",
                    message: error.to_string(),
                },
            }
        }
        Err(error) => MediaAttachOutcome::Failed {
            kind: "file paste",
            message: error.to_string(),
        },
    }
}

fn classify_pasted_image(path: PathBuf) -> PastedImageOutcome {
    match path_has_supported_image_magic(&path) {
        Ok(true) => match read_image_file(&path) {
            Ok(image) => PastedImageOutcome::Image(image),
            Err(error) => PastedImageOutcome::Failed {
                kind: "image paste",
                message: error.to_string(),
            },
        },
        Ok(false) => PastedImageOutcome::NotImage,
        Err(error) => PastedImageOutcome::Failed {
            kind: "file paste",
            message: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::host_handles_right_click;

    fn detect(vars: &[(&str, &str)]) -> bool {
        host_handles_right_click(|key| {
            vars.iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| (*value).to_owned())
        })
    }

    #[test]
    fn any_named_terminal_owns_right_click() {
        assert!(detect(&[("TERM_PROGRAM", "vscode")]));
        assert!(detect(&[("TERM_PROGRAM", "WezTerm")]));
        assert!(detect(&[("TERM_PROGRAM", "mintty")]));
    }

    #[test]
    fn console_hosts_keep_rho_right_click_paste() {
        assert!(!detect(&[]));
        assert!(!detect(&[("TERM_PROGRAM", "  ")]));
        assert!(!detect(&[("WT_SESSION", "abc")]));
    }
}
