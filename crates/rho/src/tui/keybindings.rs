use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::Backend, Terminal};

use crate::tui::DefaultTerminal;

use super::{App, InteractiveRuntime};

pub(super) enum UndoDirection {
    Undo,
    Redo,
}

/// Use only as the final configurable-chord fallback, so new undo defaults
/// never shadow an existing binding. Undo wins when both chords match.
pub(super) fn undo_direction(
    keybindings: &crate::keybindings::Keybindings,
    key: KeyEvent,
) -> Option<UndoDirection> {
    if keybindings.undo.matches(key) {
        Some(UndoDirection::Undo)
    } else if keybindings.redo.matches(key) {
        Some(UndoDirection::Redo)
    } else {
        None
    }
}

impl App {
    pub(super) fn handle_configurable_running_key<B: Backend>(
        &mut self,
        key: KeyEvent,
        terminal: &mut Terminal<B>,
    ) -> anyhow::Result<bool> {
        if self.info.runtime.keybindings.queue_prompt_matches(key) {
            self.queue_prompt_after_turn()?;
        } else if self
            .info
            .runtime
            .keybindings
            .cycle_permission_mode_matches(key)
        {
            self.queue_permission_mode_cycle();
        } else if self.info.runtime.keybindings.paste_image.matches(key)
            || matches!(
                (key.modifiers, key.code),
                (KeyModifiers::ALT, KeyCode::Char('v'))
            )
        {
            self.paste_clipboard_image();
        } else if self
            .info
            .runtime
            .keybindings
            .toggle_tool_output
            .matches(key)
        {
            self.toggle_latest_tool_output(terminal)?;
        } else if self
            .info
            .runtime
            .keybindings
            .cycle_streaming_mode
            .matches(key)
        {
            self.cycle_streaming_mode();
        } else if self
            .info
            .runtime
            .keybindings
            .reset_conversation_matches(key)
        {
            self.notify_status("/new is unavailable while a model turn is running");
        } else if self
            .info
            .runtime
            .keybindings
            .search_prompt_history
            .matches(key)
        {
            self.open_prompt_history_search();
        } else if self.info.runtime.keybindings.insert_newline.matches(key) {
            // Before search_transcript: configs that already bound the new
            // default chord to insert_newline keep inserting newlines.
            self.insert_input_char('\n');
        } else if self.info.runtime.keybindings.search_transcript.matches(key) {
            self.open_transcript_search(terminal)
                .map_err(|error| anyhow::anyhow!("could not read terminal size: {error}"))?;
        } else {
            match undo_direction(&self.info.runtime.keybindings, key) {
                Some(UndoDirection::Undo) => self.undo_input(),
                Some(UndoDirection::Redo) => self.redo_input(),
                None => return Ok(false),
            }
        }
        self.input_ui.clear_paste_burst();
        self.ctrl_c_streak = 0;
        Ok(true)
    }

    pub(super) async fn handle_configurable_composer_key(
        &mut self,
        key: KeyEvent,
        terminal: &mut DefaultTerminal,
        agent: &mut InteractiveRuntime,
    ) -> anyhow::Result<bool> {
        if self.info.runtime.keybindings.queue_prompt_matches(key) {
            if agent.is_compacting() {
                self.queue_prompt_after_turn()?;
            } else {
                self.insert_input_char('\n');
            }
        } else if self
            .info
            .runtime
            .keybindings
            .cycle_permission_mode_matches(key)
        {
            self.cycle_permission_mode(agent).await?;
        } else if self.info.runtime.keybindings.paste_image.matches(key)
            || matches!(
                (key.modifiers, key.code),
                (KeyModifiers::ALT, KeyCode::Char('v'))
            )
        {
            self.paste_clipboard_image();
        } else if self
            .info
            .runtime
            .keybindings
            .toggle_tool_output
            .matches(key)
        {
            self.toggle_latest_tool_output(terminal)?;
        } else if self
            .info
            .runtime
            .keybindings
            .cycle_streaming_mode
            .matches(key)
        {
            self.cycle_streaming_mode();
        } else if self
            .info
            .runtime
            .keybindings
            .reset_conversation_matches(key)
        {
            self.execute_new_command(terminal, agent).await?;
        } else if self
            .info
            .runtime
            .keybindings
            .search_prompt_history
            .matches(key)
        {
            self.open_prompt_history_search();
        } else if self.info.runtime.keybindings.insert_newline.matches(key) {
            // Before search_transcript; see handle_configurable_running_key.
            self.insert_input_char('\n');
        } else if self.info.runtime.keybindings.search_transcript.matches(key) {
            self.open_transcript_search(terminal)?;
        } else {
            match undo_direction(&self.info.runtime.keybindings, key) {
                Some(UndoDirection::Undo) => self.undo_input(),
                Some(UndoDirection::Redo) => self.redo_input(),
                None => return Ok(false),
            }
        }
        self.input_ui.clear_paste_burst();
        self.ctrl_c_streak = 0;
        Ok(true)
    }
}
