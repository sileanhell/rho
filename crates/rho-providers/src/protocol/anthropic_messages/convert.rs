use crate::{
    model::handoff::prepare_assistant,
    provider_backend::{
        ContentBlock, Message, ModelError, ModelResponse, ModelUsage, ToolCall, ToolSpec,
    },
};

use super::types::{
    AnthropicContentBlock, AnthropicImageSource, AnthropicMessage, AnthropicResponse,
    AnthropicRole, AnthropicTool, AnthropicUsage,
};

fn resolved_cache_read_tokens(usage: &AnthropicUsage) -> Option<u64> {
    usage.cache_read_input_tokens
}

fn resolved_cache_write_tokens(usage: &AnthropicUsage) -> Option<u64> {
    usage.cache_creation_input_tokens.or_else(|| {
        let cache = usage.cache_creation.as_ref()?;
        match (
            cache.ephemeral_1h_input_tokens,
            cache.ephemeral_5m_input_tokens,
        ) {
            (None, None) => None,
            (one_hour, five_minutes) => Some(
                one_hour
                    .unwrap_or_default()
                    .saturating_add(five_minutes.unwrap_or_default()),
            ),
        }
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderContextReplay {
    Enabled,
    Disabled,
}

pub(crate) fn split_system_and_messages(
    messages: &[Message],
    target: &crate::model::ModelIdentity,
    provider_context_replay: ProviderContextReplay,
) -> Result<(Option<String>, Vec<AnthropicMessage>), ModelError> {
    let messages = crate::protocol::late_tool_results::normalize_late_tool_results(messages);
    let mut system = Vec::new();
    let mut converted = Vec::new();
    for message in messages.iter().map(std::borrow::Cow::as_ref) {
        match message {
            Message::System(content) => system.push(content.as_str()),
            Message::User(blocks) => push_message(
                &mut converted,
                AnthropicRole::User,
                blocks.iter().map(user_block).collect(),
            ),
            Message::Assistant(blocks) => push_message(
                &mut converted,
                AnthropicRole::Assistant,
                blocks.iter().map(assistant_block).collect(),
            ),
            Message::EnrichedAssistant(message) => {
                let mut message = (**message).clone();
                if provider_context_replay == ProviderContextReplay::Disabled {
                    message.retain_portable_context();
                }
                let prepared = prepare_assistant(message, target);
                let mut content = prepared
                    .content
                    .iter()
                    .map(assistant_block)
                    .collect::<Vec<_>>();
                for block in prepared.replay_context {
                    if provider_context_replay == ProviderContextReplay::Disabled
                        || block.kind != "anthropic_content_block"
                    {
                        continue;
                    }
                    if let Ok(block_data) = serde_json::from_value(block.data) {
                        content.insert(
                            block.position.unwrap_or(content.len()).min(content.len()),
                            block_data,
                        );
                    }
                }
                push_message(&mut converted, AnthropicRole::Assistant, content);
            }
            Message::AbortedAssistant(message) => {
                let content = message
                    .content
                    .iter()
                    .map(|block| match block {
                        ContentBlock::ToolCall(call) => ContentBlock::Text(render_tool_call(call)),
                        other => other.clone(),
                    })
                    .collect::<Vec<_>>();
                let mut enriched = crate::model::AssistantMessage {
                    content,
                    provenance: message.provenance.clone(),
                    reasoning_summary: message.reasoning_summary.clone(),
                    provider_context: message.provider_context.clone(),
                };
                enriched
                    .content
                    .push(ContentBlock::Text("[Operation aborted]".into()));
                if provider_context_replay == ProviderContextReplay::Disabled {
                    enriched.retain_portable_context();
                }
                let prepared = prepare_assistant(enriched, target);
                let mut content = prepared
                    .content
                    .iter()
                    .map(assistant_block)
                    .collect::<Vec<_>>();
                for block in prepared.replay_context {
                    if provider_context_replay == ProviderContextReplay::Enabled
                        && block.kind == "anthropic_content_block"
                    {
                        if let Ok(block_data) = serde_json::from_value(block.data) {
                            content.insert(
                                block.position.unwrap_or(content.len()).min(content.len()),
                                block_data,
                            );
                        }
                    }
                }
                push_message(&mut converted, AnthropicRole::Assistant, content);
            }
            Message::ToolResult(result) => push_message(
                &mut converted,
                AnthropicRole::User,
                vec![AnthropicContentBlock::ToolResult {
                    tool_use_id: result.id.clone(),
                    content: result.content.clone(),
                    is_error: !result.ok,
                    cache_control: None,
                }],
            ),
        }
    }
    // Anthropic 400s on an assistant message whose final block is thinking or
    // redacted_thinking, and on empty message content. A replayed thinking
    // block can land last (interleaved-thinking positions) or be the whole
    // message; stub text keeps the request valid (Hermes #13010).
    for message in &mut converted {
        if message.role != AnthropicRole::Assistant {
            continue;
        }
        let stub = if message.content.is_empty() {
            Some("(empty)")
        } else if matches!(
            message.content.last(),
            Some(
                AnthropicContentBlock::Thinking { .. }
                    | AnthropicContentBlock::RedactedThinking { .. }
            )
        ) {
            Some("(continued)")
        } else {
            None
        };
        if let Some(text) = stub {
            message.content.push(AnthropicContentBlock::Text {
                text: text.into(),
                cache_control: None,
            });
        }
    }
    let system = (!system.is_empty()).then(|| system.join("\n\n"));
    Ok((system, converted))
}

fn push_message(
    messages: &mut Vec<AnthropicMessage>,
    role: AnthropicRole,
    mut content: Vec<AnthropicContentBlock>,
) {
    if let Some(previous) = messages.last_mut().filter(|message| message.role == role) {
        previous.content.append(&mut content);
    } else {
        messages.push(AnthropicMessage::new(role, content));
    }
}

fn user_block(block: &ContentBlock) -> AnthropicContentBlock {
    match block {
        ContentBlock::Text(text) => AnthropicContentBlock::Text {
            text: text.clone(),
            cache_control: None,
        },
        ContentBlock::Image(image) => AnthropicContentBlock::Image {
            source: AnthropicImageSource {
                kind: "base64".into(),
                media_type: image.mime_type.clone(),
                data: image.data.clone(),
            },
        },
        ContentBlock::ToolCall(call) => AnthropicContentBlock::Text {
            text: render_tool_call(call),
            cache_control: None,
        },
    }
}

fn assistant_block(block: &ContentBlock) -> AnthropicContentBlock {
    match block {
        ContentBlock::Text(text) => AnthropicContentBlock::Text {
            text: text.clone(),
            cache_control: None,
        },
        ContentBlock::Image(image) => AnthropicContentBlock::Text {
            text: format!("[image: {}]", image.mime_type),
            cache_control: None,
        },
        ContentBlock::ToolCall(call) => AnthropicContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.arguments.clone(),
        },
    }
}

fn render_tool_call(call: &ToolCall) -> String {
    let arguments = serde_json::to_string_pretty(&call.arguments)
        .unwrap_or_else(|_| call.arguments.to_string());
    format!("Tool call: {}\n{}", call.name, arguments)
}

pub(crate) fn to_anthropic_tool(tool: ToolSpec) -> AnthropicTool {
    AnthropicTool {
        name: tool.name,
        description: tool.description,
        input_schema: sanitize_anthropic_input_schema(tool.input_schema),
        cache_control: None,
    }
}

/// Anthropic custom tools require `input_schema.type` and reject top-level
/// `oneOf` / `anyOf` / `allOf`. Strip those keywords and force a root object
/// type; tool specs own portable parameter advertising (no composition fold).
fn sanitize_anthropic_input_schema(mut input_schema: serde_json::Value) -> serde_json::Value {
    let Some(schema) = input_schema.as_object_mut() else {
        return serde_json::json!({
            "type": "object",
            "properties": {},
        });
    };

    schema.remove("oneOf");
    schema.remove("anyOf");
    schema.remove("allOf");
    schema.insert("type".into(), serde_json::json!("object"));
    input_schema
}

pub(crate) fn convert_anthropic_response(
    response: AnthropicResponse,
) -> Result<ModelResponse, ModelError> {
    let _usage = response.usage.map(usage_to_model_usage);
    convert_content_blocks(response.content, response.stop_reason.as_deref())
}

pub(crate) fn convert_content_blocks(
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<&str>,
) -> Result<ModelResponse, ModelError> {
    let mut blocks = Vec::new();
    for block in content {
        match block {
            AnthropicContentBlock::Text { text, .. } if !text.is_empty() => {
                blocks.push(ContentBlock::Text(text));
            }
            AnthropicContentBlock::Text { .. } => {}
            AnthropicContentBlock::Thinking { .. }
            | AnthropicContentBlock::RedactedThinking { .. } => {}
            AnthropicContentBlock::Image { .. } => {}
            AnthropicContentBlock::ToolUse { id, name, input } => {
                blocks.push(ContentBlock::ToolCall(ToolCall {
                    id,
                    name,
                    arguments: input,
                }));
            }
            AnthropicContentBlock::ToolResult { .. } => {
                return Err(ModelError::InvalidResponse(
                    "assistant response contained tool_result block".into(),
                ));
            }
        }
    }
    if !blocks.is_empty() {
        return Ok(ModelResponse::Assistant(blocks));
    }
    match stop_reason {
        // Empty content on a normal finish is a legitimate "nothing more to
        // add" turn: it follows an assistant message that already delivered
        // text alongside a trivial tool call, or a thinking-only completion
        // (Hermes #14158). Rejecting it burns the retry budget on an
        // identical request that deterministically returns the same shape,
        // so accept it with the placeholder text Hermes uses for
        // reasoning-only completions. Early stops (tool_use, max_tokens,
        // refusal, pause_turn) stay retryable-invalid below.
        Some("end_turn" | "stop_sequence") => {
            Ok(ModelResponse::Assistant(vec![ContentBlock::Text(
                "(empty)".into(),
            )]))
        }
        Some(reason) => Err(ModelError::empty_assistant_because(format!(
            "assistant message had no content or tool calls (stop_reason: {reason})"
        ))),
        None => Err(ModelError::empty_assistant()),
    }
}

pub(crate) fn usage_to_model_usage(usage: AnthropicUsage) -> ModelUsage {
    let cache_read_tokens = resolved_cache_read_tokens(&usage);
    let cache_write_tokens = resolved_cache_write_tokens(&usage);
    let total_tokens = usage
        .input_tokens
        .unwrap_or_default()
        .saturating_add(cache_read_tokens.unwrap_or_default())
        .saturating_add(cache_write_tokens.unwrap_or_default())
        .saturating_add(usage.output_tokens.unwrap_or_default());
    let has_total = usage.input_tokens.is_some()
        || cache_read_tokens.is_some()
        || cache_write_tokens.is_some()
        || usage.output_tokens.is_some();
    ModelUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_tokens,
        cache_write_tokens,
        total_tokens: has_total.then_some(total_tokens),
        context_window: None,
        cost_usd_micros: None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::model::{AssistantMessage, ModelIdentity, ProviderContextBlock};

    fn target() -> ModelIdentity {
        ModelIdentity::new("anthropic", "anthropic-messages", "claude-test")
    }

    fn thinking_context(position: usize) -> ProviderContextBlock {
        ProviderContextBlock {
            identity: target(),
            kind: "anthropic_content_block".into(),
            position: Some(position),
            data: serde_json::to_value(AnthropicContentBlock::Thinking {
                thinking: "draft".into(),
                signature: "sig".into(),
            })
            .unwrap(),
        }
    }

    #[test]
    fn response_with_only_thinking_and_end_turn_is_accepted() {
        let response = AnthropicResponse {
            content: vec![AnthropicContentBlock::Thinking {
                thinking: "private".into(),
                signature: "sig".into(),
            }],
            usage: None,
            stop_reason: Some("end_turn".into()),
        };

        let ModelResponse::Assistant(blocks) = convert_anthropic_response(response).unwrap();
        assert!(matches!(blocks.as_slice(), [ContentBlock::Text(text)] if !text.is_empty()));
    }

    #[test]
    fn empty_response_with_an_early_stop_stays_retryable() {
        let response = AnthropicResponse {
            content: Vec::new(),
            usage: None,
            stop_reason: Some("max_tokens".into()),
        };

        let error = convert_anthropic_response(response).unwrap_err();
        match error {
            ModelError::RetryableInvalidResponse { message, .. } => {
                assert!(message.contains("max_tokens"), "{message}");
            }
            other => panic!("expected retryable invalid response, got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_ending_in_thinking_gets_stub_text() {
        // Interleaved thinking can sit after a tool call in the stored
        // positions; replayed verbatim it leaves a trailing thinking block,
        // which Anthropic rejects (Hermes #13010).
        let messages = vec![Message::assistant(AssistantMessage {
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "toolu_1".into(),
                name: "bash".into(),
                arguments: json!({}),
            })],
            provider_context: vec![thinking_context(1)],
            ..AssistantMessage::default()
        })];

        let (_, converted) =
            split_system_and_messages(&messages, &target(), ProviderContextReplay::Enabled)
                .unwrap();

        let message = converted
            .iter()
            .find(|message| message.role == AnthropicRole::Assistant)
            .expect("assistant message was converted");
        assert!(matches!(
            message.content.as_slice(),
            [
                AnthropicContentBlock::ToolUse { .. },
                AnthropicContentBlock::Thinking { .. },
                AnthropicContentBlock::Text { ref text, .. }
            ] if !text.is_empty()
        ));
    }

    #[test]
    fn empty_assistant_message_gets_stub_text() {
        let messages = vec![Message::assistant(AssistantMessage::default())];

        let (_, converted) =
            split_system_and_messages(&messages, &target(), ProviderContextReplay::Enabled)
                .unwrap();

        let [message] = converted.as_slice() else {
            panic!("expected one converted message");
        };
        assert!(matches!(
            message.content.as_slice(),
            [AnthropicContentBlock::Text { ref text, .. }] if !text.is_empty()
        ));
    }
}
