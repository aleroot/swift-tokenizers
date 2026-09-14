# Chat templates

Turn a conversation into the exact prompt a chat model was trained on.

## Overview

Instruction tuned models expect their own markup around each turn. That markup is shipped
with the model as a Jinja template, and this library renders it with
[swift-jinja](https://github.com/huggingface/swift-jinja), the same way `transformers` does.

```swift
let messages: [Message] = [
    ["role": "system", "content": "You are terse."],
    ["role": "user", "content": "Why is the sky blue?"],
]

let ids = try tokenizer.applyChatTemplate(messages: messages)
```

A ``Message`` is a plain `[String: any Sendable]` dictionary. Use the keys the model's template
expects, normally `role` and `content`.

To see the prompt instead of the ids, render it:

```swift
let prompt = try tokenizer.renderChatTemplate(messages: messages, addGenerationPrompt: true)
```

That is the first thing to try when a model answers strangely. Nine times out of ten the
prompt is not what you assumed.

### The generation prompt

`addGenerationPrompt` appends the opening of the assistant's turn, for example
`<|im_start|>assistant\n`. Without it the model has to guess that it is now its turn to speak.

Watch the defaults, they follow swift-transformers:

```swift
// generation prompt added
try tokenizer.applyChatTemplate(messages: messages)
try tokenizer.applyChatTemplate(messages: messages, tools: tools)

// generation prompt NOT added unless you ask
try tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: true, tools: tools)
```

The short overloads add it because that is what generation wants. The long overload takes the
flag explicitly and defaults it to `false`.

### Tools

Pass tool definitions in the OpenAI function calling shape and the template decides how to
present them:

```swift
let tools: [ToolSpec] = [[
    "type": "function",
    "function": [
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": [
            "type": "object",
            "properties": ["city": ["type": "string", "description": "City name"]],
            "required": ["city"],
        ],
    ],
]]

let ids = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
```

Keys render in the canonical order `transformers.utils.get_json_schema` produces, so the
prompt is byte identical to Python for standard schemas. Containers you name yourself, such
as `properties` or `arguments`, are Swift dictionaries and therefore unordered. When a
specific order matters, pass `KeyValuePairs` or a `Jinja.Value` and it is preserved.

If the model ships several templates, one named `tool_use` is selected automatically whenever
you pass a non-empty `tools` array.

### Extra template variables

Some families read variables of their own, for example Llama 3.1's `tools_in_user_message`:

```swift
try tokenizer.applyChatTemplate(
    messages: messages,
    tools: tools,
    additionalContext: ["tools_in_user_message": false])
```

### Choosing a template

```swift
try tokenizer.applyChatTemplate(messages: messages, chatTemplate: .name("rag"))
try tokenizer.applyChatTemplate(messages: messages, chatTemplate: .literal(myTemplate))
```

``ChatTemplateArgument/name(_:)`` picks one of the named templates in the configuration.
``ChatTemplateArgument/literal(_:)`` supplies your own source and ignores whatever the model
shipped. The literal form is the escape hatch when a vision model's bundled template lacks a
branch you need.

Check first, rather than catching:

```swift
guard tokenizer.hasChatTemplate else { return }
```

Base models have no template, and asking for one throws
``TokenizerError/missingChatTemplate``.

### Training: assistant loss masks

For fine tuning you need to know which tokens the model should be scored on. Templates that
mark assistant content with `{% generation %}` can report that alignment:

```swift
let result = try tokenizer.applyChatTemplateWithAssistantMask(messages: conversation)

for (id, isAssistant) in zip(result.ids, result.assistantMask) where isAssistant {
    // train on this token
}
```

``ChatTemplateEncoding/assistantMask`` has one flag per token. A token that straddles the
boundary between prompt and completion counts as assistant content; special tokens inserted
afterwards by the post-processor do not.

A template without `{% generation %}` annotations throws instead of returning an all-false
mask, because a silently empty mask trains on nothing and looks like a bad learning rate.
