# Parsing responses

Turn what a chat model generated back into a structured message: reasoning, content and tool calls.

## Overview

A chat template turns messages into a prompt. A response template does the reverse: it
describes where a model writes its reasoning, its answer and its tool calls, so the generated
text can become a message again, ready to append to the conversation.

Models that ship one carry it as `response_template` in `tokenizer_config.json`, Gemma 4 among
them. This library reads the same format with the same rules as `transformers`'
`parse_response`, and is checked against it event by event.

```swift
let prompt = try tokenizer.renderChatTemplate(messages: messages, addGenerationPrompt: true, tools: tools)
let output = generate(prompt)

let message = try tokenizer.parseResponse(output, prefix: prompt, tools: tools)
// ["role": "assistant", "thinking": "…", "tool_calls": [["type": "function", "function": […]]]]
```

The `prefix` is the prompt the model continued. It is required because templates often open
the assistant message themselves, with a `<think>` for instance, and the parser has to see
that to know the model is reasoning. Pass `""` when the output holds the whole message.

Passing the `tools` you offered types the tool-call arguments from each tool's JSON Schema:
`"3"` becomes `3` for an integer parameter and stays `"3"` for a string one.

### While the model is still writing

``ResponseParser`` takes the text piece by piece and reports which region each piece belongs to,
so a reasoning trace can be shown as it streams and the answer kept apart:

```swift
var parser = try tokenizer.makeResponseParser(prefix: prompt, tools: tools)
for event in parser.initialEvents { render(event) }
for try await piece in generation {
    for event in try parser.feed(piece) { render(event) }
}
let (message, events) = try parser.finalize()
```

Text that might be the start of a delimiter is held back until the next piece decides, so a
`</think>` never reaches the screen half written. A piece costs well under a microsecond.

### Feeding the message back

The parsed message is a ``ResponseObject`` whose keys keep the order the model wrote them in.
``ResponseObject/message`` gives it the shape chat templates accept, and tool-call arguments
render in that same order:

```swift
messages.append(message.message)
```

### Templates without a tokenizer

A ``ResponseTemplate`` can also be built from JSON, for models whose files do not carry one
yet. Compile it once and share it: it is `Sendable` and holds its regular expressions compiled.

```swift
let template = try ResponseTemplate(jsonString: json)
let message = try ResponseParser.parse(output, template: template, prefix: prompt)
```
