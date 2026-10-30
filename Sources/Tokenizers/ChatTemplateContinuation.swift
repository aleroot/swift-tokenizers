import Foundation

/// Continuation of the final message of a conversation, as Hugging Face `continue_final_message`
/// does: the rendered prompt ends inside the last message, so the model writes the rest of it
/// instead of opening a new turn.
///
/// The final message is marked before the template renders it, and the rendered text is cut at the
/// mark. The marker is the one `transformers` uses, so a template that transforms the message
/// (trims it, for instance) ends the prompt in the same place in both.
public enum ChatTemplateContinuation {
    /// The context key that asks a chat template to continue its final message. `transformers`
    /// names it the same way.
    ///
    /// Only the Boolean `true` asks for a continuation, and a request never reaches the template.
    /// Any other value is the caller's own context and reaches the template as it did before the
    /// key was reserved.
    public static let key = "continue_final_message"

    /// Whether the context asks for a continuation.
    static func isRequested(_ additionalContext: [String: any Sendable]?) -> Bool {
        additionalContext?[key] as? Bool == true
    }

    /// The context without the request, so a template never reads it.
    static func withoutRequest(_ additionalContext: [String: any Sendable]?) -> [String: any Sendable]? {
        guard isRequested(additionalContext) else { return additionalContext }
        var context = additionalContext
        context?[key] = nil
        return context
    }

    /// The marker that finds the end of the final message inside a rendered template.
    private static let marker = "CONTINUE_FINAL_MESSAGE_TAG"

    /// The marker as it is written into the message. The space keeps it apart from the text.
    private static let markedSuffix = marker + " "

    /// The conversation with the marker appended to its final message, and the text of that
    /// message.
    static func marking(_ messages: [Message]) throws -> (messages: [Message], text: String) {
        guard var final = messages.last else {
            throw TokenizerError.chatTemplate("There is no message to continue.")
        }
        let text: String
        if let content = final["content"] as? String {
            text = content
            final["content"] = content + markedSuffix
        } else if var parts = final["content"] as? [[String: any Sendable]] {
            // A vision model writes its final message as parts. The last text part continues.
            guard let index = parts.lastIndex(where: { $0["text"] is String }),
                  let content = parts[index]["text"] as? String
            else {
                throw TokenizerError.chatTemplate("The final message has no text to continue.")
            }
            text = content
            parts[index]["text"] = content + markedSuffix
            final["content"] = parts
        } else {
            throw TokenizerError.chatTemplate("The final message has no text to continue.")
        }
        var marked = messages
        marked[marked.count - 1] = final
        return (marked, text)
    }

    /// The rendered prompt up to the end of the final message.
    static func prompt(_ rendered: String, finalText: String) throws -> String {
        guard rendered.contains(finalText.trimmingCharacters(in: .whitespacesAndNewlines)),
              let mark = rendered.range(of: marker, options: .backwards)
        else {
            throw TokenizerError.chatTemplate(
                "The chat template does not show the final message, so it cannot be continued.")
        }
        // The template either keeps the space that follows the marker or trims it.
        if rendered[mark.lowerBound...].hasPrefix(markedSuffix) {
            return String(rendered[..<mark.lowerBound])
        }
        return Self.droppingTrailingWhitespace(String(rendered[..<mark.lowerBound]))
    }

    /// The text without the whitespace at its end.
    private static func droppingTrailingWhitespace(_ text: String) -> String {
        var result = text
        while let last = result.last, last.isWhitespace { result.removeLast() }
        return result
    }
}
