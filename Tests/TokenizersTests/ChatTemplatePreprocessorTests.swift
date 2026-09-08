// The preprocessor fills the jinja2 whitespace-control gaps in swift-jinja:
// comment strip markers (`{#-` / `-#}`) and a literal `{` before a
// whitespace-controlled tag. Expectations below are jinja2 semantics,
// verified against CPython jinja2 3.x (transformers' rendering engine).

import Foundation
import Jinja
import Testing

@testable import Tokenizers

private func render(_ source: String) throws -> String {
    try Template(ChatTemplatePreprocessor.preprocess(source), with: .init(lstripBlocks: true, trimBlocks: true))
        .render([:])
}

@Suite("Chat template preprocessor")
struct ChatTemplatePreprocessorTests {
    @Test("Comment with left strip removes preceding whitespace")
    func commentLeftStrip() throws {
        // jinja2: "A  \n\n{#- c -#}B" renders "AB"
        #expect(try render("A  \n\n{#- c -#}B") == "AB")
    }

    @Test("Comment with right strip removes following whitespace")
    func commentRightStrip() throws {
        #expect(try render("A{#- c -#}  \n\nB") == "AB")
    }

    @Test("Comment without strip markers keeps surrounding whitespace")
    func commentNoStrip() throws {
        #expect(try render("A {# c #} B") == "A  B")
    }

    @Test("Comment between tags, gemma-4 pattern")
    func commentBetweenBlocks() throws {
        let source = "{%- if true -%}X{%- endif %}\n\n{#- note -#}\n{%- if true -%}Y{%- endif -%}"
        // jinja2: the `\n\n` between `%}` and `{#-` is stripped by the comment's left marker
        #expect(try render(source) == "XY")
    }

    @Test("Literal brace before stripped expression emits no space")
    func braceBeforeStrippedExpression() throws {
        // jinja2: "A{ {{- 'x' -}} }B" renders "A{x}B"
        #expect(try render("A{ {{- 'x' -}} }B") == "A{x}B")
        #expect(try render("{ {{- 'x' -}} }") == "{x}")
    }

    @Test("Literal brace before stripped statement emits no space")
    func braceBeforeStrippedStatement() throws {
        #expect(try render("A{ {%- if true %}x{% endif -%} B") == "A{xB")
    }

    @Test("Brace inside tag string literal is untouched")
    func braceInStringLiteral() throws {
        #expect(try render("{{ '{#-' }}") == "{#-")
        #expect(try render("{{ '{ x' }}") == "{ x")
    }

    @Test("Plain comment keeps segmentation; trim_blocks removes one newline after")
    func commentSegmentation() throws {
        // jinja2 (trim_blocks/lstrip_blocks): 'A\nB' — the comment blocks the
        // following {%- from stripping the newline that precedes the comment.
        #expect(try render("A\n{# c #}\nB") == "A\nB")
        #expect(try render("A\n{%- if true %}\n\n{# c #}\n{%- endif %}B") == "A\nB")
        // Without the comment, the newline is stripped.
        #expect(try render("A\n{%- if true %}\n\n{%- endif %}B") == "AB")
    }

    @Test("Unrelated constructs unchanged")
    func passThrough() throws {
        #expect(try render("A {{- 'x' }} B") == "Ax B")
        #expect(try render("A({{- 'x' -}})B") == "A(x)B")
        #expect(try render("plain text { no tags") == "plain text { no tags")
    }
}
