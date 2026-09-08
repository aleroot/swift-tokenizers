import Foundation
import Tokenizers

let arg = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "mlx-community/Qwen3-0.6B-Base-DQ5"
let folder =
    arg.hasPrefix("/")
    ? URL(fileURLWithPath: arg)
    : FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("swift-tokenizers-tests/" + arg)
let tokenizer = try AutoTokenizer.load(from: folder)
let para =
    "Byte-pair encoding (BPE) is a tokenization algorithm originally proposed for data compression by Philip Gage in 1994. It was later adapted for use in neural machine translation by Sennrich, Haddow, and Birch in 2015, and is now the dominant sub-word tokenization scheme for modern large language models including the GPT, Llama, Qwen, and Mistral families. The algorithm operates by iteratively replacing the most frequent adjacent pair of bytes in a corpus with a new symbol, building up a vocabulary of merges that compactly represents both common words and rare strings.\n\n"
let text = String(repeating: para, count: 200)
var total = 0
let start = Date()
while Date().timeIntervalSince(start) < 6 {
    total += tokenizer.encode(text: text, addSpecialTokens: false).count
}
print("tokens:", total)
