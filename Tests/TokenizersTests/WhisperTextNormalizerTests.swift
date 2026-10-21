import Foundation
import Testing

@testable import Tokenizers

@Suite("Whisper text normalizers match transformers")
struct WhisperTextNormalizerTests {
    /// Input, then `BasicTextNormalizer` output by default, removing diacritics and splitting letters,
    /// as recorded from `transformers.models.whisper.english_normalizer`.
    static let basicCases: [(String, String, String, String)] = [
        ("  \t\n  ", " ", " ", " "),
        (
            "<intro> Welcome [music] to (laughter) the show",
            " welcome to the show",
            " welcome to the show",
            " w e l c o m e t o t h e s h o w"
        ),
        ("a [b c] d [e", "a d e", "a d e", "a d e"),
        ("x () y (z) (( w ) v)", "x y v ", "x y v ", "x y v "),
        ("a(b)c(d", "ac d", "ac d", "a c d"),
        (
            "٣ apples and ٤.٥ pears",
            "٣ apples and ٤ ٥ pears",
            "٣ apples and ٤ ٥ pears",
            "٣ a p p l e s a n d ٤ ٥ p e a r s"
        ),
        ("１２３ ｆｕｌｌ ｗｉｄｔｈ", "123 full width", "123 full width", "1 2 3 f u l l w i d t h"),
        ("²³ ⅓ Ⅻ", "23 1 3 xii", "23 1 3 xii", "2 3 1 3 x i i"),
        ("a%b% c %", "a b c ", "a b c ", "a b c "),
        ("café résumé naïve", "café résumé naïve", "cafe resume naive", "c a f é r é s u m é n a ï v e"),
        ("œuvre Œdipe", "œuvre œdipe", "oeuvre oedipe", "œ u v r e œ d i p e"),
        ("Straße ẞ", "straße ß", "strasse ss", "s t r a ß e ß"),
        ("þorn ðorn Þ Ð", "þorn ðorn þ ð", "thorn dorn th d", "þ o r n ð o r n þ ð"),
        ("ﬁ ﬂ ﬃ", "fi fl ffi", "fi fl ffi", "f i f l f f i"),
        ("こんにちは 世界", "こんにちは 世界", "こんにちは 世界", "こ ん に ち は 世 界"),
        ("ह\u{93F}न\u{94D}द\u{940} द\u{947}वन\u{93E}गर\u{940}", "ह न द द वन गर ", "ह नद दवन गर ", "ह न द द व न ग र "),
        (
            "Ελληνικά ΟΔΥΣΣΕΥΣ ΣΑΣ Σ ΑΣ. ΑΣ'Σ",
            "ελληνικά οδυσσευς σας σ ας ασ ς",
            "ελληνικα οδυσσευς σας σ ας ασ ς",
            "ε λ λ η ν ι κ ά ο δ υ σ σ ε υ ς σ α ς σ α ς α σ ς"
        ),
        ("İstanbul ı I", "i stanbul ı i", "istanbul ı i", "i s t a n b u l ı i"),
        ("ǅ ǈ ǋ", "dž lj nj", "dz lj nj", "d ž l j n j"),
        ("\u{301} leading mark", " leading mark", " leading mark", " l e a d i n g m a r k"),
        ("a\u{20DD} b\u{488}", "a b ", "a b ", "a b "),
        ("क\u{94D}ष\u{93F} ष\u{94D}ट\u{94D}र", "क ष ष ट र", "कष षटर", "क ष ष ट र"),
        ("𐎹\u{301} \u{11F06}\u{345} a\u{1D400}\u{301}", "𐎹 \u{11F06} aá", "𐎹 \u{11F06} aa", "𐎹 \u{11F06} a á"),
        ("' 05.ß\n", " 05 ß ", " 05 ss ", " 0 5 ß "),
        ("œ 05thirties\n", "œ 05thirties ", "oe 05thirties ", "œ 0 5 t h i r t i e s "),
        ("café's ", "café s ", "cafe s ", "c a f é s "),
    ]

    /// Input and `EnglishTextNormalizer` output, as recorded from transformers.
    static let englishCases: [(String, String)] = [
        ("<intro> Welcome [music] to (laughter) the show", "welcome to the show"),
        ("x () y (z) (( w ) v)", "x y v"),
        ("He's like, um, gonna do it", "he is like going to do it"),
        ("hmm mm mhm mmm uh um", ""),
        ("UM, UH.", ""),
        ("won't worry", "will not worry"),
        ("Let's go", "let us go"),
        ("Mr. Smith met Dr. Jones and Mrs. Brown", "mister smith met doctor jones and missus brown"),
        ("you woulda coulda shoulda", "you would have could have should have"),
        ("she's been there", "she has been there"),
        ("the mr's hat", "the mister is hat"),
        ("can'tn't", "can not not"),
        ("o'clock rock'n'roll", "0 clock rock n roll"),
        ("twenty-seven dogs", "27 dogs"),
        ("five hundred and twenty-three", "523"),
        ("the second time", "the 2nd time"),
        ("three point one four", "3.14"),
        ("twenty million dollars", "$20000000"),
        ("It's $20 million in revenue", "it is $20000000 in revenue"),
        ("minus fifty", "-50"),
        ("double oh seven", "007"),
        ("two and a half", "2.5"),
        ("$1,234.56", "$1234.56"),
        ("five dollars and seven cents", "$5.07"),
        ("$0.07", "¢7"),
        ("€0.5 and €0.05", "¢5 and ¢5"),
        ("one decillion", "1000000000000000000000000000000000"),
        ("two thousand five hundred and sixty first", "2561st"),
        ("the nineteen eighties", "the 1980s"),
        ("plus one minus two positive three negative", "+one -2 +3 negative"),
        ("one cent", " one"),
        ("192.168.0.1", "192.168.0 one"),
        ("a1b2c3", "a one b 2 c 3"),
        ("$5 $-5 -$5 +5 +$5", "$5 5 $5 5 $5"),
        ("٣ apples and ٤.٥ pears", "3 apples and 4 5 pears"),
        ("²³ ⅓ Ⅻ", "23 one 3 xii"),
        ("The colour of the centre", "the color of the center"),
        ("Colour COLOUR colour.", "color color color"),
        ("Don't worry — twenty-seven dogs ran past!", "do not worry 27 dogs ran past"),
        ("e.g. i.e. etc.", "e g i e etc"),
        ("café résumé naïve", "cafe resume naive"),
        ("Straße ẞ", "strasse ss"),
        ("Ελληνικά ΟΔΥΣΣΕΥΣ ΣΑΣ Σ ΑΣ. ΑΣ'Σ", "ελληνικα οδυσσευς σας σ ας ασ ς"),
        ("İstanbul ı I", "istanbul ı i"),
        ("क\u{94D}ष\u{93F} ष\u{94D}ट\u{94D}र", "कष षटर"),
        ("𐎹\u{301} \u{11F06}\u{345} a\u{1D400}\u{301}", "𐎹 \u{11F06} aA"),
        ("' 05.ß\n", "5 ss"),
    ]

    /// Optional larger replay: a JSON file of `{"cases": [{"input", "basic", "basic_remove_diacritics",
    /// "basic_split_letters", "english"}]}` named by `WHISPER_NORMALIZER_GOLDEN`.
    static let golden: [[String: String]] = {
        guard let path = ProcessInfo.processInfo.environment["WHISPER_NORMALIZER_GOLDEN"] else { return [] }
        let json = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
        return (json as! [String: [[String: String]]])["cases"]!
    }()

    /// The spelling mapping is the `normalizer.json` of Whisper repositories.
    static let english = Task {
        let folder = try await HubFixtures.modelFolder(for: "openai/whisper-tiny.en", files: ["normalizer.json"])
        return try EnglishTextNormalizer(contentsOf: folder.appending(path: "normalizer.json"))
    }

    @Test("BasicTextNormalizer", arguments: basicCases)
    func basic(_ test: (String, String, String, String)) {
        #expect(BasicTextNormalizer()(test.0) == test.1)
        #expect(BasicTextNormalizer(removeDiacritics: true)(test.0) == test.2)
        #expect(BasicTextNormalizer(splitLetters: true)(test.0) == test.3)
    }

    @Test("EnglishTextNormalizer", arguments: englishCases)
    func english(_ test: (String, String)) async throws {
        #expect(try await Self.english.value(test.0) == test.1)
    }

    @Test("Recorded golden file", .enabled(if: !golden.isEmpty))
    func recorded() async throws {
        let english = try await Self.english.value
        for test in Self.golden {
            let input = test["input"]!
            #expect(BasicTextNormalizer()(input) == test["basic"], "\(input.debugDescription)")
            #expect(BasicTextNormalizer(removeDiacritics: true)(input) == test["basic_remove_diacritics"])
            #expect(BasicTextNormalizer(splitLetters: true)(input) == test["basic_split_letters"])
            #expect(english(input) == test["english"], "\(input.debugDescription)")
        }
    }

    @Test("Documented examples")
    func examples() async throws {
        let english = try await Self.english.value
        #expect(BasicTextNormalizer()("Hello, World! [music]") == "hello world ")
        #expect(BasicTextNormalizer()("¿Qué tal? [risas]") == " qué tal ")
        #expect(BasicTextNormalizer(removeDiacritics: true)("¿Qué tal? [risas]") == " que tal ")
        #expect(BasicTextNormalizer(splitLetters: true)("你好") == "你 好")
        #expect(english("Mr. Smith won't pay twenty five dollars.") == "mister smith will not pay $25")
        #expect(english("The colour of the centre") == "the color of the center")
    }
}
