// Test-only helper that fetches tokenizer files from the Hugging Face Hub into a local
// cache directory. The library itself has no download code; the upstream swift-transformers
// test-suite relies on `HubApi`, so this minimal fetcher keeps those tests runnable.
//
// Set `HF_TOKEN` to access gated repositories. Files are cached under
// `~/Library/Caches/swift-tokenizers-tests/<repo>` (or `SWIFT_TOKENIZERS_FIXTURES`).

import Foundation
import Tokenizers

enum HubFixtures {
    static let tokenizerFiles = [
        "config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.json", "chat_template.jinja",
    ]

    static let cacheRoot: URL = {
        if let override = ProcessInfo.processInfo.environment["SWIFT_TOKENIZERS_FIXTURES"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appendingPathComponent("swift-tokenizers-tests", isDirectory: true)
    }()

    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 600
        return URLSession(configuration: configuration)
    }()

    /// Downloads (or reuses) the tokenizer files of `repo` and returns the folder URL.
    static func modelFolder(
        for repo: String, revision: String = "main", files: [String] = tokenizerFiles
    ) async throws -> URL {
        try await Downloader.shared.modelFolder(repo: repo, revision: revision, files: files)
    }

    /// Loads a tokenizer for a Hub repository.
    static func tokenizer(for repo: String, strict: Bool = true) async throws -> any Tokenizer {
        let folder = try await modelFolder(for: repo)
        return try await AutoTokenizer.from(modelFolder: folder, strict: strict)
    }

    /// Loads a tokenizer and casts it to `PreTrainedTokenizer`.
    static func preTrainedTokenizer(for repo: String, strict: Bool = true) async throws -> PreTrainedTokenizer {
        guard let tokenizer = try await tokenizer(for: repo, strict: strict) as? PreTrainedTokenizer else {
            throw FixtureError.unsupportedTokenizer
        }
        return tokenizer
    }

    /// Loads the resolved configuration for a Hub repository.
    static func configuration(for repo: String) async throws -> LocalModelConfiguration {
        try LocalModelConfiguration(modelFolder: try await modelFolder(for: repo))
    }

    enum FixtureError: Error, CustomStringConvertible {
        case httpError(Int, URL)
        case unsupportedTokenizer

        var description: String {
            switch self {
            case let .httpError(code, url): "HTTP \(code) for \(url)"
            case .unsupportedTokenizer: "Tokenizer is not a PreTrainedTokenizer"
            }
        }
    }

    /// Serialises downloads per repository so concurrent tests share one fetch.
    private actor Downloader {
        static let shared = Downloader()
        private var inFlight: [String: Task<URL, Error>] = [:]

        func modelFolder(repo: String, revision: String, files: [String]) async throws -> URL {
            let key = "\(repo)@\(revision)"
            if let task = inFlight[key] {
                return try await task.value
            }
            let task = Task<URL, Error> {
                try await HubFixtures.fetch(repo: repo, revision: revision, files: files)
            }
            inFlight[key] = task
            return try await task.value
        }
    }

    private static func fetch(repo: String, revision: String, files: [String]) async throws -> URL {
        let folder = cacheRoot.appendingPathComponent(repo, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        // A marker records which files were checked so missing optional files are not re-requested.
        let marker = folder.appendingPathComponent(".fetched")
        let alreadyChecked: Set<String> =
            (try? String(contentsOf: marker, encoding: .utf8))
            .map { Set($0.split(separator: "\n").map(String.init)) } ?? []

        var checked = alreadyChecked
        for file in files where !alreadyChecked.contains(file) {
            let destination = folder.appendingPathComponent(file)
            if FileManager.default.fileExists(atPath: destination.path) {
                checked.insert(file)
                continue
            }
            let url = URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(file)")!
            var request = URLRequest(url: url)
            if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { continue }
            switch http.statusCode {
            case 200:
                try data.write(to: destination, options: .atomic)
                checked.insert(file)
            case 404:
                checked.insert(file)  // optional file absent from the repo
            default:
                throw FixtureError.httpError(http.statusCode, url)
            }
        }
        try checked.sorted().joined(separator: "\n").write(to: marker, atomically: true, encoding: .utf8)
        return folder
    }
}
