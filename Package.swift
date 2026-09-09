// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "swift-tokenizers",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "Tokenizers", targets: ["Tokenizers"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.0"),
    ],
    targets: [
        .target(
            name: "Tokenizers",
            dependencies: [
                .product(name: "Jinja", package: "swift-jinja"),
            ],
            resources: [
                .process("Resources"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("ExistentialAny"),
            ]
        ),
        .testTarget(
            name: "TokenizersTests",
            dependencies: ["Tokenizers"],
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "Benchmarks",
            dependencies: ["Tokenizers"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
