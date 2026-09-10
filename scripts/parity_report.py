#!/usr/bin/env python3
"""Aggregate results emitted by DifferentialTests; fail on missing cases or differences.

TOKENIZERS_PARITY_REPORT_DIR=/tmp/parity swift test -c release --filter DifferentialTests
python3 scripts/parity_report.py /tmp/parity

Use a fresh report directory for each run. The report covers checked-in goldens, not a
live HF run; token surfaces, offsets, and component suites are outside its counts.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "Tests/TokenizersTests/Resources"
SURFACES = ("ids", "idsNoSpecial", "decoded", "decodedSkipSpecial")


def aggregate(directory):
    fixtures = sorted((RESOURCES / "differential").glob("hf__*.json"))
    if not fixtures:
        raise ValueError("No HF reference fixtures found")
    rows = []
    for fixture in fixtures:
        name = fixture.stem.removeprefix("hf__")
        row = json.loads((directory / (name + ".json")).read_text())
        model = row["model"]
        if model.replace("/", "__") != name:
            raise ValueError(f"Unexpected model in {name}")
        expected_count = len(json.loads(fixture.read_text()))
        if row["cases"] != expected_count or expected_count == 0:
            raise ValueError(f"Incomplete case count for {model}")
        if set(row["differences"]) != set(SURFACES):
            raise ValueError(f"Incomplete surfaces for {model}")
        if any(row["differences"].values()):
            raise ValueError(f"Reference differences for {model}: {row['differences']}")
        row["goldenSHA256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        rows.append(row)
    return {
        "schemaVersion": 1,
        "scope": "Checked-in HF ID/decode goldens; excludes token surfaces, offsets, and component suites.",
        "references": {
            "idsAndDecoding": "Historical transformers goldens; exact generator versions were not recorded.",
        },
        "models": rows,
        "modelCount": len(rows),
        "casesPerSurface": sum(row["cases"] for row in rows),
        "differences": {key: sum(row["differences"][key] for row in rows) for key in SURFACES},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = aggregate(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"HF golden parity: {report['modelCount']} tokenizers, {report['casesPerSurface']:,} cases per surface.")
    print("\n| Surface | Differences |\n| --- | ---: |")
    for surface, count in report["differences"].items():
        print(f"| {surface} | {count} |")
    print("\nCounts cover checked-in references; token surfaces, offsets, and component suites run separately.")
    print("These historical ID/decode goldens predate exact generator-version metadata.")


if __name__ == "__main__":
    main()
