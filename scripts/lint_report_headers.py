#!/usr/bin/env python3
"""
Structural Lint Checker for VQ-bench Research Reports.
Enforces mandatory header metadata and pre-registration invariants across all docs/content/*.md files.
"""

import os
import sys
import glob
import re

MANDATORY_FIELDS = [
    r"\*\*Status:\*\*\s*(EXPLORATORY|CONFIRMATORY)",
    r"\*\*Forecast file:\*\*\s*(.+)",
    r"\*\*Labels & Provenance:\*\*\s*(.+)",
]

def check_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []

    # Check for mandatory header fields
    for pattern in MANDATORY_FIELDS:
        if not re.search(pattern, content, re.IGNORECASE):
            errors.append(f"Missing mandatory header matching: {pattern}")

    # Check Status vs Forecast Invariant
    status_match = re.search(r"\*\*Status:\*\*\s*(EXPLORATORY|CONFIRMATORY)", content, re.IGNORECASE)
    forecast_match = re.search(r"\*\*Forecast file:\*\*\s*(.+)", content, re.IGNORECASE)

    if status_match and forecast_match:
        status = status_match.group(1).upper()
        forecast = forecast_match.group(1).strip()

        if status == "CONFIRMATORY":
            if forecast.upper() == "NONE" or "docs/forecasts" not in forecast:
                errors.append("Status is CONFIRMATORY but Forecast file does not point to a valid 'docs/forecasts/...' document.")
        elif status == "EXPLORATORY":
            # Check for verdict inflation in exploratory documents
            if re.search(r"production systems should adopt", content, re.IGNORECASE):
                errors.append("Exploratory document contains ungrounded claim: 'production systems should adopt'. Must be 'we hypothesize, pending confirmation'.")
            if re.search(r"confirms that", content, re.IGNORECASE):
                errors.append("Exploratory document uses 'confirms that' instead of 'suggests the hypothesis that'.")

    return errors

def is_research_report(fp):
    basename = os.path.basename(fp)
    doc_pages = [
        "overview.md", "usage.md", "add-a-primitive.md", "add-a-quantizer.md",
        "advanced-quantizers.md", "research-charter.md", "phase2-research-charter.md",
        "domain-entropy-code-compression-paper.md"
    ]
    if basename in doc_pages:
        return False
    return basename.endswith("-report.md") or basename.endswith("-study.md") or basename.endswith("-memo.md") or basename.endswith("-ledger.md") or basename.endswith("-findings.md") or basename.endswith("-summary.md")

def main():
    all_files = glob.glob("docs/content/*.md")
    report_files = [fp for fp in all_files if is_research_report(fp)]
    if not report_files:
        print("No report files found in docs/content/")
        sys.exit(1)

    all_passed = True
    print("=" * 90)
    print(" VQ-BENCH REPORT HEADER & PRE-REGISTRATION LINTER")
    print("=" * 90)

    for fp in sorted(report_files):
        errors = check_file(fp)
        if errors:
            all_passed = False
            print(f"\n❌ FAIL: {fp}")
            for err in errors:
                print(f"   └── Error: {err}")
        else:
            print(f"✅ PASS: {fp}")

    print("=" * 90)
    if all_passed:
        print("All research reports satisfy structural header and pre-registration invariants.")
        sys.exit(0)
    else:
        print("Report linter detected structural defects. Fix above errors before committing.")
        sys.exit(1)

if __name__ == "__main__":
    main()
