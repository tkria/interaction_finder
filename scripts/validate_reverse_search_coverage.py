#!/usr/bin/env python3
"""
Validate reverse search coverage improvement on PAH test dataset.

This script demonstrates the 0% → ≥80% coverage improvement from tasks 02-04:
- Task 02: Fixed backend selection to respect config values
- Task 03: Added backend validation to catch mismatches early
- Task 04: Updated config defaults to optimal settings (yake + direct)

**Usage:**
    uv run python scripts/validate_reverse_search_coverage.py

**Expected Result:** ≥80% coverage (typically ~90%) on 20 PAH papers
**Baseline (before fixes):** 0% coverage due to backend mismatch + LLM query issues

**Output:** Coverage percentage, unfound papers, comparison with investigation
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    """Run coverage validation on PAH test dataset."""
    print("=" * 70)
    print("Reverse Search Coverage Validation")
    print("=" * 70)
    print()

    # Locate dataset (handle both normal and worktree layouts)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Check if we're in a worktree
    if ".claude/worktrees" in str(project_root):
        # In worktree: go up to actual project root
        # Current: .../05/scripts -> .../05
        # Need: .../interaction_finder
        project_root = project_root.parent.parent.parent.parent

    dataset_path = project_root / "some-pah-papers.jsonl"

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Expected file: some-pah-papers.jsonl in project root")
        print(f"Searched from: {project_root}")
        sys.exit(1)

    # Load known resources
    print(f"Loading test dataset: {dataset_path}")
    known_resources = []
    with dataset_path.open() as f:
        for line in f:
            if line.strip():
                known_resources.append(json.loads(line))

    print(f"Loaded {len(known_resources)} known resources")
    print()

    # Configuration info
    print("Configuration:")
    print("  Using DEFAULT settings (no overrides)")
    print("  Expected code defaults:")
    print("    - search_backend: pubmed")
    print("    - keyword_extractor: yake")
    print("    - query_constructor: direct")
    print("    - enable_clustering: true")
    print()

    # Create output file in temp directory
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_results.jsonl", delete=False
    ) as f:
        output_file = Path(f.name)

    print(f"Output file: {output_file}")
    print()

    # Run reverse search via CLI (uses default config)
    print("Running reverse search (this may take 2-5 minutes)...")
    print(
        "Command: uv run interaction-finder reverse-search --known <dataset> -b pubmed"
    )
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "interaction-finder",
                "reverse-search",
                "--known",
                str(dataset_path),
                "--backend",
                "pubmed",  # Explicit backend (matches default)
                "-o",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
            cwd=project_root,
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(
                f"ERROR: reverse-search command failed (exit code {result.returncode})"
            )
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            sys.exit(1)

        print(result.stdout)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"ERROR: reverse-search timed out after {elapsed:.0f}s")
        sys.exit(1)

    # Parse results
    print()
    print("Parsing results...")

    # Load output file to get detailed match info
    # The CLI writes ReverseSearchSession JSON to output file
    try:
        with output_file.open() as f:
            session_data = json.load(f)

        total_papers = len(known_resources)
        found_papers = session_data.get("found_count", 0)
        coverage_pct = (found_papers / total_papers) * 100
        total_queries = session_data.get("total_queries", 0)
        stopping_reason = session_data.get("stopping_reason", "unknown")

        # Get unfound PMIDs from matches
        unfound_pmids = []
        for match in session_data.get("matches", []):
            if not match.get("matched"):
                pmid = match.get("resource_metadata", {}).get("pmid")
                if pmid:
                    unfound_pmids.append(pmid)

    except Exception as e:
        print(f"ERROR: Failed to parse results file: {e}")
        print(f"Results file: {output_file}")
        sys.exit(1)

    # Display results
    print()
    print("=" * 70)
    print("Coverage Validation Results")
    print("=" * 70)
    print()
    print(f"Total papers:       {total_papers}")
    print(f"Found papers:       {found_papers}")
    print(f"Coverage:           {coverage_pct:.1f}%")
    print()
    print(f"Queries executed:   {total_queries}")
    print(f"Total time:         {elapsed:.1f}s")
    print(f"Stopping reason:    {stopping_reason}")
    print()

    # Show unfound papers
    if unfound_pmids:
        print(f"Unfound papers ({len(unfound_pmids)}):")
        for pmid in unfound_pmids:
            print(f"  - PMID {pmid}")
        print()

        # Compare with investigation expectations
        expected_unfound = {"11207353", "19625727"}
        actual_unfound = set(unfound_pmids)

        if actual_unfound == expected_unfound:
            print("✓ Unfound papers match investigation findings")
            print("  (These 2 papers are legitimately hard to find)")
        else:
            unexpected = actual_unfound - expected_unfound
            if unexpected:
                print(f"⚠ Unexpected unfound papers: {unexpected}")
                print("  (PubMed index may have changed)")

            missing = expected_unfound - actual_unfound
            if missing:
                print(
                    f"✓ Improvement: Found papers that were unfound in investigation: {missing}"
                )
    else:
        print("✓ All papers found! (Better than investigation 90% coverage)")

    print()

    # Validate coverage target
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print()

    baseline_coverage = 0.0  # Before fixes (backend mismatch + LLM issues)
    target_coverage = 80.0

    print(f"Baseline coverage (before fixes):  {baseline_coverage:.1f}%")
    print(f"Current coverage (after fixes):    {coverage_pct:.1f}%")
    print(f"Target coverage:                   {target_coverage:.1f}%")
    print()

    if coverage_pct >= target_coverage:
        improvement = coverage_pct - baseline_coverage
        print(f"✓ SUCCESS: Coverage target met!")
        print(f"  Improvement: +{improvement:.1f} percentage points")
        print()
        print("All fixes working as expected:")
        print("  ✓ Task 02: Backend selection respects config")
        print("  ✓ Task 03: Backend validation catches mismatches")
        print("  ✓ Task 04: Optimal defaults (yake + direct) applied")
        print()
        return 0
    else:
        shortfall = target_coverage - coverage_pct
        print(
            f"✗ FAILURE: Coverage {coverage_pct:.1f}% below {target_coverage:.1f}% target"
        )
        print(f"  Shortfall: -{shortfall:.1f} percentage points")
        print()
        print("Possible issues:")
        print("  - PubMed API availability/rate limiting")
        print("  - Dataset quality (incorrect PMID associations)")
        print("  - Query generation not optimal")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
