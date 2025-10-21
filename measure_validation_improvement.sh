#!/usr/bin/env bash
# Measurement script for validation improvement comparison
#
# Compares quote validation warnings between:
# - Baseline: commit d8b29bb (before validators)
# - Enhanced: current HEAD (with validators)

set -euo pipefail

WORKDIR="/home/tec/Documents/PhD/Projects/InteractionReconstruction/interaction_finder"
BASELINE_COMMIT="d8b29bb"
ENHANCED_COMMIT="HEAD"
BASELINE_WORKTREE="$WORKDIR/.claude/worktrees/v3-adopt-v2-validation/04-baseline"
RESULTS_DIR="$WORKDIR/.claude/worktrees/v3-adopt-v2-validation/04/validation-results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== V3 Validation Improvement Measurement ===${NC}"
echo ""
echo "Baseline commit: $BASELINE_COMMIT (before validators)"
echo "Enhanced commit: $ENHANCED_COMMIT (with validators)"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Test command to execute
CONFIG_PATH="/home/tec/Documents/PhD/Projects/InteractionReconstruction/casestudies/disease-gene/config.toml"
TEST_CMD="uv run --env-file=$WORKDIR/.env.local interaction-finder extract --config $CONFIG_PATH --term 'pulmonary arterial hypertension' -p1 -v"

echo -e "${YELLOW}Step 1: Running baseline measurement (without validators)${NC}"
echo "Creating temporary worktree at baseline commit..."

# Clean up any existing baseline worktree
git worktree remove "$BASELINE_WORKTREE" --force 2>/dev/null || true
git worktree add "$BASELINE_WORKTREE" "$BASELINE_COMMIT" --detach

cd "$BASELINE_WORKTREE"

echo "Running extraction (this may take several minutes)..."
START_TIME=$(date +%s)

# Run baseline test, capturing all output
$TEST_CMD 2>&1 | tee "$RESULTS_DIR/baseline-full.log"
BASELINE_EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
BASELINE_DURATION=$((END_TIME - START_TIME))

echo "Baseline completed in ${BASELINE_DURATION}s"
echo ""

# Count validation warnings in baseline
BASELINE_WARNINGS=$(grep -c "Quote validation failed" "$RESULTS_DIR/baseline-full.log" || echo "0")
echo -e "${GREEN}Baseline warnings: ${BASELINE_WARNINGS}${NC}"

# Extract baseline metrics
echo "{\"warnings\": $BASELINE_WARNINGS, \"duration_s\": $BASELINE_DURATION, \"exit_code\": $BASELINE_EXIT_CODE, \"commit\": \"$BASELINE_COMMIT\"}" > "$RESULTS_DIR/baseline-metrics.json"

# Return to main worktree
cd "$WORKDIR/.claude/worktrees/v3-adopt-v2-validation/04"

echo ""
echo -e "${YELLOW}Step 2: Running enhanced measurement (with validators)${NC}"
echo "Running extraction with validators active..."

START_TIME=$(date +%s)

# Run enhanced test
$TEST_CMD 2>&1 | tee "$RESULTS_DIR/enhanced-full.log"
ENHANCED_EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
ENHANCED_DURATION=$((END_TIME - START_TIME))

echo "Enhanced completed in ${ENHANCED_DURATION}s"
echo ""

# Count validation warnings in enhanced
ENHANCED_WARNINGS=$(grep -c "Quote validation failed" "$RESULTS_DIR/enhanced-full.log" || echo "0")
echo -e "${GREEN}Enhanced warnings: ${ENHANCED_WARNINGS}${NC}"

# Count ModelRetry feedback messages (expected with validators)
RETRY_MESSAGES=$(grep -c "ModelRetry" "$RESULTS_DIR/enhanced-full.log" || echo "0")
echo -e "${BLUE}Retry feedback messages: ${RETRY_MESSAGES}${NC}"

# Extract enhanced metrics
echo "{\"warnings\": $ENHANCED_WARNINGS, \"duration_s\": $ENHANCED_DURATION, \"exit_code\": $ENHANCED_EXIT_CODE, \"commit\": \"$(git rev-parse HEAD)\", \"retry_messages\": $RETRY_MESSAGES}" > "$RESULTS_DIR/enhanced-metrics.json"

echo ""
echo -e "${YELLOW}Step 3: Comparison${NC}"

# Calculate improvement
if [ "$BASELINE_WARNINGS" -gt 0 ]; then
    REDUCTION=$(python3 -c "print(f'{(($BASELINE_WARNINGS - $ENHANCED_WARNINGS) / $BASELINE_WARNINGS * 100):.1f}')")
    TIME_INCREASE=$(python3 -c "print(f'{(($ENHANCED_DURATION - $BASELINE_DURATION) / $BASELINE_DURATION * 100):.1f}')")
else
    REDUCTION="N/A"
    TIME_INCREASE="N/A"
fi

echo "Baseline warnings:  $BASELINE_WARNINGS"
echo "Enhanced warnings:  $ENHANCED_WARNINGS"
echo "Reduction:          ${REDUCTION}%"
echo ""
echo "Baseline time:      ${BASELINE_DURATION}s"
echo "Enhanced time:      ${ENHANCED_DURATION}s"
echo "Time increase:      ${TIME_INCREASE}%"
echo ""

# Save comparison
cat > "$RESULTS_DIR/comparison.json" <<EOF
{
  "baseline": {
    "warnings": $BASELINE_WARNINGS,
    "duration_s": $BASELINE_DURATION,
    "commit": "$BASELINE_COMMIT"
  },
  "enhanced": {
    "warnings": $ENHANCED_WARNINGS,
    "duration_s": $ENHANCED_DURATION,
    "retry_messages": $RETRY_MESSAGES,
    "commit": "$(git rev-parse HEAD)"
  },
  "improvement": {
    "warning_reduction_pct": "$REDUCTION",
    "time_increase_pct": "$TIME_INCREASE",
    "warnings_reduced": $((BASELINE_WARNINGS - ENHANCED_WARNINGS))
  }
}
EOF

# Check success criteria
echo -e "${YELLOW}Success Criteria Check:${NC}"

if [ "$REDUCTION" != "N/A" ]; then
    REDUCTION_NUM=$(echo "$REDUCTION" | tr -d '%')
    if (( $(echo "$REDUCTION_NUM >= 30" | bc -l) )); then
        echo -e "${GREEN}✓ Warning reduction ≥30%: ${REDUCTION}%${NC}"
        SUCCESS_WARNINGS=true
    else
        echo -e "${RED}✗ Warning reduction <30%: ${REDUCTION}% (target: ≥30%)${NC}"
        SUCCESS_WARNINGS=false
    fi
fi

if [ "$TIME_INCREASE" != "N/A" ]; then
    TIME_NUM=$(echo "$TIME_INCREASE" | tr -d '%')
    if (( $(echo "$TIME_NUM <= 25" | bc -l) )); then
        echo -e "${GREEN}✓ Time increase ≤25%: ${TIME_INCREASE}%${NC}"
        SUCCESS_TIME=true
    else
        echo -e "${YELLOW}⚠ Time increase >25%: ${TIME_INCREASE}% (target: ≤25%)${NC}"
        SUCCESS_TIME=false
    fi
fi

echo ""
echo -e "${BLUE}Results saved to: $RESULTS_DIR${NC}"
echo "  - baseline-full.log: Complete baseline output"
echo "  - enhanced-full.log: Complete enhanced output"
echo "  - baseline-metrics.json: Baseline metrics"
echo "  - enhanced-metrics.json: Enhanced metrics"
echo "  - comparison.json: Comparison summary"

# Cleanup baseline worktree
echo ""
echo "Cleaning up baseline worktree..."
git worktree remove "$BASELINE_WORKTREE" --force

echo ""
echo -e "${GREEN}Measurement complete!${NC}"
