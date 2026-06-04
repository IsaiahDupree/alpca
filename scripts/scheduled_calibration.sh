#!/bin/zsh -l
# Scheduled market-open calibration runner (fired by launchd at ~09:36 ET).
#
# Safety: the pipeline itself re-checks Alpaca's clock and refuses to trade if the
# market isn't actually open, so a holiday / early-close / half-day can't cause a
# bad run. This wrapper just sets cwd, picks the venv python, timestamps the log,
# and (after a successful run) disables the one-shot job so it doesn't re-fire.

set -u
ALPCA_DIR="/Users/isaiahdupree/Documents/Software/Alpca"
PY="$ALPCA_DIR/.venv/bin/python"
LOG="$ALPCA_DIR/data/calibration_run.log"
LABEL="com.alpca.calibration"

cd "$ALPCA_DIR" || exit 1
mkdir -p "$ALPCA_DIR/data"

{
  echo "=================================================================="
  echo "[scheduled-calibration] fired at $(date '+%Y-%m-%d %H:%M:%S %Z')"
} >> "$LOG" 2>&1

# Run the full pipeline (collect real fills -> fit -> write calibration.json -> parity).
# Credentials load from Alpca/.env automatically. SPY, 16 cycles across sizes 1,2,3.
"$PY" scripts/run_calibration_pipeline.py --symbol SPY --cycles 16 --sizes 1,2,3 \
  >> "$LOG" 2>&1
RC=$?

echo "[scheduled-calibration] exit=$RC at $(date '+%H:%M:%S %Z')" >> "$LOG" 2>&1

# One-shot: if it completed (market was open, rc=0), unload the job so it won't
# fire again on subsequent days. rc=2 means market was closed -> leave it armed.
if [ "$RC" -eq 0 ]; then
  echo "[scheduled-calibration] success — unloading one-shot launchd job $LABEL" >> "$LOG" 2>&1
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null \
    || launchctl unload "$HOME/Library/LaunchAgents/$LABEL.plist" 2>/dev/null
fi
exit $RC
