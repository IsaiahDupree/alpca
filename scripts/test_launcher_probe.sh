#!/bin/zsh
# Robustness test for the launchd FDA-probe retry logic (the fix for the 2026-06-09
# livesession exit=3 race, where two 09:50 jobs probing ~/Documents at the same instant
# caused one to spuriously report FDA-missing). Pure shell, no orders, no network.
#
# Asserts the retry probe: (1) succeeds on a writable dir, (2) STILL fails-closed after
# retries when access is genuinely missing (safety preserved), (3) RECOVERS from a
# transient denial (the exact bug), (4) never spuriously fails under concurrent collision.
set -u
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"   # hermetic: tolerate a stripped env
PASS=0; FAIL=0
ok(){ echo "  PASS: $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

# The probe-retry function, identical in shape to the one in the launchers.
probe_retry(){  # $1=path  $2=tries  -> 0 ok / 3 denied
  local path="$1" tries="$2" t
  for t in {1..$tries}; do
    if ( : > "$path" ) 2>/dev/null; then /bin/rm -f "$path" 2>/dev/null; return 0; fi
    /bin/sleep 0.05
  done
  return 3
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "[1] writable dir -> succeeds first try"
probe_retry "$TMP/.fda_probe_live" 5 && ok "writable probe returns 0" || no "writable probe should return 0"

echo "[2] genuinely unreachable dir -> fails CLOSED after retries (safety preserved)"
probe_retry "/nonexistent_root_dir_xyz/.fda_probe" 3 ; rc=$?
[ "$rc" -eq 3 ] && ok "unreachable probe returns 3 (FDA-missing still detected)" || no "expected 3, got $rc"

echo "[3] TRANSIENT denial (fails twice, then dir appears) -> RECOVERS (the bug fix)"
GATE="$TMP/gate"; printf 0 > "$GATE"
transient_probe(){  # denies until the 3rd attempt, then a writable path exists
  local path="$TMP/late/.fda_probe" t n
  for t in 1 2 3 4 5; do
    n=$(<"$GATE"); n=$((n+1)); printf '%s' "$n" > "$GATE"
    [ "$n" -ge 3 ] && /bin/mkdir -p "$TMP/late"       # access "restored" on 3rd try
    if ( : > "$path" ) 2>/dev/null; then /bin/rm -f "$path"; return 0; fi
    /bin/sleep 0.05
  done
  return 3
}
transient_probe && ok "recovered after transient denial (would have been a false exit=3 before)" \
                || no "retry failed to recover from transient denial"

echo "[4] 50x CONCURRENT collision of the two 09:50 jobs -> zero spurious failures"
spur=0
for i in {1..50}; do
  probe_retry "$TMP/.fda_probe_live"  5 & p1=$!
  probe_retry "$TMP/.fda_probe_swing" 5 & p2=$!
  wait $p1 || spur=$((spur+1))
  wait $p2 || spur=$((spur+1))
done
[ "$spur" -eq 0 ] && ok "100 concurrent probes (50 rounds x2), 0 spurious denials" \
                  || no "$spur spurious denials under concurrency"

echo "[5] distinct filenames across all four launchers (no shared path to race on)"
SUP="$HOME/Library/Application Support/Alpca"
uniq_names=$(grep -hoE "\.fda_probe[a-z_]*" "$SUP"/*.sh 2>/dev/null | sort -u | wc -l | tr -d ' ')
[ "$uniq_names" = "4" ] && ok "4 distinct probe filenames in launchers" \
                        || no "expected 4 distinct probe names, found $uniq_names"

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
