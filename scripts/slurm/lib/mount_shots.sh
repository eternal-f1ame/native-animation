# Mount shots.sqsh (if built) to node-local scratch and export the extra root
# for pack-aware readers. Source AFTER paths.env, and last among prologues —
# it installs an EXIT trap (reader sbatches keep no other EXIT trap).
SQSH="$SAKUGA_ROOT/shots/shots.sqsh"
if [ -f "$SQSH" ] && command -v squashfuse >/dev/null 2>&1; then
  NA_SQSH_MNT="${SLURM_TMPDIR:-/tmp}/na-shots-mnt-${SLURM_JOB_ID:-$$}"
  mkdir -p "$NA_SQSH_MNT"
  if squashfuse "$SQSH" "$NA_SQSH_MNT"; then
    export NA_SHOTS_EXTRA_ROOTS="$NA_SQSH_MNT"
    trap 'fusermount3 -u "$NA_SQSH_MNT" 2>/dev/null || fusermount -u "$NA_SQSH_MNT" 2>/dev/null || true' EXIT
    echo "shots.sqsh mounted at $NA_SQSH_MNT"
  else
    echo "WARN: squashfuse mount failed; readers fall back to loose/pack lookup" >&2
  fi
fi
