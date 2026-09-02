# Mount every shots_*.sqsh image to node-local scratch and export the
# colon-joined roots for pack-aware readers. Source AFTER paths.env, and last
# among prologues — it installs an EXIT trap (reader sbatches keep no other
# EXIT trap).
NA_SQSH_ROOTS=""
NA_SQSH_MNT_BASE="${SLURM_TMPDIR:-/tmp}/na-shots-mnt-${SLURM_JOB_ID:-$$}"
for _sqsh in "$SAKUGA_ROOT"/shots/shots_*.sqsh; do
  [ -f "$_sqsh" ] || continue
  command -v squashfuse >/dev/null 2>&1 || break
  _mnt="$NA_SQSH_MNT_BASE/$(basename "$_sqsh" .sqsh)"
  mkdir -p "$_mnt"
  if squashfuse "$_sqsh" "$_mnt"; then
    NA_SQSH_ROOTS="${NA_SQSH_ROOTS:+$NA_SQSH_ROOTS:}$_mnt"
    echo "mounted $(basename "$_sqsh") at $_mnt"
  else
    echo "WARN: squashfuse failed for $_sqsh; readers fall back to loose/pack lookup" >&2
  fi
done
if [ -n "$NA_SQSH_ROOTS" ]; then
  export NA_SHOTS_EXTRA_ROOTS="$NA_SQSH_ROOTS"
  trap 'for _m in "$NA_SQSH_MNT_BASE"/*; do fusermount3 -u "$_m" 2>/dev/null || fusermount -u "$_m" 2>/dev/null || true; done' EXIT
fi
