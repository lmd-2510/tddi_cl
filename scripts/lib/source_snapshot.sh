#!/usr/bin/env bash
# Immutable source snapshot helpers for long-running experiments.
#
# The snapshot contains exactly one committed Git tree. Runtime datasets and
# outputs remain outside it and are linked explicitly. A run must resume from
# the same snapshot, so later git pulls in the developer checkout cannot change
# member code or invalidate implementation hashes halfway through an ensemble.

cil_snapshot_die() {
  echo "[STOP] $*" >&2
  return 2
}

cil_assert_snapshot_source_clean() {
  local root="$1" tracked untracked
  git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || cil_snapshot_die "Source snapshot requires a Git working tree: $root" \
    || return 2
  tracked="$(git -C "$root" status --porcelain --untracked-files=no -- \
    src scripts configs requirements.txt \
    study_assets/task_protocols study_assets/data_schema)"
  [[ -z "$tracked" ]] \
    || cil_snapshot_die "Tracked source/config changes are uncommitted; commit them before starting a reproducible run." \
    || return 2
  untracked="$(git -C "$root" ls-files --others --exclude-standard -- \
    src scripts configs requirements.txt \
    study_assets/task_protocols study_assets/data_schema)"
  [[ -z "$untracked" ]] \
    || cil_snapshot_die "Untracked implementation/config files would be absent from the snapshot: $untracked" \
    || return 2
}

cil_create_source_snapshot() {
  local root="$1" run_home="$2" snapshot archive commit tree archive_sha name
  root="$(cd "$root" && pwd)" || return 2
  run_home="$(mkdir -p "$run_home" && cd "$run_home" && pwd)" || return 2
  snapshot="$run_home/source_snapshot"
  archive="$run_home/.source_snapshot.tar"
  [[ ! -e "$snapshot" && ! -e "$archive" ]] \
    || cil_snapshot_die "Source snapshot namespace already exists: $snapshot" \
    || return 2
  cil_assert_snapshot_source_clean "$root" || return 2

  commit="$(git -C "$root" rev-parse HEAD)" || return 2
  tree="$(git -C "$root" rev-parse 'HEAD^{tree}')" || return 2
  mkdir -p "$snapshot" || return 2
  if ! git -C "$root" archive --format=tar --output="$archive" HEAD; then
    cil_snapshot_die "Cannot archive committed source tree $commit"
    return 2
  fi
  archive_sha="$(sha256sum "$archive" | awk '{print $1}')" || return 2
  if ! tar -xf "$archive" -C "$snapshot"; then
    cil_snapshot_die "Cannot extract committed source snapshot"
    return 2
  fi
  rm -f -- "$archive"

  # Large runtime artifacts are never copied into the immutable source tree.
  # Their own hashes/manifests remain the scientific integrity boundary.
  [[ -d "$root/outputs" ]] || mkdir -p "$root/outputs"
  ln -s "$root/outputs" "$snapshot/outputs" || return 2
  for name in train_extracted.parquet validation_extracted.parquet test_extracted.parquet; do
    [[ -s "$root/$name" ]] || {
      cil_snapshot_die "Missing runtime dataset required by snapshot: $root/$name"
      return 2
    }
    ln -s "$root/$name" "$snapshot/$name" || return 2
  done

  if ! (cd "$snapshot" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) \
    > "$run_home/source_snapshot_files_sha256.txt"; then
    cil_snapshot_die "Cannot build the source snapshot file manifest"
    return 2
  fi

  printf '%s\n' "$snapshot" > "$run_home/source_snapshot_root.txt"
  printf '%s\n' "$commit" > "$run_home/source_commit.txt"
  printf '%s\n' "$tree" > "$run_home/source_tree.txt"
  printf '%s\n' "$archive_sha" > "$run_home/source_archive_sha256.txt"
  printf '%s\n' \
    "{\"schema_version\":1,\"kind\":\"ddi_cil_immutable_source_snapshot\",\"git_commit\":\"$commit\",\"git_tree\":\"$tree\",\"archive_sha256\":\"$archive_sha\",\"file_manifest\":\"source_snapshot_files_sha256.txt\",\"snapshot_root\":\"$snapshot\",\"developer_checkout\":\"$root\"}" \
    > "$run_home/source_snapshot_manifest.json"
  echo "[OK] Immutable source snapshot: commit=$commit tree=$tree root=$snapshot" >&2
  printf '%s\n' "$snapshot"
}

cil_load_source_snapshot() {
  local run_home="$1" snapshot
  [[ -s "$run_home/source_snapshot_root.txt" ]] \
    || cil_snapshot_die "Run has no immutable source snapshot: $run_home" \
    || return 2
  snapshot="$(<"$run_home/source_snapshot_root.txt")"
  [[ -d "$snapshot/src" && -d "$snapshot/scripts" && -s "$run_home/source_commit.txt" \
    && -s "$run_home/source_snapshot_files_sha256.txt" ]] \
    || cil_snapshot_die "Run source snapshot is incomplete: $snapshot" \
    || return 2
  if ! (cd "$snapshot" && sha256sum -c "$run_home/source_snapshot_files_sha256.txt" >/dev/null); then
    cil_snapshot_die "Run source snapshot was modified or corrupted: $snapshot"
    return 2
  fi
  printf '%s\n' "$snapshot"
}

cil_snapshot_path() {
  local root="$1" snapshot="$2" path="$3"
  case "$path" in
    "$root") printf '%s\n' "$snapshot" ;;
    "$root"/*) printf '%s/%s\n' "$snapshot" "${path#"$root"/}" ;;
    *) printf '%s\n' "$path" ;;
  esac
}
