#!/usr/bin/env bash
set -euo pipefail

# Package the current validated FLAIR four-core branch.
#
# Usage:
#   cp FLAIR_4CORE_HANDOFF.md package_flair_4core.sh <FLAIR repo root>/
#   cd <FLAIR repo root>
#   bash package_flair_4core.sh
#
# Optional:
#   PACKAGE_NAME=my-name TAG_NAME=my-tag INCLUDE_BUILDS=0 bash package_flair_4core.sh

PACKAGE_NAME="${PACKAGE_NAME:-flair-4core-validated}"
TAG_NAME="${TAG_NAME:-flair-4core-validated}"
INCLUDE_BUILDS="${INCLUDE_BUILDS:-1}"
OUT_ROOT="${OUT_ROOT:-handoff_packages}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_ROOT}/${PACKAGE_NAME}-${STAMP}"
STAGE="${OUT_DIR}/package"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

git rev-parse --show-toplevel >/dev/null 2>&1 || fail "Run this script inside the FLAIR Git repository."
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BRANCH="$(git branch --show-current)"
[[ -n "$BRANCH" ]] || fail "Detached HEAD. Create or check out the final branch first."

[[ -f FLAIR_4CORE_HANDOFF.md ]] || fail "Copy FLAIR_4CORE_HANDOFF.md into the repository root first."

mkdir -p "$STAGE"

echo "Repository: $ROOT"
echo "Branch:     $BRANCH"
echo "Commit:     $(git rev-parse HEAD)"
echo "Output:     $OUT_DIR"

if [[ -n "$(git status --porcelain)" ]]; then
  echo
  echo "The worktree is not clean:"
  git status --short
  echo
  fail "Commit or intentionally remove all changes before packaging."
fi

if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
  TAG_COMMIT="$(git rev-list -n 1 "$TAG_NAME")"
  HEAD_COMMIT="$(git rev-parse HEAD)"
  [[ "$TAG_COMMIT" == "$HEAD_COMMIT" ]] || \
    fail "Tag '$TAG_NAME' already exists on another commit."
else
  git tag -a "$TAG_NAME" -m "Validated four-core FLAIR NPU pipeline"
fi

# Reproducible Git transport.
git bundle create \
  "$STAGE/${PACKAGE_NAME}.bundle" \
  "$BRANCH" \
  "$TAG_NAME"

git bundle verify "$STAGE/${PACKAGE_NAME}.bundle"

# Clean committed source snapshot.
git archive \
  --format=tar.gz \
  --prefix="${PACKAGE_NAME}/" \
  --output="$STAGE/${PACKAGE_NAME}-source.tar.gz" \
  "$TAG_NAME"

cp FLAIR_4CORE_HANDOFF.md "$STAGE/"

copy_if_present() {
  local src="$1"
  local dst_root="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$dst_root/$(dirname "$src")"
    cp -p "$src" "$dst_root/$src"
  fi
}

# Model checkpoint.
copy_if_present \
  "experiments/results/flair_h64_full.pt" \
  "$STAGE"

# Final logs and per-window scores.
copy_if_present \
  "npu/results/benchmarks_4core/full_codegen_only.txt" \
  "$STAGE"
copy_if_present \
  "npu/results/benchmarks_4core/full_codegen_only_scores.csv" \
  "$STAGE"
copy_if_present \
  "npu/results/benchmarks_4core/full_4core_unfused_b8_summary.txt" \
  "$STAGE"

# Convenience artifacts. These may need rebuilding on another environment.
if [[ "$INCLUDE_BUILDS" == "1" ]]; then
  copy_if_present "npu/build/gru_4core.xclbin" "$STAGE"
  copy_if_present "npu/build/gru_4core_insts.bin" "$STAGE"
  copy_if_present "npu/build/decoder_4core.xclbin" "$STAGE"
  copy_if_present "npu/build/decoder_4core_insts.bin" "$STAGE"
  copy_if_present "npu/batch_infer.exe" "$STAGE"
fi

{
  echo "FLAIR four-core validated handoff"
  echo
  echo "Created: $(date --iso-8601=seconds)"
  echo "Repository: $ROOT"
  echo "Branch: $BRANCH"
  echo "Tag: $TAG_NAME"
  echo "Commit: $(git rev-parse HEAD)"
  echo
  echo "Git status:"
  git status --short
  echo
  echo "Recent commits:"
  git log --oneline --decorate -10
  echo
  echo "Included files:"
  find "$STAGE" -type f -printf '%P\n' | sort
} > "$STAGE/MANIFEST.txt"

(
  cd "$STAGE"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
)

FINAL_ARCHIVE="${OUT_DIR}/${PACKAGE_NAME}-${STAMP}.tar.gz"
tar -C "$STAGE" -czf "$FINAL_ARCHIVE" .

echo
echo "Created handoff:"
echo "  $FINAL_ARCHIVE"
echo
echo "Bundle:"
echo "  $STAGE/${PACKAGE_NAME}.bundle"
echo
echo "Verify:"
echo "  tar -tzf '$FINAL_ARCHIVE' | head"
echo "  (cd '$STAGE' && sha256sum -c SHA256SUMS)"
