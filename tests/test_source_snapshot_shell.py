from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bash_executable() -> str | None:
    if os.name != "nt":
        return shutil.which("bash")
    for candidate in (
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


@unittest.skipUnless(_bash_executable(), "bash is required for launcher snapshot tests")
class SourceSnapshotShellTest(unittest.TestCase):
    def test_snapshot_isolated_integrity_checked_and_source_dirty_rejected(self) -> None:
        script = r'''
set -euo pipefail
source scripts/lib/source_snapshot.sh
t=$(mktemp -d)
trap 'rm -rf -- "$t"' EXIT
r="$t/repo"
mkdir -p "$r/src" "$r/scripts" "$r/configs" "$r/docs" "$r/outputs"
printf 'print(1)\n' > "$r/src/a.py"
printf '#!/bin/sh\n' > "$r/scripts/a.sh"
printf '{}\n' > "$r/configs/a.json"
printf 'initial\n' > "$r/docs/note.md"
printf '*.parquet\noutputs/\n' > "$r/.gitignore"
git -C "$r" init -q
git -C "$r" config user.email test@example.invalid
git -C "$r" config user.name test
git -C "$r" add .
git -C "$r" commit -qm initial
printf x > "$r/train_extracted.parquet"
printf x > "$r/validation_extracted.parquet"
printf x > "$r/test_extracted.parquet"

# Documentation edits do not alter training and need not block a run.
printf 'working note\n' > "$r/docs/note.md"
s=$(cil_create_source_snapshot "$r" "$t/run")
test "$(cil_load_source_snapshot "$t/run")" = "$s"

# Later checkout edits cannot change the already-created run snapshot.
printf 'print(2)\n' > "$r/src/a.py"
test "$(cat "$s/src/a.py")" = 'print(1)'
if cil_assert_snapshot_source_clean "$r" >/dev/null 2>&1; then
  exit 20
fi

# Resume refuses a snapshot whose files were changed in place.
printf 'tampered\n' > "$s/src/a.py"
if cil_load_source_snapshot "$t/run" >/dev/null 2>&1; then
  exit 21
fi
'''
        result = subprocess.run(
            [_bash_executable(), "-lc", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
