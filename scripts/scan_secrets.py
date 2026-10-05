"""Fail if a tracked (or staged) file contains something shaped like an API key.

Usage: python scripts/scan_secrets.py [files...]   (default: all git-tracked files)
"""
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "Groq key": r"gsk_[A-Za-z0-9]{20,}",
    "Google API key": r"AIza[0-9A-Za-z_\-]{30,}",
    "Google AI Studio key": r"\bAQ\.[A-Za-z0-9_\-]{30,}",
    "OpenAI/Anthropic-style key": r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}",
    "GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "Private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}
SKIP_SUFFIX = {".png", ".jpg", ".gif", ".pdf", ".spice", ".ico"}


def scan_text(text: str) -> list[str]:
    return [name for name, pat in PATTERNS.items() if re.search(pat, text)]


def main(files: list[str]) -> int:
    if not files:
        files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    bad = 0
    for f in files:
        p = Path(f)
        if p.suffix in SKIP_SUFFIX or not p.is_file():
            continue
        try:
            hits = scan_text(p.read_text(errors="ignore"))
        except OSError:
            continue
        for h in hits:
            print(f"{f}: possible {h}")
            bad += 1
    if bad:
        print(f"\n{bad} potential secret(s) found. Remove them and rotate the key.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
