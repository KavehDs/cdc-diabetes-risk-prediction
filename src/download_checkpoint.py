"""Download the TabICLv2 classifier checkpoint with a plain streaming
HTTP download (no torch, no hf-xet) so it uses almost no RAM.
Resumes a partial download with HTTP Range. Saves to
~/.cache/tabicl/tabicl-classifier-v2-20260212.ckpt (outside the repo —
weights are never bundled in the project).
"""

import os
import sys
import urllib.request

URL = (
    "https://huggingface.co/jingang/TabICL/resolve/main/"
    "tabicl-classifier-v2-20260212.ckpt"
)
DEST = os.path.expanduser(
    "~/.cache/tabicl/tabicl-classifier-v2-20260212.ckpt"
)
CHUNK = 8 * 1024 * 1024


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    have = os.path.getsize(DEST) if os.path.exists(DEST) else 0

    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers["Content-Length"])
    print(f"Remote size: {total:,} bytes ({total / 1e6:.1f} MB)")
    if have >= total:
        print(f"Already complete: {DEST} ({have:,} bytes)")
        return
    print(f"Resuming from {have:,} bytes...")

    req = urllib.request.Request(URL)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as r:
        mode = "ab" if have and r.status == 206 else "wb"
        if mode == "wb":
            have = 0
        downloaded = have
        with open(DEST, mode) as f:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                print(f"\r{downloaded:,}/{total:,} bytes "
                      f"({100 * downloaded / total:.1f}%)",
                      end="", flush=True)
    print(f"\nDone: {DEST} ({os.path.getsize(DEST):,} bytes)")
    if os.path.getsize(DEST) != total:
        print("SIZE MISMATCH — download incomplete", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
