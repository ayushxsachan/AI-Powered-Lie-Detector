from __future__ import annotations

import sys


def main() -> int:
    try:
        from ui_dashboard import launch
    except ImportError as exc:
        print("Missing dependency:", exc)
        print("Install dependencies with: pip install -r requirements.txt")
        return 1
    return int(launch())


if __name__ == "__main__":
    raise SystemExit(main())
