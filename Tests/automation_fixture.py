"""Small child processes used to exercise the real automation launchers."""

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["bridge", "child"])
    args = parser.parse_args()

    if args.role == "bridge":
        print("AUTOMATION_FIXTURE_READY", flush=True)
        while True:
            time.sleep(0.1)

    print("AUTOMATION_FIXTURE_CHILD_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
