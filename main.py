"""
main.py
───────
Student Performance Prediction — Pipeline Runner

Usage:
    python main.py --phase generate
    python main.py --phase ingest
    python main.py --phase all
"""

import argparse
import subprocess
import sys


def run(script: str):
    print(f"\n{'─'*60}")
    print(f"▶  Running: {script}")
    print(f"{'─'*60}")
    subprocess.run([sys.executable, script], check=True)


def main():
    parser = argparse.ArgumentParser(description="Student Performance Prediction Pipeline")
    parser.add_argument("--phase", choices=["generate", "ingest", "all"],
                        default="generate")
    args = parser.parse_args()

    print("🎓 Student Performance Prediction System")
    print("=" * 60)

    if args.phase in ("generate", "all"):
        run("generate_data.py")

    if args.phase in ("ingest", "all"):
        run("notebooks/01_ingest.py")

    if args.phase == "all":
        print("\n✅ Phase 1 complete!")
        print("   Next: python notebooks/02_eda.py")


if __name__ == "__main__":
    main()