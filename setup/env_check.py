"""
Segment 00 - Environment Check

Usage (from repo root):
  python 00_setup/env_check.py
"""

import platform
import sys


def try_import(name: str) -> str:
    try:
        __import__(name)
        return "OK"
    except Exception as e:
        return f"FAIL ({type(e).__name__}: {e})"


def main() -> None:
    print("=== Environment Check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print()

    libs = ["numpy", "pandas", "sklearn", "matplotlib", "joblib"]
    print("=== Imports ===")
    for lib in libs:
        print(f"{lib:>12}: {try_import(lib)}")

    print("\nIf everything shows OK, you're good to go.")


if __name__ == "__main__":
    main()
