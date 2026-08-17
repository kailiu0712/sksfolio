"""Repository-level entry point for the real-data six-panel scaling benchmark."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sksfolio.benchmarks.run_four_scaling_real import main


if __name__ == "__main__":
    main()
