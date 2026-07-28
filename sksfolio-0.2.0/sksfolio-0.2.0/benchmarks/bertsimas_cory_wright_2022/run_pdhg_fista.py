"""Repository entry point for the matched PDHG--FISTA benchmark."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sksfolio.benchmarks.run_pdhg_fista import main


if __name__ == "__main__":
    main()
