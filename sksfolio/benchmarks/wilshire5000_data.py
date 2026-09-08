"""Real-return dataset builder for :mod:`run_four_scaling_real`.

Builds three *nested* stock pools -- S&P 500 subset Russell-1000-proxy
subset Wilshire-5000-proxy -- and caches daily returns plus pool membership
to disk as a small dataset that :mod:`run_four_scaling_real` loads and
slices per instance.

Source data and why the two larger tiers are proxies, not the licensed
indices:

- S&P 500: scraped from the Wikipedia constituent table. This is today's
  membership, not any particular historical snapshot, so there is
  survivorship bias versus a point-in-time index -- documented here rather
  than hidden.
- Russell 1000 / Wilshire 5000: neither index publishes a free constituent
  list reachable from a plain HTTP client (the iShares holdings CSV
  endpoints redirect through a JS-rendered page here). Instead this module
  ranks NASDAQ/NYSE/AMEX common stocks -- sourced from the NASDAQ Trader
  symbol directories -- by median daily dollar volume (Close * Volume) over
  the sample window, a standard liquidity/size proxy used when a market-cap
  panel is not available, and takes the top slice of that ranking as each
  tier. Treat "russell1000" and "wilshire5000" here as size-ranked proxies,
  not literal index membership.

Nesting (sp500 subset russell1000 subset wilshire5000) is enforced by
construction via set union, not assumed from the sources.

This module has real, optional dependencies the core ``sksfolio`` package
does not need: ``pip install sksfolio[real-data]``.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "wilshire5000_data requires pandas and pyarrow even to load an "
        "existing dataset"
    ) from error
try:
    import requests
except ImportError:  # pragma: no cover
    requests = None
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):  # type: ignore[misc]
        return iterable


SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
USER_AGENT = "Mozilla/5.0 (compatible; sksfolio-benchmark/1.0)"

def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _results_root() -> Path:
    return _workspace_root() / "proximal" / "code0821brian" / "results"


DEFAULT_OUTPUT_DIR = _results_root() / "data" / "wilshire5000"
DEFAULT_MIN_COVERAGE = 0.98
DEFAULT_RUSSELL_SIZE = 1000
DEFAULT_BATCH_SIZE = 150
DEFAULT_PAUSE_SECONDS = 1.0
TIER_NAMES = ("sp500", "russell1000", "wilshire5000")

_EXCLUDED_NAME_KEYWORDS = (
    "warrant",
    "right",
    "unit",
    "preferred",
    "depositary",
    "notes",
    "debenture",
    "when issued",
    " wt",
    " rts",
)
_EXCLUDED_SYMBOL_CHARS = ("$", "+", "=", "#", "~", "*")


@dataclass(frozen=True)
class RealDataset:
    """Daily simple returns plus the three nested ticker pools."""

    returns: "pd.DataFrame"
    tiers: Dict[str, Tuple[str, ...]]
    metadata: Dict[str, Any]


def _require_builder_dependencies() -> None:
    missing: List[str] = []
    if requests is None:
        missing.append("requests")
    if yf is None:
        missing.append("yfinance")
    if missing:
        missing_text = ", ".join(missing)
        raise ImportError(
            "wilshire5000_data requires optional real-data download "
            f"dependencies ({missing_text}) to build a dataset; install "
            "with `pip install sksfolio[real-data]`"
        )


def _http_get_text(url: str, timeout: float = 30.0) -> str:
    _require_builder_dependencies()
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.text


def normalize_yahoo_ticker(symbol: str) -> Optional[str]:
    """Map a listing symbol to Yahoo Finance's dash share-class format.

    Returns ``None`` for symbols carrying a legacy Nasdaq suffix character
    that denotes a non-common instrument (warrant, unit, preferred, ...).
    """
    symbol = symbol.strip().upper()
    if not symbol or any(char in symbol for char in _EXCLUDED_SYMBOL_CHARS):
        return None
    return symbol.replace(".", "-")


def fetch_sp500_constituents(timeout: float = 30.0) -> Tuple[str, ...]:
    """Scrape the current S&P 500 constituent table from Wikipedia."""
    html = _http_get_text(SP500_WIKIPEDIA_URL, timeout=timeout)
    table = pd.read_html(io.StringIO(html))[0]
    if "Symbol" not in table.columns:
        raise ValueError("unexpected S&P 500 table layout: no Symbol column")
    tickers = (normalize_yahoo_ticker(str(symbol)) for symbol in table["Symbol"])
    return tuple(sorted({ticker for ticker in tickers if ticker}))


def _parse_symbol_directory(
    text: str,
    symbol_column: str,
    name_column: str,
) -> Tuple[str, ...]:
    lines = [line for line in text.splitlines() if line and "|" in line]
    if not lines:
        return ()
    header = lines[0].split("|")
    rows = [line.split("|") for line in lines[1:] if len(line.split("|")) == len(header)]
    frame = pd.DataFrame(rows, columns=header)
    keep = (frame.get("ETF", "N") != "Y") & (frame.get("Test Issue", "N") != "Y")
    frame = frame[keep]
    lowered_names = frame[name_column].str.lower()
    excluded = lowered_names.apply(
        lambda name: any(keyword in name for keyword in _EXCLUDED_NAME_KEYWORDS)
    )
    frame = frame[~excluded]
    tickers = (normalize_yahoo_ticker(str(symbol)) for symbol in frame[symbol_column])
    return tuple(sorted({ticker for ticker in tickers if ticker}))


def fetch_candidate_common_stocks(timeout: float = 30.0) -> Tuple[str, ...]:
    """Return a heuristically filtered NASDAQ/NYSE/AMEX common-stock universe.

    This is a filter over the NASDAQ Trader symbol directories, not an
    authoritative common-stock classification -- a handful of non-common
    instruments may slip through. The downstream coverage and liquidity
    filters in :func:`build_dataset` remove nearly all of what leaks in.
    """
    nasdaq = _parse_symbol_directory(
        _http_get_text(NASDAQ_LISTED_URL, timeout=timeout),
        symbol_column="Symbol",
        name_column="Security Name",
    )
    other = _parse_symbol_directory(
        _http_get_text(OTHER_LISTED_URL, timeout=timeout),
        symbol_column="ACT Symbol",
        name_column="Security Name",
    )
    return tuple(sorted(set(nasdaq) | set(other)))


def download_price_and_volume(
    tickers: Sequence[str],
    start: date,
    end: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    progress: bool = True,
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Download adjusted daily close and volume for ``tickers`` in batches."""
    _require_builder_dependencies()
    prices: Dict[str, "pd.Series"] = {}
    volumes: Dict[str, "pd.Series"] = {}
    batches = [list(tickers[i : i + batch_size]) for i in range(0, len(tickers), batch_size)]
    iterator = tqdm(batches, desc="price download", unit="batch") if progress else batches
    for batch in iterator:
        try:
            frame = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as error:  # network/format hiccups must not kill a long run
            print(f"    batch of {len(batch)} tickers failed: {error}")
            time.sleep(pause_seconds)
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in frame.columns.get_level_values(0):
                    continue
                sub = frame[ticker]
                if "Close" not in sub.columns or sub["Close"].dropna().empty:
                    continue
                prices[ticker] = sub["Close"]
                volumes[ticker] = sub["Volume"]
        elif len(batch) == 1 and "Close" in frame.columns and not frame["Close"].dropna().empty:
            prices[batch[0]] = frame["Close"]
            volumes[batch[0]] = frame["Volume"]
        time.sleep(pause_seconds)
    price_frame = pd.DataFrame(prices).sort_index()
    volume_frame = pd.DataFrame(volumes).sort_index()
    return price_frame, volume_frame


def filter_by_coverage(
    prices: "pd.DataFrame",
    volumes: "pd.DataFrame",
    min_coverage: float,
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Keep tickers with enough non-missing history, then drop dead tickers.

    A ticker passes if at least ``min_coverage`` of the window is observed
    (recent IPOs and delistings fail this and are dropped); the remaining
    small gaps are forward/back-filled. Tickers with zero return variance
    (halted, or a data error) are dropped afterward.
    """
    coverage = prices.notna().mean(axis=0)
    keep = coverage[coverage >= min_coverage].index
    prices = prices[keep].ffill().bfill()
    volumes = volumes[keep].ffill().bfill()
    daily_returns = prices.pct_change().iloc[1:]
    non_degenerate = daily_returns.std(axis=0).fillna(0.0) > 0.0
    columns = non_degenerate[non_degenerate].index
    return prices[columns], volumes[columns]


def rank_by_dollar_volume(prices: "pd.DataFrame", volumes: "pd.DataFrame") -> List[str]:
    """Rank tickers by median daily dollar volume, largest first."""
    dollar_volume = (prices * volumes).median(axis=0)
    return list(dollar_volume.sort_values(ascending=False).index)


def build_universe_tiers(
    sp500: Sequence[str],
    ranked_pool: Sequence[str],
    russell_size: int = DEFAULT_RUSSELL_SIZE,
) -> Dict[str, Tuple[str, ...]]:
    """Build the nested sp500 subset russell1000 subset wilshire5000 pools.

    ``ranked_pool`` must already be restricted to tickers with usable data
    (see :func:`filter_by_coverage`) and ordered largest-size-proxy first.
    """
    sp500_tier = tuple(sorted(set(sp500) & set(ranked_pool)))
    top_ranked = ranked_pool[: max(russell_size, 0)]
    russell_tier = tuple(sorted(set(sp500_tier) | set(top_ranked)))
    wilshire_tier = tuple(sorted(set(russell_tier) | set(ranked_pool)))
    return {"sp500": sp500_tier, "russell1000": russell_tier, "wilshire5000": wilshire_tier}


def build_dataset(
    *,
    start: date,
    end: date,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    russell_size: int = DEFAULT_RUSSELL_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_candidates: Optional[int] = None,
    progress: bool = True,
) -> RealDataset:
    """Fetch, filter, and tier a real-return dataset from public sources."""
    _require_builder_dependencies()
    sp500 = fetch_sp500_constituents()
    candidates = sorted(set(fetch_candidate_common_stocks()) | set(sp500))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
        sp500 = tuple(ticker for ticker in sp500 if ticker in set(candidates))

    prices, volumes = download_price_and_volume(
        candidates,
        start,
        end,
        batch_size=batch_size,
        pause_seconds=pause_seconds,
        progress=progress,
    )
    prices, volumes = filter_by_coverage(prices, volumes, min_coverage)
    ranked_pool = rank_by_dollar_volume(prices, volumes)
    tiers = build_universe_tiers(sp500, ranked_pool, russell_size=russell_size)
    returns = prices.pct_change().iloc[1:]

    metadata = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "min_coverage": min_coverage,
        "russell_size": russell_size,
        "candidate_count": len(candidates),
        "downloaded_count": int(prices.shape[1]),
        "tier_sizes": {name: len(members) for name, members in tiers.items()},
        "sources": {
            "sp500": SP500_WIKIPEDIA_URL,
            "candidate_common_stock": [NASDAQ_LISTED_URL, OTHER_LISTED_URL],
            "prices": "Yahoo Finance via yfinance",
            "size_proxy": "median daily dollar volume (Close * Volume) over the sample window",
        },
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }
    return RealDataset(returns=returns, tiers=tiers, metadata=metadata)


def save_dataset(dataset: RealDataset, output_dir: Path, overwrite: bool = False) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / "returns.parquet"
    if returns_path.exists() and not overwrite:
        raise FileExistsError(f"{returns_path} already exists; pass overwrite=True to replace it")
    dataset.returns.to_parquet(returns_path)
    membership = {name: list(members) for name, members in dataset.tiers.items()}
    (output_dir / "universe_membership.json").write_text(json.dumps(membership, indent=2))
    (output_dir / "metadata.json").write_text(json.dumps(dataset.metadata, indent=2))
    return output_dir


def load_dataset(dataset_dir: Path) -> RealDataset:
    dataset_dir = Path(dataset_dir)
    returns = pd.read_parquet(dataset_dir / "returns.parquet")
    membership = json.loads((dataset_dir / "universe_membership.json").read_text())
    tiers = {name: tuple(members) for name, members in membership.items()}
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    return RealDataset(returns=returns, tiers=tiers, metadata=metadata)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Build the S&P 500 / Russell-1000-proxy / Wilshire-5000-proxy "
            "real-return dataset used by run_four_scaling_real.py"
        )
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="trailing window length in years, ending at --end-date",
    )
    result.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD); defaults to today",
    )
    result.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    result.add_argument("--russell-size", type=int, default=DEFAULT_RUSSELL_SIZE)
    result.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    result.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS)
    result.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="cap the candidate universe before downloading; for quick smoke tests",
    )
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--quiet", action="store_true")
    return result


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    end = date.fromisoformat(args.end_date) if args.end_date else date.today()
    start = end - timedelta(days=int(args.years * 365.25))

    dataset = build_dataset(
        start=start,
        end=end,
        min_coverage=args.min_coverage,
        russell_size=args.russell_size,
        batch_size=args.batch_size,
        pause_seconds=args.pause_seconds,
        max_candidates=args.max_candidates,
        progress=not args.quiet,
    )
    output_dir = save_dataset(dataset, args.output_dir, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "window": [start.isoformat(), end.isoformat()],
                "tier_sizes": dataset.metadata["tier_sizes"],
                "downloaded_count": dataset.metadata["downloaded_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
