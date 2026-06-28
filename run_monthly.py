"""
Monthly backtest runner — runs one month at a time, archives each checkpoint,
and waits between months so the PC can cool down.

Usage:
    python run_monthly.py
    python run_monthly.py --mode momentum
    python run_monthly.py --crypto ETH --cooling 120
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# ── Configuration ─────────────────────────────────────────────────────────────

MONTHS = [
    ("2026-02-01", "2026-03-01"),
    ("2026-03-01", "2026-04-01"),
    ("2026-04-01", "2026-05-01"),
    ("2026-05-01", "2026-06-01"),
    ("2026-06-01", "2026-06-20"),
]

DEFAULT_CRYPTO  = "BTC"
DEFAULT_COOLING = 90   # seconds between months (PC cooldown)
DEFAULT_MODE    = "ai" # "ai" or "momentum"

# ── Helpers ───────────────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
PYTHON     = sys.executable


def main_ckpt(crypto: str, mode: str) -> str:
    suffix = "_momentum" if mode == "momentum" else ""
    return os.path.join(CKPT_DIR, f"backtest_{crypto}{suffix}.json")


def monthly_ckpt(crypto: str, label: str, mode: str) -> str:
    suffix = "_momentum" if mode == "momentum" else ""
    return os.path.join(CKPT_DIR, f"backtest_{crypto}{suffix}_{label}.json")


def read_equity(path: str, default: float) -> float:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data["portfolio"].get("portfolio_value", default))
    except Exception:
        return default


def archive(crypto: str, label: str, mode: str):
    src = main_ckpt(crypto, mode)
    if not os.path.exists(src):
        return
    dst = monthly_ckpt(crypto, label, mode)
    shutil.copy(src, dst)
    tag = "_momentum" if mode == "momentum" else ""
    print(f"[Wrapper] Saved  -> checkpoints/backtest_{crypto}{tag}_{label}.json")


def cooling_wait(seconds: int):
    if seconds <= 0:
        return
    print(f"\n[Cooling] {seconds}s break -- PC cooldown", end="", flush=True)
    for _ in range(seconds, 0, -5):
        time.sleep(5)
        print(".", end="", flush=True)
    print(" done.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(crypto: str, cooling: int, mode: str, risk_pct: float):
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"[Wrapper] Mode: {mode.upper()}")

    # Archive any existing January checkpoint before overwriting it
    ckpt = main_ckpt(crypto, mode)
    jan_archive = monthly_ckpt(crypto, "2026-01", mode)
    if os.path.exists(ckpt) and not os.path.exists(jan_archive):
        shutil.copy(ckpt, jan_archive)
        tag = "_momentum" if mode == "momentum" else ""
        print(f"[Wrapper] Archived existing checkpoint -> backtest_{crypto}{tag}_2026-01.json")

    # Starting capital = end-of-January equity (or 100 000 if no checkpoint)
    capital = read_equity(ckpt, 100_000.0)
    print(f"[Wrapper] Starting capital carried forward: ${capital:,.2f}\n")

    results = []

    for i, (start, end) in enumerate(MONTHS):
        label = start[:7]

        # Skip months that are already archived
        if os.path.exists(monthly_ckpt(crypto, label, mode)):
            saved_eq = read_equity(monthly_ckpt(crypto, label, mode), capital)
            print(f"[Wrapper] {label} already done (equity {saved_eq:,.2f}) -- skipping.")
            capital = saved_eq
            results.append((label, capital))
            continue

        print(f"\n{'='*62}")
        print(f"  Month: {label}  ({start} -> {end})  [{mode}]")
        print(f"  Capital in: ${capital:,.2f}")
        print(f"{'='*62}\n")

        cmd = [
            PYTHON, "src/backtester.py",
            "--crypto",          crypto,
            "--start-date",      start,
            "--end-date",        end,
            "--initial-capital", f"{capital:.2f}",
            "--reset",
            "--mode",            mode,
            "--risk-pct",        f"{risk_pct}",
        ]

        ret = subprocess.run(cmd, cwd=ROOT)

        if ret.returncode != 0:
            print(f"[Wrapper] WARNING backtester exited with code {ret.returncode} for {label}.")

        archive(crypto, label, mode)
        capital = read_equity(ckpt, capital)
        results.append((label, capital))
        print(f"\n[Wrapper] {label} done -- equity: ${capital:,.2f}")

        if i < len(MONTHS) - 1:
            cooling_wait(cooling)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  MULTI-MONTH SUMMARY  --  {crypto}  [{mode}]")
    print(f"{'='*62}")
    for label, eq in results:
        print(f"  {label}   equity: ${eq:,.2f}")
    if results:
        first_eq = read_equity(monthly_ckpt(crypto, "2026-01", mode), 100_000.0)
        final_eq = results[-1][1]
        total_ret = (final_eq - first_eq) / first_eq * 100
        print(f"  ---------------------------------")
        print(f"  Jan start : ${first_eq:,.2f}")
        print(f"  Jun end   : ${final_eq:,.2f}")
        print(f"  Total ret : {total_ret:+.2f}%")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtester month by month")
    parser.add_argument("--crypto",  default=DEFAULT_CRYPTO, help="Crypto symbol (default: BTC)")
    parser.add_argument("--cooling", type=int, default=DEFAULT_COOLING,
                        help=f"Seconds to wait between months (default: {DEFAULT_COOLING})")
    parser.add_argument("--mode", choices=["ai", "momentum"], default=DEFAULT_MODE,
                        help="'ai' = DeepSeek agent  |  'momentum' = 3-day price rule baseline")
    parser.add_argument("--risk-pct", type=float, default=0.05,
                        help="Fraction of equity risked per trade (default: 0.05 = 5%%)")
    args = parser.parse_args()
    run(args.crypto, args.cooling, args.mode, args.risk_pct)
