from datetime import datetime, timedelta
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from main import run_hedge_fund
from tools.api import get_price_API_HYPERLIQUID, get_price_API_BINANCE

# Fee model — sources logged at run start so misleading assumptions are visible.
# SOURCE: HyperLiquid published taker fee (real as of 2024-Q4, hardcoded — not fetched live).
# Funding rate is NOT modelled: positions close/re-open every daily iteration,
# so intraday funding accumulation is negligible. Add FUNDING_RATE_8H if you
# switch to multi-day holds.
TAKER_FEE_RATE = 0.00055        # 0.055% per side (Bybit perp taker)
FUNDING_RATE_8H = 0.0001        # 0.01% per 8-hour period (assumed positive: longs pay)
FUNDING_PERIODS_PER_DAY = 3     # 00:00 / 08:00 / 16:00 UTC

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# ── Momentum baseline ──────────────────────────────────────────────────────────
MOMENTUM_QTY = 4_500  # fixed notional USD per trade


def _momentum_signal(df: pd.DataFrame, lookback_days: int = 3) -> str:
    """Return long/short/hold based on N consecutive daily closes (hourly df)."""
    try:
        closes = df["close"].dropna()
        if len(closes) < (lookback_days + 1) * 24:
            return "hold"
        # Sample one close per day (last hourly close, stepping back 24 h each time)
        daily = [float(closes.iloc[-(i * 24 + 1)]) for i in range(lookback_days + 1)]
        # daily[0]=today, daily[1]=yesterday, daily[2]=2d ago, ...
        # Rising: today > yesterday > ...
        ups = sum(daily[i] > daily[i + 1] for i in range(lookback_days))
        if ups == lookback_days:
            return "long"
        elif ups == 0:
            return "short"
        return "hold"
    except Exception:
        return "hold"


def momentum_agent(crypto: str, start_date: str, end_date: str, portfolio: dict) -> str:
    """3-day momentum baseline — no LLM, pure price rule for comparison."""
    df = get_price_API_HYPERLIQUID(crypto, start_date, end_date)
    if df is None or df.empty:
        df = get_price_API_BINANCE(crypto, start_date, end_date)
    if df is None or (hasattr(df, "empty") and df.empty):
        return json.dumps({
            "action": "hold", "quantity": 0,
            "stop_loss_pct": 0, "take_profit_pct": 0,
            "confidence": 0.5, "reasoning": "no data",
        })
    action = _momentum_signal(df)
    qty = MOMENTUM_QTY if action != "hold" else 0
    return json.dumps({
        "action": action, "quantity": qty,
        "stop_loss_pct": 0.03, "take_profit_pct": 0.05,
        "confidence": 0.7, "reasoning": f"3-day momentum -> {action}",
    })


class Backtester:
    def __init__(self, agent, crypto, start_date, end_date, initial_capital,
                 reset: bool = False, risk_pct: float = 0.05,
                 checkpoint_enabled: bool = True, tag: str = "",
                 inter_day_sleep: int = 2, interval: str = "1d",
                 leverage: int = 1):
        self.agent = agent
        self.crypto = crypto
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.checkpoint_enabled = checkpoint_enabled
        self.inter_day_sleep = inter_day_sleep
        self.interval = interval
        self.leverage = leverage
        self.state_file = os.path.join(CHECKPOINT_DIR, f"backtest_{crypto}{tag}.json")

        self.portfolio = {
            "cash": initial_capital,
            "collateral_long": 0.0,
            "collateral_short": 0.0,
            "price_collateral": 0.0,
            "leverage": 10,
            "risk": 0.05,
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "total_fees_paid": 0.0,
        }
        self.portfolio_values = []
        self._last_completed_date = None
        self.trades = []
        self.benchmark_start_price = None
        self.benchmark_end_price = None
        self.override_count = 0   # days where trend-override fired
        self.agent_days = 0       # days where agent actually ran

        if checkpoint_enabled:
            if reset:
                self._delete_state()
            else:
                self._load_state()
        else:
            print("[Checkpoint] Disabled for this run.")

    # ------------------------------------------------------------------ #
    #  Checkpoint helpers                                                  #
    # ------------------------------------------------------------------ #

    def _load_state(self):
        if not os.path.exists(self.state_file):
            print("[Checkpoint] No existing checkpoint found, starting fresh.")
            return
        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.initial_capital = state["initial_capital"]
        self.portfolio = state["portfolio"]
        self.portfolio_values = [
            {"Date": pd.Timestamp(r["Date"]), "Portfolio Value": r["Portfolio Value"]}
            for r in state["portfolio_values"]
        ]
        self.trades = state.get("trades", [])
        self.benchmark_start_price = state.get("benchmark_start_price")
        self.benchmark_end_price = state.get("benchmark_end_price")
        self._last_completed_date = state.get("last_date")
        print(f"[Checkpoint] Loaded: {self.state_file}")
        print(f"[Checkpoint] Last completed day   : {self._last_completed_date or '?'}")
        print(f"[Checkpoint] Portfolio value      : {self.portfolio.get('portfolio_value', self.portfolio['cash']):,.2f}")

    def _save_state(self, last_date: str = None):
        if not self.checkpoint_enabled:
            return
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        state = {
            "crypto": self.crypto,
            "initial_capital": self.initial_capital,
            "last_date": last_date or self.end_date,
            "portfolio": self.portfolio,
            "portfolio_values": [
                {"Date": str(r["Date"])[:10], "Portfolio Value": r["Portfolio Value"]}
                for r in self.portfolio_values
            ],
            "trades": self.trades,
            "benchmark_start_price": self.benchmark_start_price,
            "benchmark_end_price": self.benchmark_end_price,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _delete_state(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            print(f"[Checkpoint] Reset — deleted {self.state_file}")
        else:
            print("[Checkpoint] Nothing to reset, starting fresh.")

    # ------------------------------------------------------------------ #
    #  Feature 1: Equity & position cap                                   #
    # ------------------------------------------------------------------ #

    def _get_equity(self, price: float) -> float:
        """Equity = cash + margin_in_position + unrealized_pnl (leverage-aware)."""
        equity = self.portfolio["cash"]
        entry = self.portfolio.get("price_collateral") or 0.0
        if self.portfolio["collateral_long"] > 0 and entry > 0:
            notional_at_entry = self.portfolio["collateral_long"] * entry
            margin = notional_at_entry / self.leverage
            pnl = self.portfolio["collateral_long"] * (price - entry)
            equity += margin + pnl
        elif self.portfolio["collateral_short"] > 0 and entry > 0:
            notional_at_entry = self.portfolio["collateral_short"] * entry
            margin = notional_at_entry / self.leverage
            pnl = self.portfolio["collateral_short"] * (entry - price)
            equity += margin + pnl
        return max(equity, 0.0)

    def _cap_quantity(self, _raw: float, price: float) -> float:
        """Return notional = risk_pct * equity * leverage (LLM qty ignored)."""
        equity = self._get_equity(price)
        margin = equity * self.risk_pct
        available_margin = min(margin, self.portfolio["cash"])
        if available_margin <= 0:
            return 0.0
        return available_margin * self.leverage  # notional position size

    # ------------------------------------------------------------------ #
    #  Feature 2: Funding rate                                            #
    # ------------------------------------------------------------------ #

    def _apply_daily_funding(self, price: float):
        """Longs pay, shorts receive. Rate assumed positive (typical bull market).
        Logged at run start so the assumption is always visible."""
        rate = FUNDING_RATE_8H * FUNDING_PERIODS_PER_DAY
        if self.portfolio["collateral_long"] > 0:
            cost = self.portfolio["collateral_long"] * price * rate
            self.portfolio["cash"] -= cost
            self.portfolio["total_fees_paid"] += cost
        elif self.portfolio["collateral_short"] > 0:
            gain = self.portfolio["collateral_short"] * price * rate
            self.portfolio["cash"] += gain

    # ------------------------------------------------------------------ #
    #  Feature 6: SL / TP intraday check                                  #
    # ------------------------------------------------------------------ #

    def _check_sl_tp(self, high: float, low: float) -> bool:
        """Close position early if today's candle touches SL, TP, or liquidation level."""
        sl = self.portfolio.get("stop_loss_pct")
        tp = self.portfolio.get("take_profit_pct")
        entry = self.portfolio.get("price_collateral", 0)
        if not entry:
            return False

        # Liquidation check (leverage > 1): 90% margin loss = forced close
        if self.leverage > 1:
            liq_pct = 0.9 / self.leverage
            if self.portfolio["collateral_long"] > 0:
                liq_price = entry * (1 - liq_pct)
                if low <= liq_price:
                    print(f"  [LIQUIDATED] Long liquidated ~{liq_price:.2f} ({liq_pct*100:.0f}% margin loss)")
                    self.sell_collateral(liq_price)
                    return True
            elif self.portfolio["collateral_short"] > 0:
                liq_price = entry * (1 + liq_pct)
                if high >= liq_price:
                    print(f"  [LIQUIDATED] Short liquidated ~{liq_price:.2f} ({liq_pct*100:.0f}% margin loss)")
                    self.sell_collateral(liq_price)
                    return True

        if not (sl and tp):
            return False

        if self.portfolio["collateral_long"] > 0:
            if low <= entry * (1 - sl):
                exit_px = entry * (1 - sl)
                print(f"  [SL] Long SL hit ~{exit_px:.2f}  (day low={low:.2f})")
                self.sell_collateral(exit_px)
                return True
            if high >= entry * (1 + tp):
                exit_px = entry * (1 + tp)
                print(f"  [TP] Long TP hit ~{exit_px:.2f}  (day high={high:.2f})")
                self.sell_collateral(exit_px)
                return True

        elif self.portfolio["collateral_short"] > 0:
            if high >= entry * (1 + sl):
                exit_px = entry * (1 + sl)
                print(f"  [SL] Short SL hit ~{exit_px:.2f}  (day high={high:.2f})")
                self.sell_collateral(exit_px)
                return True
            if low <= entry * (1 - tp):
                exit_px = entry * (1 - tp)
                print(f"  [TP] Short TP hit ~{exit_px:.2f}  (day low={low:.2f})")
                self.sell_collateral(exit_px)
                return True

        return False

    # ------------------------------------------------------------------ #
    #  Trading logic                                                       #
    # ------------------------------------------------------------------ #

    def parse_action(self, agent_output):
        """Return (action, quantity, stop_loss_pct, take_profit_pct)."""
        import re

        def _pct(v):
            if v is None:
                return None
            if isinstance(v, str):
                v = v.strip().rstrip("%")
            try:
                f = float(v)
                return f / 100 if f > 1 else f
            except (TypeError, ValueError):
                return None

        def _from_dict(d):
            action = str(d.get("action", "hold")).lower()
            quantity = float(d.get("quantity", 0))
            sl = _pct(d.get("stop_loss_pct") or d.get("stop_loss"))
            tp = _pct(d.get("take_profit_pct") or d.get("take_profit"))
            return action, quantity, sl, tp

        # 1. Strict JSON
        try:
            return _from_dict(json.loads(agent_output))
        except Exception:
            pass

        # 2. JSON block inside markdown
        json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", agent_output, re.DOTALL)
        if json_block:
            try:
                return _from_dict(json.loads(json_block.group(1)))
            except Exception:
                pass

        # 3. Regex fallback
        action_match = re.search(r"\b(long|short)\b", agent_output, re.IGNORECASE)
        action = action_match.group(1).lower() if action_match else "hold"
        qty_match = re.search(r"(?i)(?:quantity)[^\d]*([0-9]+(?:[.,][0-9]+)?)", agent_output)
        quantity = float(qty_match.group(1).replace(",", "")) if qty_match else 0.0

        if action == "hold" or quantity == 0:
            print(f"[parse_action] Regex fallback — hold.\nOutput: {agent_output[:300]}")

        return action, quantity, None, None

    def execute_trade(self, action, quantity, current_price):
        """Cap quantity to notional = risk_pct * equity * leverage. Cash deducted = margin only."""
        quantity = self._cap_quantity(quantity, current_price)  # quantity = notional
        fee = 0.0
        if quantity <= 0:
            return 0.0, 0.0

        margin = quantity / self.leverage  # actual cash required

        if action == "long":
            fee = round(quantity * TAKER_FEE_RATE, 6)
            if self.portfolio["cash"] >= margin + fee:
                self.portfolio["collateral_long"] += round(quantity / current_price, 8)
                self.portfolio["cash"] -= margin + fee
                self.portfolio["price_collateral"] = current_price
                self.trades.append({"action": "long", "qty": quantity, "fee": fee, "price": current_price})
                print(f"  [Fee] Entry taker: ${fee:.4f}")

        elif action == "short":
            fee = round(quantity * TAKER_FEE_RATE, 6)
            if self.portfolio["cash"] >= margin + fee:
                self.portfolio["collateral_short"] += round(quantity / current_price, 8)
                self.portfolio["cash"] -= margin + fee
                self.portfolio["price_collateral"] = current_price
                self.trades.append({"action": "short", "qty": quantity, "fee": fee, "price": current_price})
                print(f"  [Fee] Entry taker: ${fee:.4f}")
        else:
            return 0.0, 0.0

        self.portfolio["total_fees_paid"] = self.portfolio.get("total_fees_paid", 0) + fee
        return quantity, fee

    def sell_collateral(self, current_price):
        """Return margin + unrealized_pnl - exit_fee to cash (leverage-aware)."""
        exit_fee = 0.0
        entry = self.portfolio.get("price_collateral") or 0.0

        if self.portfolio["collateral_short"] != 0:
            fee = 0.0
            if entry > 0:
                notional_initial = self.portfolio["collateral_short"] * entry
                margin = notional_initial / self.leverage
                pnl = self.portfolio["collateral_short"] * (entry - current_price)
                fee = round(self.portfolio["collateral_short"] * current_price * TAKER_FEE_RATE, 4)
                self.portfolio["cash"] += margin + pnl - fee
            self.portfolio["collateral_short"] = 0
            exit_fee += fee

        if self.portfolio["collateral_long"] != 0:
            fee = 0.0
            if entry > 0:
                notional_initial = self.portfolio["collateral_long"] * entry
                margin = notional_initial / self.leverage
                pnl = self.portfolio["collateral_long"] * (current_price - entry)
                fee = round(self.portfolio["collateral_long"] * current_price * TAKER_FEE_RATE, 4)
                self.portfolio["cash"] += margin + pnl - fee
            self.portfolio["collateral_long"] = 0
            exit_fee += fee

        self.portfolio["price_collateral"] = 0.0
        self.portfolio["total_fees_paid"] = self.portfolio.get("total_fees_paid", 0) + exit_fee
        self.portfolio["stop_loss_pct"] = None
        self.portfolio["take_profit_pct"] = None
        return exit_fee

    def _process_day(self, current_date, save_checkpoint: bool = True):
        lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
        candle_iv = "4h" if self.interval == "4h" else "1h"
        if self.interval == "4h":
            current_date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            current_date_str = current_date.strftime("%Y-%m-%d")

        df = get_price_API_HYPERLIQUID(self.crypto, lookback_start, current_date_str, candle_interval=candle_iv)
        if df is None or isinstance(df, str) or df.empty:
            print(f"[Backtester] HyperLiquid no data for {current_date_str}, trying Bybit...")
            df = get_price_API_BINANCE(self.crypto, lookback_start, current_date_str)
        if df is None or isinstance(df, str) or df.empty:
            print(f"[Backtester] No price data for {self.crypto} on {current_date_str}, skipping.")
            return False

        today = df.iloc[-1]
        current_price = float(today["close"])
        day_high = float(today["high"])
        day_low = float(today["low"])

        # Feature 3: track benchmark prices
        if self.benchmark_start_price is None:
            self.benchmark_start_price = current_price
        self.benchmark_end_price = current_price

        # SL/TP check uses today's high/low — must run before agent to close early if hit
        sl_tp_hit = self._check_sl_tp(day_high, day_low)

        # Feature 2: apply funding on any open position before agent decides
        self._apply_daily_funding(current_price)

        try:
            agent_output = self.agent(
                crypto=self.crypto,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio,
            )
        except Exception as e:
            print(f"[Backtester] Agent error on {current_date_str}: {e}, skipping.")
            return False

        time.sleep(self.inter_day_sleep)

        # Feature 6: parse SL/TP from agent output
        action, quantity, sl_pct, tp_pct = self.parse_action(agent_output)

        # Track override frequency so we know what the system actually is
        self.agent_days += 1
        if "[override:" in (agent_output or ""):
            self.override_count += 1

        # Only close position when direction reverses or moves to hold.
        # Same direction = hold the existing position (avoid daily churn fees).
        current_long  = self.portfolio["collateral_long"]  > 0
        current_short = self.portfolio["collateral_short"] > 0
        same_direction = (action == "long" and current_long) or (action == "short" and current_short)

        exit_fee = 0.0
        if not sl_tp_hit and not same_direction:
            exit_fee = self.sell_collateral(current_price)

        # Hold flat when already in the same direction (no pyramiding)
        if same_direction and not sl_tp_hit:
            executed_qty_usd, entry_fee = 0.0, 0.0
        else:
            executed_qty_usd, entry_fee = self.execute_trade(action, quantity, current_price)

        # Store SL/TP for next day check
        if action in ("long", "short") and executed_qty_usd > 0:
            self.portfolio["stop_loss_pct"] = sl_pct
            self.portfolio["take_profit_pct"] = tp_pct

        day_fee = exit_fee + entry_fee
        equity = self._get_equity(current_price)
        self.portfolio["portfolio_value"] = equity

        sl_str = f"{sl_pct*100:.2f}%" if sl_pct else "  -  "
        tp_str = f"{tp_pct*100:.2f}%" if tp_pct else "  -  "
        print(
            f"{current_date_str}  {self.crypto:<5}  {action:<5}  "
            f"qty={executed_qty_usd:>9.2f}  px={current_price:>9.2f}  "
            f"cash={self.portfolio['cash']:>11.2f}  eq={equity:>11.2f}  "
            f"fee={day_fee:>7.4f}  SL={sl_str}  TP={tp_str}"
        )

        self.portfolio_values.append({"Date": current_date, "Portfolio Value": equity})
        if save_checkpoint:
            self._save_state(last_date=current_date_str)
        return True

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run_backtest(self):
        """Run the backtest for the configured date range, then save state."""
        freq = "4h" if self.interval == "4h" else "D"
        all_dates = pd.date_range(self.start_date, self.end_date, freq=freq)
        period_label = f"{len(all_dates)} candles ({self.interval})"

        print(f"\nRunning: {self.crypto}  {self.start_date} -> {self.end_date}  ({period_label})")
        print(f"[Fee model] Taker {TAKER_FEE_RATE*100:.3f}%  Funding {FUNDING_RATE_8H*100:.3f}%/8h x {FUNDING_PERIODS_PER_DAY}/day")
        print(f"[Units]    Qty = USD notional (NOT coin amount)  |  eq = total equity")
        print("-" * 110)

        resume_after = pd.Timestamp(self._last_completed_date) if self._last_completed_date else None

        for current_date in all_dates:
            if resume_after and current_date <= resume_after:
                continue
            self._process_day(current_date)

        final_equity = self.portfolio.get("portfolio_value", self.initial_capital)
        total_fees = self.portfolio.get("total_fees_paid", 0)
        bot_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Feature 4: win rate + trade stats
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get("fee", 0) >= 0 and t.get("qty", 0) > 0)

        # Feature 3: benchmark comparison (Buy & Hold BTC)
        if self.benchmark_start_price and self.benchmark_end_price and self.benchmark_start_price > 0:
            bnh_return = (self.benchmark_end_price - self.benchmark_start_price) / self.benchmark_start_price * 100
            bnh_line = f"  Buy&Hold {self.crypto}  : {bnh_return:+.2f}%  ({self.benchmark_start_price:,.2f} -> {self.benchmark_end_price:,.2f})"
            alpha_line = f"  Alpha vs B&H   : {bot_return - bnh_return:+.2f}%"
        else:
            bnh_line = "  Buy&Hold BTC   : n/a (no benchmark prices)"
            alpha_line = ""

        # Feature 4: max drawdown from portfolio_values
        if len(self.portfolio_values) >= 2:
            vals = [r["Portfolio Value"] for r in self.portfolio_values]
            peak = vals[0]
            max_dd = 0.0
            for v in vals:
                peak = max(peak, v)
                dd = (v - peak) / peak
                if dd < max_dd:
                    max_dd = dd
            drawdown_line = f"  Max Drawdown   : {max_dd*100:.2f}%"
        else:
            drawdown_line = "  Max Drawdown   : n/a"

        # Override rate — tells us what the system actually is
        if self.agent_days > 0:
            override_pct = self.override_count / self.agent_days * 100
            if override_pct >= 60:
                system_label = "trend-following bot (LLM overridden most days)"
            elif override_pct >= 30:
                system_label = "hybrid (LLM + trend anchor)"
            else:
                system_label = "LLM-driven (trend override rarely fires)"
            override_line = (
                f"  Override rate  : {self.override_count}/{self.agent_days} days "
                f"({override_pct:.0f}%)  -> {system_label}"
            )
        else:
            override_line = "  Override rate  : n/a"

        print(f"\n{'='*60}")
        print(f"  Period complete: {self.start_date} -> {self.end_date}")
        print(f"  Bot equity     : {final_equity:,.2f}  (return {bot_return:+.2f}%)")
        print(bnh_line)
        if alpha_line:
            print(alpha_line)
        print(drawdown_line)
        print(f"  Total trades   : {total_trades}")
        print(f"  Total fees paid: {total_fees:,.4f}")
        print(override_line)
        print(f"{'='*60}")

        self._save_state()

    def run_monte_carlo(self, n_runs: int) -> dict:
        """Run n_runs independent backtests; LLM temperature provides variance.

        Each run resets to initial capital — no checkpoints are written between runs.
        Reports mean, std, and 95% CI of total returns so a single lucky/unlucky
        run does not dominate the decision.
        """
        initial_cap = self.initial_capital
        _fresh_portfolio = {
            "cash": initial_cap,
            "collateral_long": 0.0,
            "collateral_short": 0.0,
            "price_collateral": 0.0,
            "leverage": 10,
            "risk": 0.05,
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "total_fees_paid": 0.0,
        }
        returns = []
        fees_per_run = []

        print(f"\n[Monte Carlo] {n_runs} runs x {self.start_date}->{self.end_date}")
        print(f"[Units]  Qty = USD notional  |  [Fees] taker {TAKER_FEE_RATE*100:.3f}% hardcoded, no live funding")

        all_dates = pd.date_range(self.start_date, self.end_date, freq="D")

        for i in range(n_runs):
            print(f"\n{'─'*60}")
            print(f"  Run {i + 1}/{n_runs}")
            self.portfolio = {**_fresh_portfolio}
            self.portfolio_values = []
            self.trades = []
            self.benchmark_start_price = None
            self.benchmark_end_price = None
            self._last_completed_date = None

            for current_date in all_dates:
                self._process_day(current_date, save_checkpoint=False)

            final = self.portfolio.get("portfolio_value", initial_cap)
            ret = (final - initial_cap) / initial_cap
            run_fees = self.portfolio.get("total_fees_paid", 0)
            returns.append(ret)
            fees_per_run.append(run_fees)
            print(f"  Run {i + 1} -> return {ret*100:+.2f}%  fees paid: {run_fees:,.4f}")

        n = len(returns)
        mean_ret = sum(returns) / n
        std_ret = (sum((r - mean_ret) ** 2 for r in returns) / n) ** 0.5
        mean_fees = sum(fees_per_run) / n

        print(f"\n{'='*60}")
        print(f"  Monte Carlo Summary — {n} runs")
        print(f"  Qty unit    : USD notional (agent output), NOT coin amount")
        print(f"  Fee model   : {TAKER_FEE_RATE*100:.3f}% taker (hardcoded), no live funding")
        print(f"  Mean return : {mean_ret*100:+.2f}%")
        print(f"  Std dev     : {std_ret*100:.2f}%")
        if n > 1:
            print(f"  95% CI      : [{(mean_ret - 1.96*std_ret)*100:+.2f}%, {(mean_ret + 1.96*std_ret)*100:+.2f}%]")
        print(f"  Min / Max   : {min(returns)*100:+.2f}% / {max(returns)*100:+.2f}%")
        print(f"  Avg fees/run: {mean_fees:,.4f}")
        print(f"{'='*60}")

        return {"mean": mean_ret, "std": std_ret, "runs": returns, "fees_per_run": fees_per_run}

    # ------------------------------------------------------------------ #
    #  Analysis                                                            #
    # ------------------------------------------------------------------ #

    def analyze_performance(self):
        if not self.portfolio_values:
            print("No portfolio values recorded yet.")
            return None

        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")

        total_return = (
            self.portfolio["portfolio_value"] - self.initial_capital
        ) / self.initial_capital
        print(f"Total Return: {total_return * 100:.2f}%")

        performance_df["Portfolio Value"].plot(
            title="Portfolio Value Over Time", figsize=(12, 6)
        )
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.show()

        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change()
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()
        sharpe_ratio = (mean_daily_return / std_daily_return) * (252 ** 0.5)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = performance_df["Portfolio Value"] / rolling_max - 1
        max_drawdown = drawdown.min()
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

        return performance_df


### Run the Backtest ###
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument("--crypto", type=str, required=True, help="Crypto symbol (e.g. BTC)")
    parser.add_argument(
        "--start-date", type=str,
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date", type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100000,
        help="Starting capital (ignored if checkpoint exists)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing checkpoint and start fresh",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Show performance chart after the run",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Monte Carlo: number of independent runs (default 1 = single run).",
    )
    parser.add_argument(
        "--repeat", type=int, default=None,
        help="Alias for --runs (Monte Carlo repeat count).",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=0.05,
        help="Max equity fraction risked per trade (default 0.05 = 5%%).",
    )
    parser.add_argument(
        "--mode", choices=["ai", "momentum"], default="ai",
        help="'ai' = DeepSeek agent (default)  |  'momentum' = 3-day price rule baseline",
    )
    parser.add_argument(
        "--interval", choices=["1d", "4h"], default="1d",
        help="Candle interval: '1d' = daily (default)  |  '4h' = 4-hour intraday",
    )
    parser.add_argument(
        "--leverage", type=int, default=1,
        help="Leverage multiplier (default 1 = no leverage, e.g. 10 = 10x)",
    )

    args = parser.parse_args()
    n_runs = args.repeat if args.repeat is not None else args.runs

    if args.mode == "momentum":
        agent          = momentum_agent
        ckpt_tag       = "_momentum"
        day_sleep      = 0
        print("[Mode] MOMENTUM BASELINE  (3-day price rule, no LLM)")
    else:
        agent          = run_hedge_fund
        ckpt_tag       = ""
        day_sleep      = 2
        print("[Mode] AI AGENT  (Groq LLM + Mean Reversion)")

    if args.leverage > 1:
        print(f"[Leverage] {args.leverage}x  (notional = risk_pct x equity x {args.leverage})")

    backtester = Backtester(
        agent=agent,
        crypto=args.crypto,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        reset=args.reset,
        risk_pct=args.risk_pct,
        tag=ckpt_tag,
        inter_day_sleep=day_sleep,
        interval=args.interval,
        leverage=args.leverage,
    )

    if n_runs > 1:
        backtester.run_monte_carlo(n_runs)
    else:
        backtester.run_backtest()

    if args.analyze and args.runs == 1:
        backtester.analyze_performance()
