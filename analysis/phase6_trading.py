from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Literal, Sequence

import numpy as np


Direction = Literal["UP", "DOWN"]
ExitReason = Literal[
    "MODEL_REVERSAL",
    "STOP_LOSS",
    "TAKE_PROFIT",
    "TIME_STOP",
    "DATA_END",
]


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


@dataclass(frozen=True)
class DailyBar:
    date: date
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        for value in (self.open, self.high, self.low, self.close):
            if not np.isfinite(float(value)):
                raise ValueError("Daily OHLC values must be finite.")
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("Daily OHLC values must be positive.")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("OHLC bar is internally inconsistent.")


@dataclass(frozen=True)
class TradeSignal:
    date: date
    direction: Direction
    probability_pct: float
    baseline_probability_pct: float
    eligible: bool
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.probability_pct) <= 100.0:
            raise ValueError("probability_pct must be between 0 and 100.")
        if not 0.0 <= float(self.baseline_probability_pct) <= 100.0:
            raise ValueError("baseline_probability_pct must be between 0 and 100.")


@dataclass(frozen=True)
class TradePolicyConfig:
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_holding_days: int = 20
    exit_on_model_reversal: bool = True

    def __post_init__(self) -> None:
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0.0:
            raise ValueError("stop_loss_pct must be positive.")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0.0:
            raise ValueError("take_profit_pct must be positive.")
        if self.max_holding_days < 1:
            raise ValueError("max_holding_days must be >= 1.")
        if (
            self.stop_loss_pct is None
            and self.take_profit_pct is None
            and not self.exit_on_model_reversal
        ):
            raise ValueError("At least one exit mechanism must be enabled.")


@dataclass(frozen=True)
class TradeRecord:
    direction: Direction
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    gross_return_pct: float
    exit_reason: ExitReason
    holding_days: int

    @property
    def pnl_directional_pct(self) -> float:
        raw = (self.exit_price - self.entry_price) / self.entry_price * 100.0
        return raw if self.direction == "UP" else -raw

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "gross_return_pct": self.gross_return_pct,
            "exit_reason": self.exit_reason,
            "holding_days": self.holding_days,
            "pnl_directional_pct": self.pnl_directional_pct,
        }


@dataclass(frozen=True)
class TradeSimulationResult:
    trades: tuple[TradeRecord, ...]
    total_trades: int
    winning_trades: int
    win_rate_pct: float
    total_directional_return_pct: float
    average_trade_return_pct: float
    max_holding_days: int
    protocol: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades": [trade.as_dict() for trade in self.trades],
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate_pct": self.win_rate_pct,
            "total_directional_return_pct": self.total_directional_return_pct,
            "average_trade_return_pct": self.average_trade_return_pct,
            "max_holding_days": self.max_holding_days,
            "protocol": self.protocol,
        }


def _trade_return(direction: Direction, entry_price: float, exit_price: float) -> float:
    raw = (float(exit_price) - float(entry_price)) / float(entry_price) * 100.0
    return raw if direction == "UP" else -raw


def _next_model_signal(
    signals: Sequence[TradeSignal],
    after_date: date,
    through_date: date,
) -> TradeSignal | None:
    candidate: TradeSignal | None = None
    for signal in signals:
        signal_date = _as_date(signal.date)
        if signal_date <= after_date:
            continue
        if signal_date > through_date:
            break
        candidate = signal
        break
    return candidate


def simulate_trades(
    signals: Iterable[TradeSignal],
    bars: Iterable[DailyBar],
    policy: TradePolicyConfig,
) -> TradeSimulationResult:
    """Simulate non-overlapping trades from point-in-time signals.

    The first entry is the first daily bar strictly after the signal date.
    Exits use only bars from the entry bar forward. Model-reversal checks use
    only signals strictly after entry.
    """

    ordered_signals = sorted(
        [signal for signal in signals if signal.eligible],
        key=lambda signal: _as_date(signal.date),
    )
    ordered_bars = sorted(bars, key=lambda bar: _as_date(bar.date))

    if not ordered_signals or not ordered_bars:
        return TradeSimulationResult(
            trades=(),
            total_trades=0,
            winning_trades=0,
            win_rate_pct=0.0,
            total_directional_return_pct=0.0,
            average_trade_return_pct=0.0,
            max_holding_days=0,
            protocol="NEXT_BAR_ENTRY_DAILY_OHLC",
        )

    trades: list[TradeRecord] = []
    next_available_bar = 0

    for signal in ordered_signals:
        signal_date = _as_date(signal.date)

        while (
            next_available_bar < len(ordered_bars)
            and _as_date(ordered_bars[next_available_bar].date) <= signal_date
        ):
            next_available_bar += 1

        if next_available_bar >= len(ordered_bars):
            break

        entry_index = next_available_bar
        entry_bar = ordered_bars[entry_index]
        entry_price = float(entry_bar.open)
        direction = signal.direction

        stop_price = None
        target_price = None

        if policy.stop_loss_pct is not None:
            stop_price = (
                entry_price * (1.0 - policy.stop_loss_pct / 100.0)
                if direction == "UP"
                else entry_price * (1.0 + policy.stop_loss_pct / 100.0)
            )

        if policy.take_profit_pct is not None:
            target_price = (
                entry_price * (1.0 + policy.take_profit_pct / 100.0)
                if direction == "UP"
                else entry_price * (1.0 - policy.take_profit_pct / 100.0)
            )

        exit_index = len(ordered_bars) - 1
        exit_price = float(ordered_bars[exit_index].close)
        exit_reason: ExitReason = "DATA_END"

        for index in range(entry_index, len(ordered_bars)):
            bar = ordered_bars[index]
            bar_date = _as_date(bar.date)

            stop_hit = False
            target_hit = False

            if stop_price is not None:
                stop_hit = (
                    bar.low <= stop_price
                    if direction == "UP"
                    else bar.high >= stop_price
                )

            if target_price is not None:
                target_hit = (
                    bar.high >= target_price
                    if direction == "UP"
                    else bar.low <= target_price
                )

            if stop_hit and target_hit:
                exit_index = index
                exit_price = float(stop_price)
                exit_reason = "STOP_LOSS"
                break

            if stop_hit:
                exit_index = index
                exit_price = float(stop_price)
                exit_reason = "STOP_LOSS"
                break

            if target_hit:
                exit_index = index
                exit_price = float(target_price)
                exit_reason = "TAKE_PROFIT"
                break

            bars_held = index - entry_index + 1
            if bars_held >= policy.max_holding_days:
                exit_index = index
                exit_price = float(bar.close)
                exit_reason = "TIME_STOP"
                break

            if policy.exit_on_model_reversal:
                reversal = _next_model_signal(
                    ordered_signals,
                    after_date=signal_date,
                    through_date=bar_date,
                )
                if reversal is not None and reversal.direction != direction:
                    exit_index = index
                    exit_price = float(bar.close)
                    exit_reason = "MODEL_REVERSAL"
                    break

        exit_date = _as_date(ordered_bars[exit_index].date)
        holding_days = exit_index - entry_index + 1
        gross_return = _trade_return(direction, entry_price, exit_price)

        trades.append(
            TradeRecord(
                direction=direction,
                entry_date=_as_date(entry_bar.date),
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                gross_return_pct=gross_return,
                exit_reason=exit_reason,
                holding_days=holding_days,
            )
        )

        next_available_bar = exit_index + 1


    wins = sum(1 for trade in trades if trade.gross_return_pct > 0.0)
    total_return = float(sum(trade.gross_return_pct for trade in trades))
    average_return = total_return / len(trades) if trades else 0.0
    maximum_holding = max((trade.holding_days for trade in trades), default=0)

    return TradeSimulationResult(
        trades=tuple(trades),
        total_trades=len(trades),
        winning_trades=wins,
        win_rate_pct=(wins / len(trades) * 100.0) if trades else 0.0,
        total_directional_return_pct=total_return,
        average_trade_return_pct=average_return,
        max_holding_days=maximum_holding,
        protocol="NEXT_BAR_ENTRY_DAILY_OHLC",
    )
