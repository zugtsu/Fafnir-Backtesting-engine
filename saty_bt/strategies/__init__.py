from . import base  # noqa: F401  (registers gates)
from . import classics, compression_breakout, indicator_lab, full_stack, golden_gate, level_fade, orb, osc_reversal, random_entry, ribbon_pullback, vomy

STRATEGIES = {
    "golden_gate": golden_gate.build_signals,            # A
    "ribbon_pullback": ribbon_pullback.build_signals,    # B
    "osc_reversal": osc_reversal.build_signals,          # C
    "compression_breakout": compression_breakout.build_signals,  # D
    "full_stack": full_stack.build_signals,              # E
    "level_retest": golden_gate.build_signals,           # F - same signal engine, different fill/exit geometry
    "saty_system": golden_gate.build_signals,            # G - Saty's own three-indicator system as he describes it
    "gate_static_r": golden_gate.build_signals,          # H - 38.2 close, PDC stop, static 1.5R target
    "fx_pips": golden_gate.build_signals,                # P - USDJPY fixed 20-pip target, limit at the level, PDC stop
    "spx_gate_fisher": golden_gate.build_signals,        # H2 - optimised SPX: gate 0.45, Fisher veto, all day, PDC, 2R
    "spx_h2_final": golden_gate.build_signals,           # H2 + 14:30 ET cutoff, the recommended SPX trade
    "usdjpy_pips_opt": golden_gate.build_signals,        # P2 - optimised USDJPY: limit 5 bars, 15 pips, PDC, all day
    "level_fade": level_fade.build_signals,              # L - fade the 1.0 ATR level back toward the 61.8
    "orb": orb.build_signals,
    "usdjpy_london_orb": orb.build_signals,              # O2 - USDJPY: London 07-09 UTC range, break by 14:00, stop other side, hold to day end
    "indicator_lab": indicator_lab.build_signals,      # I - the textbook indicators, one exit geometry
    "classics": classics.build_signals,                  # daily-bar yardsticks: donchian | rsi2                            # O - opening range breakout / London range / Williams open+ATR
    "vomy": vomy.build_signals,                          # V - Saty's ribbon reversal (vomiting dolphin)
    "random_entry": random_entry.build_signals,          # validation baseline
}

__all__ = ["STRATEGIES", "base", "golden_gate", "ribbon_pullback", "osc_reversal",
           "compression_breakout", "full_stack"]
