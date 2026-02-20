"""
Broker Basket Export — One-click CSV for Zerodha Basket Orders
================================================================
Generates a CSV compatible with Zerodha Kite basket order import.
Also supports generic CSV format for other brokers.
"""

import pandas as pd
import io
from typing import List, Dict, Optional


def generate_zerodha_basket(signals: list, capital: float = 500000,
                            risk_pct: float = 1.0,
                            position_multiplier: float = 1.0) -> bytes:
    """
    Generate Zerodha Kite Basket Order CSV.
    
    Zerodha format:
    Instrument, Qty, Price, LTP, Trigger Price, Order Type, Product, Validity
    
    Args:
        signals: List of ScanResult objects
        capital: Trading capital
        risk_pct: Risk per trade %
        position_multiplier: Regime-based position sizing multiplier
    """
    rows = []
    
    for sig in signals:
        # Calculate position size
        risk_amount = capital * (risk_pct / 100) * position_multiplier
        risk_per_share = abs(sig.entry - sig.stop_loss)
        if risk_per_share <= 0:
            continue
        
        qty = max(1, int(risk_amount / risk_per_share))
        
        # Cap at 20% of capital
        max_qty = int((capital * 0.20) / sig.entry)
        qty = min(qty, max_qty)
        
        # Determine order type
        if sig.entry_type == "AT CMP":
            order_type = "MARKET"
            price = 0
            trigger = 0
        else:
            order_type = "SL"
            price = round(sig.entry * 1.002, 2)  # Small buffer above trigger
            trigger = round(sig.entry, 2)
        
        # Transaction type
        txn_type = "BUY" if sig.signal == "BUY" else "SELL"
        
        # NSE exchange symbol
        instrument = f"NSE:{sig.symbol}-EQ" if "-" not in sig.symbol else f"NSE:{sig.symbol}"
        
        rows.append({
            "Instrument": instrument,
            "Qty": qty,
            "Price": price,
            "Trigger Price": trigger,
            "Order Type": order_type,
            "Product": "CNC",  # Delivery
            "Validity": "DAY",
        })
    
    if not rows:
        return b""
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def generate_generic_basket(signals: list, capital: float = 500000,
                            risk_pct: float = 1.0,
                            position_multiplier: float = 1.0,
                            regime: str = "") -> bytes:
    """
    Generate generic broker CSV with full trade plan.
    Includes entry, SL, targets, position size, and risk metrics.
    """
    rows = []
    total_risk = 0
    
    for sig in signals:
        risk_amount = capital * (risk_pct / 100) * position_multiplier
        risk_per_share = abs(sig.entry - sig.stop_loss)
        if risk_per_share <= 0:
            continue
        
        qty = max(1, int(risk_amount / risk_per_share))
        max_qty = int((capital * 0.20) / sig.entry)
        qty = min(qty, max_qty)
        
        position_value = qty * sig.entry
        trade_risk = qty * risk_per_share
        total_risk += trade_risk
        
        rows.append({
            "Symbol": sig.symbol,
            "Signal": sig.signal,
            "Strategy": sig.strategy,
            "Entry": round(sig.entry, 2),
            "Entry_Type": sig.entry_type,
            "Stop_Loss": round(sig.stop_loss, 2),
            "Target_1": round(sig.target_1, 2),
            "Target_2": round(sig.target_2, 2),
            "Target_3": round(sig.target_3, 2),
            "Risk_Reward": round(sig.risk_reward, 1),
            "Qty": qty,
            "Position_Value": round(position_value, 2),
            "Risk_Amount": round(trade_risk, 2),
            "Risk_Pct": round((trade_risk / capital) * 100, 2),
            "Confidence": sig.confidence,
            "RS_Rating": round(sig.rs_rating, 0),
            "Sector": getattr(sig, 'sector', ''),
            "Regime": regime,
            "SQI": getattr(sig, 'sqi', ''),
            "SQI_Grade": getattr(sig, 'sqi_grade', ''),
        })
    
    if not rows:
        return b""
    
    df = pd.DataFrame(rows)
    
    # Add summary row
    summary = pd.DataFrame([{
        "Symbol": "=== SUMMARY ===",
        "Signal": f"Capital: ₹{capital:,.0f}",
        "Strategy": f"Risk/Trade: {risk_pct}%",
        "Entry": f"Total Risk: ₹{total_risk:,.0f}",
        "Entry_Type": f"Heat: {(total_risk/capital)*100:.1f}%",
        "Stop_Loss": f"Positions: {len(rows)}",
        "Target_1": f"Regime: {regime}",
        "Target_2": "",
        "Target_3": "",
        "Risk_Reward": "",
        "Qty": "",
        "Position_Value": round(sum(r["Position_Value"] for r in rows), 2),
        "Risk_Amount": round(total_risk, 2),
        "Risk_Pct": round((total_risk / capital) * 100, 2),
        "Confidence": "",
        "RS_Rating": "",
        "Sector": "",
        "Regime": "",
        "SQI": "",
        "SQI_Grade": "",
    }])
    
    df = pd.concat([df, summary], ignore_index=True)
    return df.to_csv(index=False).encode("utf-8")
