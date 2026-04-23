# -*- coding: utf-8 -*-
def analyze_user_trade(entry, tp, sl, position, current_indicators):
    ema21 = current_indicators.get('EMA_21', entry)
    bb_upper = current_indicators.get('BB_U', entry * 1.05)
    bb_lower = current_indicators.get('BB_L', entry * 0.95)

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr_ratio = reward / risk if risk > 0 else 0

    suggestions = []

    # Phân tích Entry
    if position == "LONG":
        if entry > bb_upper:
            suggestions.append("⚠️ Entry đang ở vùng Quá Mua (ngoài BB Upper), rủi ro đu đỉnh cao.")
        if entry > ema21 * 1.02:
            suggestions.append(f"💡 Entry hơi xa EMA21. Nên chờ Pullback về vùng {ema21:.2f} để tối ưu.")
    elif position == "SHORT":
        if entry < bb_lower:
            suggestions.append("⚠️ Entry đang ở vùng Quá Bán (ngoài BB Lower), rủi ro bán đáy cao.")

    # Phân tích RR
    if rr_ratio < 1.5:
        suggestions.append(f"❌ Kèo có tỉ lệ RR quá thấp ({rr_ratio:.2f} < 1.5). Rủi ro cao, không nên vào lệnh.")
    elif rr_ratio > 4:
        suggestions.append("🚩 TP quá xa so với biến động trung bình, cân nhắc chốt lời từng phần.")

    return {
        "rr": round(rr_ratio, 2),
        "suggestions": suggestions,
        "is_valid": rr_ratio >= 1.0
    }