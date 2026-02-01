import pandas as pd
import numpy as np

def calculate_shakeout_score(df):
    if len(df) < 60: return 0, "N/A"
    
    # 1. Lấy dữ liệu gần đây
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    last_30 = df.tail(30)
    
    score = 0
    reasons = []

    # TIÊU CHÍ 1: THANH KHOẢN (Bắt buộc > 400k)
    avg_vol_year = df['volume'].mean()
    if avg_vol_year < 400000:
        return -1, "Vol quá thấp"

    # TIÊU CHÍ 2: ĐANG TRONG TRẠNG THÁI RŨ (Giá nằm dưới MA20 và MA50)
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma50 = df['close'].rolling(50).mean().iloc[-1]
    if latest['close'] < ma20 and latest['close'] < ma50:
        score += 30
        reasons.append("Giá gãy hỗ trợ (Rũ hàng)")

    # TIÊU CHÍ 3: KIỆT VOL KHI GIẢM (Dấu hiệu tổ chức không thoát hàng)
    avg_vol_20 = df['volume'].rolling(20).mean().iloc[-1]
    if latest['volume'] < avg_vol_20:
        score += 40
        reasons.append("Kiệt Vol (Tổ chức giữ hàng)")

    # TIÊU CHÍ 4: VÙNG QUÁ BÁN (Ép bán cực độ)
    # Tính RSI nhanh
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss)).iloc[-1]
    
    if rsi < 30:
        score += 20
        reasons.append("Quá bán (RSI < 30)")
    
    # TIÊU CHÍ 5: NẾN RÚT CHÂN (Bắt đầu có lực cầu đỡ)
    body = abs(latest['open'] - latest['close'])
    lower_shadow = min(latest['open'], latest['close']) - latest['low']
    if lower_shadow > body * 1.5:
        score += 10
        reasons.append("Nến rút chân (Lực cầu ẩn)")

    return score, ", ".join(reasons)

# --- CHẠY LỌC ---
try:
    stocks = pd.read_csv('Hose.csv') # Đảm bảo file này có dữ liệu năm 2026
    stocks.columns = stocks.columns.str.lower()
    
    final_list = []
    for ticker in stocks['symbol'].unique():
        df_ticker = stocks[stocks['symbol'] == ticker].copy().sort_values('date')
        
        score, reason = calculate_shakeout_score(df_ticker)
        
        if score >= 50: # Chỉ lấy các mã có dấu hiệu rõ ràng
            final_list.append({
                'Mã': ticker,
                'Điểm Rũ': score,
                'Giá hiện tại': df_ticker['close'].iloc[-1],
                'Lý do': reason
            })

    results = pd.DataFrame(final_list).sort_values(by='Điểm Rũ', ascending=False)
    
    print("\n--- DANH SÁCH CỔ PHIẾU ĐANG RŨ HÀNG (DÕI THEO TỔ CHỨC) ---")
    if not results.empty:
        print(results.to_string(index=False))
        results.to_csv('Co_Phieu_Theo_Doi.csv', index=False, encoding='utf-8-sig')
    else:
        print("Không có mã nào đạt tiêu chí rũ hàng sạch sẽ.")

except Exception as e:
    print(f"Lỗi: {e}")