import pandas as pd
import numpy as np

def add_indicators(df):
    # Đảm bảo cột giá là số thực
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # 1. Moving Average (MA)
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Bollinger Bands (BB)
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['ma20'] + (df['std'] * 2)
    df['lower_bb'] = df['ma20'] - (df['std'] * 2)
    df['bb_width'] = df['upper_bb'] - df['lower_bb']
    
    # 4. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    return df

# --- CHƯƠNG TRÌNH CHÍNH ---
try:
    # 1. Kiểm tra VN-Index
    vni = pd.read_csv('VNINDEX.csv')
    vni.columns = vni.columns.str.lower()
    vni = add_indicators(vni)
    
    v_latest = vni.iloc[-1]
    v_prev = vni.iloc[-2]
    
    # Điều kiện VNI: MA20 > MA50, MA20 đang hướng lên, BB đang mở rộng
    vni_ok = (v_latest['ma20'] > v_latest['ma50']) and \
             (v_latest['ma20'] > v_prev['ma20']) and \
             (v_latest['bb_width'] > v_prev['bb_width'])

    print(f"--- PHÂN TÍCH THỊ TRƯỜNG CHUNG ---")
    print(f"VN-Index Trend: {'TỐT' if vni_ok else 'THẬN TRỌNG'}")
    print(f"RSI VN-Index: {v_latest['rsi']:.1f}")
    print("-" * 40)

    # 2. Lọc Cổ phiếu
    stocks = pd.read_csv('Hose.csv')
    stocks.columns = stocks.columns.str.lower()
    recommendations = []

    for ticker in stocks['symbol'].unique():
        df = stocks[stocks['symbol'] == ticker].copy()
        if len(df) < 50: continue
        
        df = add_indicators(df)
        s_latest = df.iloc[-1]
        s_prev = df.iloc[-2]
        
        # ĐIỀU KIỆN MUA:
        # - VNI phải ổn
        # - Giá nằm trên MA20
        # - MACD cắt lên Signal (Giao cắt vàng)
        # - RSI chưa quá 65
        if vni_ok:
            if (s_latest['close'] > s_latest['ma20']) and \
               (s_latest['macd'] > s_latest['signal'] and s_prev['macd'] <= s_prev['signal']) and \
               (s_latest['rsi'] < 65):
                
                price = s_latest['close']
                recommendations.append({
                    'Mã': ticker,
                    'Giá Mua': price,
                    'Target (+15%)': round(price * 1.15, 0),
                    'Stoploss': round(price * 0.93, 0), # -7%
                    'RSI': round(s_latest['rsi'], 1)
                })

    if recommendations:
        result_df = pd.DataFrame(recommendations)
        print("DANH SÁCH ĐIỂM MUA TIỀM NĂNG:")
        print(result_df.to_string(index=False))
        result_df.to_csv('KHUYEN_NGHI.csv', index=False, encoding='utf-8-sig')
        print("\n--- Đã lưu kết quả vào file KHUYEN_NGHI.csv ---")
    else:
        print("Không tìm thấy mã nào thỏa mãn điểm mua kỹ thuật.")

except Exception as e:
    print(f"Lỗi: {e}. Hãy kiểm tra lại tên cột trong file CSV (Date, Open, High, Low, Close, Volume, Symbol)")