import pandas as pd
import numpy as np

# Hàm tính toán các chỉ báo kỹ thuật
def add_indicators(df):
    # 1. Tính Moving Average
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # 2. Tính RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Tính Bollinger Bands
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['ma20'] + (df['std'] * 2)
    df['lower_bb'] = df['ma20'] - (df['std'] * 2)
    df['bb_width'] = df['upper_bb'] - df['lower_bb']
    
    # 4. Tính MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    return df

# --- BƯỚC 1: KIỂM TRA VN-INDEX ---
def check_vni_health(vni_path):
    vni = pd.read_csv(vni_path)
    vni = add_indicators(vni)
    
    latest = vni.iloc[-1]
    prev = vni.iloc[-2]
    
    # Điều kiện lọc VN-Index
    is_trend_up = latest['ma20'] > latest['ma50'] and latest['ma20'] > prev['ma20']
    is_rsi_ok = 40 < latest['rsi'] < 70
    is_vol_ok = latest['volume'] > vni['volume'].tail(20).mean() * 0.8 # Dòng tiền không quá yếu
    is_bb_expanding = latest['bb_width'] > prev['bb_width']
    
    status = is_trend_up and is_rsi_ok and is_vol_ok
    return status, latest['rsi']

# --- BƯỚC 2: TÌM ĐIỂM MUA CỔ PHIẾU ---
def find_buy_signals(stock_data_path, vni_status):
    if not vni_status:
        print("⚠️ CẢNH BÁO: Thị trường chung (VN-Index) chưa ổn định. Hạn chế mua mới!")
        return
    
    df_all = pd.read_csv(stock_data_path)
    symbols = df_all['symbol'].unique()
    recommendations = []
    
    for ticker in symbols:
        df = df_all[df_all['symbol'] == ticker].copy()
        df = add_indicators(df)
        
        if len(df) < 50: continue
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Tiêu chuẩn chọn điểm mua (Buy Signal)
        # 1. Giá cắt lên MA20 hoặc nằm trên MA20
        # 2. MACD cắt lên đường Signal
        # 3. RSI đang hướng lên và < 65
        buy_cond = (latest['close'] > latest['ma20']) and \
                   (latest['macd'] > latest['signal']) and \
                   (prev['macd'] <= prev['signal']) and \
                   (latest['rsi'] < 65)
        
        if buy_cond:
            price = latest['close']
            target = price * 1.15  # Target +15%
            stoploss = latest['ma50'] if latest['ma50'] < price else price * 0.93 # Cắt lỗ tại MA50 hoặc -7%
            
            recommendations.append({
                'Mã': ticker,
                'Giá mua': price,
                'Target': round(target, 1),
                'Stoploss': round(stoploss, 1),
                'RSI': round(latest['rsi'], 1),
                'Khuyến nghị': 'MUA MỚI'
            })
            
    return pd.DataFrame(recommendations)

# --- THỰC THI ---
# Giả sử bạn đã có 2 file này trong folder Colab
vni_ok, vni_rsi = check_vni_health('Vnindex.csv')
print(f"Sức khỏe VN-Index: {'TỐT' if vni_ok else 'XẤU'} (RSI: {vni_rsi})")

final_recs = find_buy_signals('du_lieu_2nam_7cot.csv', vni_ok)
if final_recs is not None and not final_recs.empty:
    print("\n--- DANH SÁCH KHUYẾN NGHỊ ---")
    print(final_recs)
else:
    print("Không có mã nào đạt điểm mua chuẩn.")