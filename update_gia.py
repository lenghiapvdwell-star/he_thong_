import pandas as pd
from vnstock3 import Vnstock
from datetime import datetime, timedelta
import os
import numpy as np

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def update_data():
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=50)).strftime('%Y-%m-%d')

    # 1. CẬP NHẬT VNINDEX
    if os.path.exists('VNINDEX.csv'):
        print("\n--- ĐANG PHÂN TÍCH SỨC KHỎE VNINDEX ---")
        try:
            vni_data = Vnstock().stock(symbol='VNINDEX', source='VCI').quote.history(start=start_date, end=today_str)
            if not vni_data.empty:
                vni_data.columns = vni_data.columns.str.lower()
                vni_data['date'] = pd.to_datetime(vni_data['time']).dt.strftime('%Y-%m-%d')
                
                # Tính toán các chỉ số kỹ thuật
                current_close = vni_data['close'].iloc[-1]
                vni_data['rsi'] = calculate_rsi(vni_data['close'])
                current_rsi = round(vni_data['rsi'].iloc[-1], 2)
                
                # So sánh Volume
                vol_now = vni_data['volume'].iloc[-1]
                vol_avg_20 = vni_data['volume'].tail(20).mean()
                vol_ratio = round(vol_now / vol_avg_20, 2)
                
                # Biến động 5 phiên
                change_5p = round(((current_close / vni_data['close'].iloc[-5]) - 1) * 100, 2)

                print(f"{'='*50}")
                print(f"CHỈ SỐ VNINDEX: {current_close:.2f}")
                print(f"RSI (14 ngày): {current_rsi} ({'QUÁ BÁN - CƠ HỘI' if current_rsi < 35 else 'BÌNH THƯỜNG' if current_rsi < 70 else 'QUÁ MUA - RỦI RO'})")
                print(f"DÒNG TIỀN: Gấp {vol_ratio} lần trung bình 20 ngày")
                print(f"BIẾN ĐỘNG 5 PHIÊN: {change_5p}%")
                
                status = "🔥 CẨN THẬN BULLTRAP" if (change_5p < 0 and vol_ratio > 1.2) else "✅ ĐANG PHỤC HỒI" if (current_rsi > 30 and change_5p > 0) else "⏳ ĐANG TÍCH LŨY"
                print(f"KHUYẾN NGHỊ VNI: {status}")
                print(f"{'='*50}")

                # Lưu vào CSV
                vni_new = vni_data[['date', 'open', 'high', 'low', 'close', 'volume']]
                vni_new['symbol'] = 'VNINDEX'
                old_vni = pd.read_csv('VNINDEX.csv')
                old_vni['date'] = pd.to_datetime(old_vni['date']).dt.strftime('%Y-%m-%d')
                pd.concat([old_vni, vni_new]).drop_duplicates(subset=['date'], keep='last').to_csv('VNINDEX.csv', index=False)
        except Exception as e:
            print(f"⚠️ Lỗi VNI: {e}")

    # 2. CẬP NHẬT HOSE
    if os.path.exists('hose.csv'):
        print("\n--- Đang cập nhật dữ liệu sàn HOSE ---")
        df_old_hose = pd.read_csv('hose.csv')
        tickers = df_old_hose['symbol'].unique()
        all_new_data = []
        for ticker in tickers:
            try:
                stock_data = Vnstock().stock(symbol=ticker, source='VCI').quote.history(start=start_date, end=today_str)
                if not stock_data.empty:
                    stock_data.columns = stock_data.columns.str.lower()
                    stock_data['symbol'] = ticker
                    stock_data['date'] = pd.to_datetime(stock_data['time']).dt.strftime('%Y-%m-%d')
                    all_new_data.append(stock_data[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']])
            except: continue
        if all_new_data:
            df_new = pd.concat(all_new_data)
            df_old_hose['date'] = pd.to_datetime(df_old_hose['date']).dt.strftime('%Y-%m-%d')
            pd.concat([df_old_hose, df_new]).drop_duplicates(subset=['date', 'symbol'], keep='last').to_csv('hose.csv', index=False)
            print(f"🚀 HOSE: Đã cập nhật xong {len(tickers)} mã.")

if __name__ == "__main__":
    update_data()