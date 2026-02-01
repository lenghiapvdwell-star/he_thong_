import yfinance as yf
import pandas as pd
from google.colab import files
import time
from datetime import datetime, timedelta

# 1. Cấu hình
tickers = ["HPG.VN", "SSI.VN", "VND.VN", "VIX.VN", "STB.VN", "SHB.VN", "MBB.VN", "VPB.VN", "DIG.VN", "GEX.VN", "VHM.VN", "VIC.VN"] # Thêm tiếp mã vào đây
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')

print(f"--- ĐANG QUÉT DỮ LIỆU OHLCV (2 NĂM) ---")

all_data = []

for symbol in tickers:
    try:
        # Tải dữ liệu 2 năm
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if not df.empty:
            # San phẳng Multi-index của Yahoo
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Kiểm tra điều kiện Volume trung bình 10 phiên gần nhất > 400k
            avg_vol = float(df['Volume'].iloc[-10:].mean())
            
            if avg_vol > 400000:
                # Reset index để đưa cột 'Date' vào dữ liệu
                df = df.reset_index()
                
                # Thêm cột Symbol
                df['symbol'] = symbol.replace(".VN", "")
                
                # Chọn và sắp xếp đúng 7 cột bạn yêu cầu
                # Lưu ý: Yahoo trả về Date, Open, High, Low, Close, Adj Close, Volume
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
                
                # Chuyển tên cột thành chữ thường cho đúng ý bạn
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                
                all_data.append(df)
                print(f"✅ Đã lấy 2 năm dữ liệu cho: {symbol}")
        
        time.sleep(0.2)
    except Exception as e:
        print(f"Lỗi tại {symbol}: {e}")

# 2. Gộp tất cả và tải về
if all_data:
    final_master_df = pd.concat(all_data, ignore_index=True)
    
    # Lưu file
    file_name = 'du_lieu_2nam_7cot.csv'
    final_master_df.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    print(f"\n--- HOÀN THÀNH! TỔNG CỘNG {len(final_master_df)} DÒNG DỮ LIỆU ---")
    files.download(file_name)
else:
    print("Không lấy được dữ liệu. Hãy kiểm tra lại danh sách mã.")