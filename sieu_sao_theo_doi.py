import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

def get_live_data(watch_list):
    print("...Đang check tín hiệu dòng tiền trực tuyến...")
    tickers = [t + ".VN" for t in watch_list]
    # Lấy thêm dữ liệu Volume để check bùng nổ
    data = yf.download(tickers, period="1d", interval="1m", progress=False)
    
    live_info = {}
    if not data.empty:
        for ticker in watch_list:
            symbol = ticker + ".VN"
            live_info[ticker] = {
                'price': data['Close'][symbol].iloc[-1],
                'volume': data['Volume'][symbol].sum() # Tổng vol khớp từ sáng
            }
    return live_info

def monitor_pro_live(vni_file, hose_file, watch_list):
    try:
        vni = pd.read_csv(vni_file)
        vni.columns = vni.columns.str.lower()
        hose = pd.read_csv(hose_file)
        hose.columns = hose.columns.str.lower()
        
        live_data = get_live_data(watch_list)
        vni_change = (vni['close'].iloc[-1] / vni['close'].iloc[-5] - 1) * 100

        print(f"\n{'='*120}")
        print(f"{'MÃ':<6} | {'GIÁ LIVE':<10} | {'ĐIỂM RS':<8} | {'DÒNG TIỀN':<12} | {'TÍN HIỆU':<15} | {'TARGET':<10} | {'STOPLOSS'}")
        print(f"{'-'*120}")

        for ticker in watch_list:
            df = hose[hose['symbol'] == ticker].copy().sort_values('date')
            if len(df) < 20: continue
            
            # Dữ liệu lịch sử & Live
            hist_price = df['close'].iloc[-1]
            info = live_data.get(ticker, {'price': hist_price, 'volume': 0})
            live_p = info['price']
            live_v = info['volume']
            
            # 1. Tính RS thực tế (Real-time RS)
            stock_change = (live_p / df['close'].iloc[-5] - 1) * 100
            rs_score = round(stock_change - vni_change, 2)
            
            # 2. Check Volume bùng nổ (Vol live > 70% trung bình cả ngày là đạt)
            avg_vol_20 = df['volume'].tail(20).mean()
            vol_ratio = live_v / avg_vol_20
            money_flow = "BÙNG NỔ 🚀" if vol_ratio > 0.8 else "YẾU ⏳"
            
            # 3. Điểm kích hoạt & Lọc Bulltrap
            trigger_p = df['high'].tail(2).max()
            
            # ĐIỀU KIỆN XÁC NHẬN (BREAKOUT THẬT): Giá vượt + RS dương + Vol ổn
            if live_p >= trigger_p and rs_score > 0:
                advice = ">>> MUA <<<"
                status = "XÁC NHẬN NỔ 🔥"
            elif live_p >= trigger_p and rs_score <= 0:
                advice = "BẪY BULLTRAP ⚠️"
                status = "HỒI ẢO"
            else:
                advice = "Theo dõi"
                status = "Đang rũ"

            target = live_p * 1.15
            stoploss = live_p * 0.93

            print(f"{ticker:<6} | {live_p:<10.0f} | {rs_score:<8} | {money_flow:<12} | {advice:<15} | {target:<10.0f} | {stoploss:<.0f}")

        print(f"{'-'*120}")
        print("💡 CẢNH BÁO: Chỉ vào lệnh khi tín hiệu là '>>> MUA <<<' và Dòng tiền 'BÙNG NỔ 🚀'.")

    except Exception as e:
        print(f"Lỗi: {e}")

my_watch_list = ['SSI', 'VND', 'DIG', 'SHB', 'HPG', 'VPB', 'GEX', 'MBB', 'VHM', 'VIC']
monitor_pro_live('VNINDEX.csv', 'hose.csv', my_watch_list)