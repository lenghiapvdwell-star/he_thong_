import pandas as pd
import numpy as np

def monitor_pro_system(vni_file, hose_file, watch_list):
    try:
        # 1. Đọc dữ liệu
        vni = pd.read_csv(vni_file)
        vni.columns = vni.columns.str.lower()
        vni = vni.sort_values('date')
        
        hose = pd.read_csv(hose_file)
        hose.columns = hose.columns.str.lower()
        
        # Tính mức thay đổi của VN-Index trong 5 phiên để làm chuẩn RS
        vni_change = (vni['close'].iloc[-1] / vni['close'].iloc[-5] - 1) * 100

        print(f"\n{'='*85}")
        print(f"{'MÃ':<6} | {'GIÁ HT':<8} | {'ĐIỂM RS':<8} | {'DÒNG TIỀN':<12} | {'TRẠNG THÁI':<15} | {'KÍCH HOẠT'}")
        print(f"{'-'*85}")

        results = []
        for ticker in watch_list:
            df = hose[hose['symbol'] == ticker].copy().sort_values('date')
            if len(df) < 20: continue
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # A. Tính Điểm RS (Sức mạnh tương đối so với VNI)
            stock_change = (latest['close'] / df['close'].iloc[-5] - 1) * 100
            rs_score = round(stock_change - vni_change, 2)
            
            # B. Kiểm tra Dòng tiền (Money Flow)
            avg_vol_20 = df['volume'].tail(20).mean()
            vol_ratio = latest['volume'] / avg_vol_20
            if vol_ratio > 1.5:
                money_flow = "BÙNG NỔ 🚀"
            elif vol_ratio > 1.0:
                money_flow = "ỔN ĐỊNH ✅"
            else:
                money_flow = "YẾU ⏳"

            # C. Trạng thái và Điểm kích hoạt
            ma20 = df['close'].tail(20).mean()
            # Điểm kích hoạt là giá cao nhất 2 phiên gần nhất (vượt đỉnh ngắn hạn để xác nhận rũ xong)
            trigger_price = df['high'].tail(2).max()
            
            status = "KHOẺ 💪" if rs_score > 0 else "YẾU 📉"
            if latest['close'] > ma20:
                status += " + Uptrend"
            else:
                status += " + Dưới MA20"

            print(f"{ticker:<6} | {latest['close']:<8.0f} | {rs_score:<8} | {money_flow:<12} | {status:<15} | > {trigger_price:<8.0f}")

        print(f"{'-'*85}")
        print("💡 GIẢI THÍCH:")
        print("- ĐIỂM RS > 0: Cổ phiếu đang kháng lại đà giảm của thị trường tốt hơn VNI.")
        print("- DÒNG TIỀN BÙNG NỔ: Có dấu hiệu tổ chức 'vét máng' hoặc bắt đầu đánh lên.")
        print("- KÍCH HOẠT: Mức giá cần vượt qua để xác nhận kết thúc nhịp rũ hàng.")

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")

# --- CẤU HÌM CHẠY ---
my_watch_list = ['SSI', 'VND', 'DIG', 'SHB', 'HPG', 'VPB', 'GEX', 'MBB', 'VHM', 'VIC']
monitor_pro_system('VNINDEX.csv', 'Hose.csv', my_watch_list)