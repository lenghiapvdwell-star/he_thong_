import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="V40 - SMART MONEY DETECTOR", layout="wide")

# --- HÀM ĐỌC DỮ LIỆU THÔNG MINH ---
def load_data(file_name):
    if not os.path.exists(file_name):
        return None
    df = pd.read_csv(file_name)
    # Chuẩn hóa tên cột
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Tự động tìm cột mã cổ phiếu
    for col in ['symbol', 'ticker', 'mã', 'ma']:
        if col in df.columns:
            df = df.rename(columns={col: 'symbol'})
            break
    # Tự động tìm cột ngày
    for col in ['date', 'ngày', 'time']:
        if col in df.columns:
            df = df.rename(columns={col: 'date'})
            break
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# --- HÀM TÍNH TOÁN DÒNG TIỀN TỔ CHỨC (CORE) ---
def calculate_signals(df):
    if df is None or len(df) < 10: return None
    df = df.sort_values('date').copy()
    
    # Ép kiểu số
    for c in ['close', 'open', 'high', 'low', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close', 'volume'])
    
    # 1. Chỉ báo xu hướng
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    # 2. Dòng tiền tổ chức (Smart Money)
    # Vol đột biến > 1.5 lần trung bình 20 phiên + Giá tăng > 2%
    df['vol_20'] = df['volume'].rolling(20).mean()
    df['is_smart_money'] = (df['volume'] > df['vol_20'] * 1.5) & (df['close'] > df['close'].shift(1) * 1.02)
    
    # 3. Sức mạnh giá (Relative Strength)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # 4. Chấm điểm tổng hợp (Thang điểm 10)
    score = 0
    last = df.iloc[-1]
    if last['close'] > last['ma10']: score += 2 # Xu hướng ngắn
    if last['close'] > last['ma20']: score += 2 # Xu hướng trung
    if last['is_smart_money']: score += 4      # Tiền tổ chức vào mạnh
    if last['rsi'] > 55: score += 2            # Sức mạnh giá tốt
    
    df['total_score'] = score
    return df

# --- GIAO DIỆN SIDEBAR ---
with st.sidebar:
    st.header("🏆 SMART MONEY V40")
    ticker = st.text_input("🔍 SOI MÃ CỤ THỂ:", "HPG").upper()
    
    st.divider()
    
    # Nút check sức khỏe VNI
    check_vni = st.button("📈 SỨC KHỎE VN-INDEX", use_container_width=True)
    
    st.divider()
    menu = st.radio("CHỨC NĂNG CHÍNH:", 
                    ["📈 ĐỒ THỊ DÒNG TIỀN", "📊 BẢNG DÒNG TIỀN NGÀNH", "🎯 LỌC SIÊU ĐIỂM MUA"])

# --- XỬ LÝ CHÍNH ---
hose_df = load_data("hose.csv")
vni_df = load_data("vnindex.csv")

if hose_df is not None:
    # 1. CHỨC NĂNG SỨC KHỎE VNI
    if check_vni:
        st.subheader("📊 PHÂN TÍCH SỨC KHỎE THỊ TRƯỜNG CHUNG (VNI)")
        if vni_df is not None:
            vni_signal = calculate_signals(vni_df)
            last_vni = vni_signal.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("Điểm VNI", f"{last_vni['total_score']}/10")
            col2.metric("Trạng thái", "TÍCH CỰC" if last_vni['total_score'] >= 5 else "RỦI RO")
            col3.metric("RSI VNI", round(last_vni['rsi'], 1))
            st.info("Lời khuyên: Chỉ nên giải ngân mạnh khi Điểm VNI > 5.")
        else:
            st.error("Thiếu file vnindex.csv để phân tích.")

    # 2. CHỨC NĂNG ĐỒ THỊ
    if menu == "📈 ĐỒ THỊ DÒNG TIỀN":
        df_ticker = hose_df[hose_df['symbol'] == ticker]
        data = calculate_signals(df_ticker)
        if data is not None:
            st.subheader(f"📊 PHÂN TÍCH DÒNG TIỀN: {ticker}")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Giá & MA
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20 (Nền)"), row=1, col=1)
            
            # Đánh dấu Smart Money
            sm = data[data['is_smart_money']]
            fig.add_trace(go.Scatter(x=sm['date'], y=sm['low']*0.97, mode='markers+text', text="💰", textfont=dict(size=18), name="Tiền vào"), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Khối lượng", marker_color='rgba(100, 149, 237, 0.6)'), row=2, col=1)
            
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Điểm sức mạnh {ticker}: {data['total_score'].iloc[-1]}/10")
        else:
            st.error(f"Không tìm thấy dữ liệu cho mã {ticker} trong file hose.csv")

    # 3. CHỨC NĂNG NGÀNH
    elif menu == "📊 BẢNG DÒNG TIỀN NGÀNH":
        st.subheader("🌊 THEO DÕI DÒNG TIỀN THEO NGÀNH")
        nganh_dict = {
            "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
            "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
            "THÉP": ['HPG','NKG','HSG'], 
            "BANK": ['VCB','TCB','MBB','STB'],
            "BĐS": ['DIG','PDR','VHM','GEX']
        }
        summary = []
        for n, mãs in nganh_dict.items():
            pts = []
            for m in mãs:
                d = calculate_signals(hose_df[hose_df['symbol'] == m])
                if d is not None: pts.append(d['total_score'].iloc[-1])
            summary.append({"Ngành": n, "Sức mạnh dòng tiền": round(np.mean(pts),1) if pts else 0, "Số mã quét": len(pts)})
        
        st.table(pd.DataFrame(summary).sort_values("Sức mạnh dòng tiền", ascending=False))

    # 4. LỌC SIÊU ĐIỂM MUA
    elif menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.subheader("🚀 DANH SÁCH CỔ PHIẾU CÓ TỔ CHỨC GOM HÀNG")
        buy_list = []
        for s in hose_df['symbol'].unique():
            d = calculate_signals(hose_df[hose_df['symbol'] == s])
            if d is not None:
                last = d.iloc[-1]
                if last['total_score'] >= 7: # Chỉ lọc mã cực mạnh
                    buy_list.append({
                        "Mã": s,
                        "Điểm": last['total_score'],
                        "Dòng tiền": "🔥 MẠNH" if last['is_smart_money'] else "ỔN ĐỊNH",
                        "RSI": round(last['rsi'], 1),
                        "Giá hiện tại": last['close']
                    })
        
        if buy_list:
            st.dataframe(pd.DataFrame(buy_list).sort_values("Điểm", ascending=False), use_container_width=True)
        else:
            st.info("Thị trường đang tích lũy, chưa có mã đạt điểm mua bùng nổ.")
else:
    st.error("❌ KHÔNG ĐỌC ĐƯỢC FILE!")
    st.info("Hãy kiểm tra: 1. File phải tên là 'hose.csv'. 2. Trong file phải có cột 'symbol' hoặc 'ticker'.")
