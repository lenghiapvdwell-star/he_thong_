import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V39.0 - LOCAL DATA ENGINE", layout="wide")

# --- DANH MỤC NGÀNH ---
NGANH_MASTER = {
    "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
    "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
    "THÉP": ['HPG','NKG','HSG'], 
    "NGÂN HÀNG": ['VCB','TCB','MBB','STB'],
    "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','GEX']
}

# --- HÀM TÍNH TOÁN CORE ---
def calculate_indicators(df_raw):
    if df_raw is None or len(df_raw) < 5: 
        return None
    
    df = df_raw.copy()
    
    # Chuẩn hóa tên cột về chữ thường
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Ép kiểu số
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['close'])
    if len(df) < 5: return None

    # Chỉ báo kỹ thuật cơ bản
    df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # Điểm dòng tiền
    vol_avg = df['volume'].rolling(window=20, min_periods=1).mean()
    df['money_in'] = df['volume'] > (vol_avg * 1.1)
    
    # Logic chấm điểm
    score = 0
    last = df.iloc[-1]
    if last['close'] >= last['ma10']: score += 4
    if last['close'] >= last['ma20']: score += 3
    if last['money_in']: score += 3
    
    df['total_score'] = score
    return df

# --- SIDEBAR: ĐỌC FILE LOCAL ---
with st.sidebar:
    st.header("⚙️ DATA LOCAL ENGINE")
    ticker_input = st.text_input("🔍 SOI MÃ (HPG, SSI...):", "HPG").upper()
    
    st.info("Hệ thống đang sử dụng dữ liệu từ: \n- hose.csv \n- vnindex.csv")
    
    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ FIREANT", "📊 DÒNG TIỀN NGÀNH", "🎯 LỌC ĐIỂM MUA"])

# --- XỬ LÝ DỮ LIỆU ---
if os.path.exists("hose.csv") and os.path.exists("vnindex.csv"):
    # Đọc dữ liệu từ file upload của bạn
    full_hose = pd.read_csv("hose.csv")
    vni_data = pd.read_csv("vnindex.csv")
    
    # Chuẩn hóa cột symbol và date
    full_hose.columns = [str(c).strip().lower() for c in full_hose.columns]
    if 'symbol' in full_hose.columns:
        full_hose['symbol'] = full_hose['symbol'].str.strip().upper()
    
    # Chuyển đổi ngày tháng
    date_col = 'date' if 'date' in full_hose.columns else 'Date'
    full_hose['date_clean'] = pd.to_datetime(full_hose[date_col.lower()], errors='coerce')

    if menu == "📈 ĐỒ THỊ FIREANT":
        st.subheader(f"📊 PHÂN TÍCH KỸ THUẬT: {ticker_input}")
        # Lọc mã chính xác
        df_mã = full_hose[full_hose['symbol'] == ticker_input].copy()
        df_final = calculate_indicators(df_mã)
        
        if df_final is not None:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # Tầng 1: Giá
            fig.add_trace(go.Candlestick(x=df_final['date_clean'], open=df_final['open'], high=df_final['high'], low=df_final['low'], close=df_final['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_final['date_clean'], y=df_final['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            
            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=df_final['date_clean'], y=df_final['volume'], name="Volume", marker_color='dodgerblue'), row=2, col=1)
            
            # Tầng 3: RSI
            fig.add_trace(go.Scatter(x=df_final['date_clean'], y=df_final['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)

            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Điểm Kỹ Thuật {ticker_input}: {df_final['total_score'].iloc[-1]}/10")
        else:
            st.error(f"Không tìm thấy dữ liệu cho mã '{ticker_input}' trong file hose.csv")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN (Dữ liệu Offline)")
        res_nganh = []
        for n, mãs in NGANH_MASTER.items():
            pts = []
            for m in mãs:
                subset = full_hose[full_hose['symbol'] == m].copy()
                d = calculate_indicators(subset)
                if d is not None:
                    pts.append(d['total_score'].iloc[-1])
            
            if len(pts) > 0:
                avg = np.mean(pts)
                res_nganh.append({"Ngành": n, "Sức Mạnh": round(avg, 1), "Số mã hợp lệ": len(pts)})
            else:
                res_nganh.append({"Ngành": n, "Sức Mạnh": 0.0, "Số mã hợp lệ": 0})
        
        st.table(pd.DataFrame(res_nganh).sort_values("Sức Mạnh", ascending=False))

    elif menu == "🎯 LỌC ĐIỂM MUA":
        st.subheader("🚀 QUÉT SIÊU ĐIỂM MUA TRONG FILE HOSE.CSV")
        results = []
        all_unique_symbols = full_hose['symbol'].unique()
        for s in all_unique_symbols:
            d = calculate_indicators(full_hose[full_hose['symbol'] == s].copy())
            if d is not None:
                l = d.iloc[-1]
                if l['total_score'] >= 7:
                    results.append({"Mã": s, "Điểm": l['total_score'], "RSI": round(l['rsi'],1)})
        
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("Điểm", ascending=False), use_container_width=True)
        else:
            st.info("Không có mã nào đủ tiêu chuẩn điểm mua (>7đ).")

else:
    st.error("❌ THIẾU FILE DỮ LIỆU!")
    st.write("Vui lòng đảm bảo file **hose.csv** và **vnindex.csv** nằm cùng thư mục với file code.")
