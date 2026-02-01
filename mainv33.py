import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V37.2 - FINAL CORE", layout="wide")

# --- DANH MỤC MÃ ---
NGANH_MASTER = {
    "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
    "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
    "THÉP": ['HPG','NKG','HSG'], 
    "NGÂN HÀNG": ['VCB','TCB','MBB','STB'],
    "BĐS": ['DIG','PDR','VHM','GEX']
}
ALL_TICKERS = [t for sub in NGANH_MASTER.values() for t in sub]

# --- HÀM TÍNH TOÁN (XỬ LÝ TRÙNG TÊN CỘT) ---
def calculate_master_signals(df, vni_df):
    if df is None or len(df) < 10: return None
    df = df.copy()
    
    # 1. Xử lý tên cột trùng lặp
    df.columns = [str(col).strip().lower() for col in df.columns]
    # Giữ lại duy nhất 1 cột nếu bị trùng tên 'date'
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'date' not in df.columns:
        df = df.reset_index()
        df.columns = [str(col).strip().lower() for col in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

    # 2. Làm sạch dữ liệu
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.drop_duplicates(subset=['date']).dropna(subset=['close', 'date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    if len(df) < 15: return None
    
    # 3. Tính toán chỉ báo
    c = pd.to_numeric(df['close'], errors='coerce')
    v = pd.to_numeric(df['volume'], errors='coerce')
    
    df['ma10'] = c.rolling(10).mean()
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # RSI
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # RS (Sức mạnh giá)
    df['money_in'] = (v > v.rolling(20).mean() * 1.1)
    
    # Chấm điểm
    score = 0
    try:
        last = df.iloc[-1]
        if last['close'] >= last['ma10']: score += 4
        if last['close'] >= last['ma20']: score += 3
        if last['money_in']: score += 3
    except: score = 0
    
    df['total_score'] = score
    df['is_buy'] = (c > df['ma20']) & (df['money_in'])
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏆 TRADING V37.2")
    ticker_input = st.text_input("🔍 SOI MÃ CHI TIẾT:", "MWG").upper()
    
    if st.button("🔄 LÀM MỚI TOÀN BỘ HỆ THỐNG", use_container_width=True):
        with st.spinner("Đang tải dữ liệu sạch..."):
            vni = yf.download("^VNINDEX", period="1y")
            vni.to_csv("vni_v37.csv")
            
            all_data = []
            for m in ALL_TICKERS:
                t = yf.download(f"{m}.VN", period="1y", progress=False)
                if not t.empty:
                    t = t.reset_index()
                    t['symbol'] = m
                    all_data.append(t)
            
            if all_data:
                pd.concat(all_data).to_csv("hose_v37.csv", index=False)
                st.success("Hệ thống đã đồng bộ!")
                st.rerun()

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ FIREANT", "📊 DÒNG TIỀN NGÀNH", "🎯 LỌC SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ ---
if os.path.exists("vni_v37.csv") and os.path.exists("hose_v37.csv"):
    vni_df = pd.read_csv("vni_v37.csv")
    hose_df = pd.read_csv("hose_v37.csv")

    if menu == "📈 ĐỒ THỊ FIREANT":
        st.subheader(f"📊 BIỂU ĐỒ KỸ THUẬT: {ticker_input}")
        # Lọc mã và reset index để tránh lỗi label
        df_ticker = hose_df[hose_df['symbol'] == ticker_input].copy()
        df_m = calculate_master_signals(df_ticker, vni_df)
        
        if df_m is not None:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
            
            # Tầng 1: Giá
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # Điểm mua
            buy_pts = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=buy_pts['date'], y=buy_pts['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="MUA"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], name="Volume", marker_color='dodgerblue'), row=2, col=1)
            
            # Tầng 3: RSI
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"🚩 {ticker_input} - Điểm Dòng Tiền: {df_m['total_score'].iloc[-1]}/10")
        else:
            st.error("Không có dữ liệu. Hãy nhấn 'LÀM MỚI TOÀN BỘ HỆ THỐNG'.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN THEO NHÓM NGÀNH")
        summary = []
        for n, mãs in NGANH_MASTER.items():
            pts = []
            for m in mãs:
                subset = hose_df[hose_df['symbol'] == m].copy()
                d = calculate_master_signals(subset, vni_df)
                if d is not None: pts.append(d['total_score'].iloc[-1])
            avg = np.mean(pts) if pts else 0
            summary.append({"Ngành": n, "Sức Mạnh": round(avg, 1), "Số mã": len(pts)})
        
        st.table(pd.DataFrame(summary).sort_values("Sức Mạnh", ascending=False))

    elif menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.write("### 🚀 QUÉT TÍN HIỆU MUA")
        results = []
        for s in hose_df['symbol'].unique():
            d = calculate_master_signals(hose_df[hose_df['symbol'] == s].copy(), vni_df)
            if d is not None:
                l = d.iloc[-1]
                if l['total_score'] >= 6:
                    results.append({"Mã": s, "Điểm": l['total_score'], "RSI": round(l['rsi'],1)})
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("Điểm", ascending=False), use_container_width=True)
        else:
            st.info("Chưa tìm thấy mã đạt tiêu chuẩn.")
else:
    st.info("Nhấn 'LÀM MỚI TOÀN BỘ HỆ THỐNG' để bắt đầu.")
