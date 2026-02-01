import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V37.5 - FINAL CORE", layout="wide")

# --- DANH MỤC MÃ ---
NGANH_MASTER = {
    "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
    "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
    "THÉP": ['HPG','NKG','HSG'], 
    "NGÂN HÀNG": ['VCB','TCB','MBB','STB'],
    "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','GEX']
}
ALL_TICKERS = [t for sub in NGANH_MASTER.values() for t in sub]

# --- HÀM TÍNH TOÁN CORE (ĐÃ FIX LỖI CẤU TRÚC) ---
def calculate_master_signals(df):
    if df is None or len(df) < 15: return None
    df = df.copy()
    
    # 1. Xử lý triệt để cấu trúc cột (Chống lỗi Multi-Index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    # Đảm bảo có cột Date để sắp xếp
    if 'date' not in df.columns:
        df = df.reset_index()
        df.columns = [str(col).strip().lower() for col in df.columns]
    
    # Xóa cột trùng và dòng trùng
    df = df.loc[:, ~df.columns.duplicated()]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'close']).drop_duplicates(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if len(df) < 15: return None

    # 2. Chuyển đổi dữ liệu số
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Tính toán chỉ báo (Dòng tiền & Kỹ thuật)
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # Money In (Vol vượt 1.2 trung bình 20 phiên)
    df['money_in'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.2)
    
    # Chấm điểm
    score = 0
    last = df.iloc[-1]
    if last['close'] > last['ma10']: score += 3
    if last['close'] > last['ma20']: score += 2
    if last['money_in']: score += 3
    if last['rsi'] > 50: score += 2
    
    df['total_score'] = score
    df['is_buy'] = (df['close'] > df['ma20']) & (df['money_in'])
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏆 TRADING V37.5")
    ticker_input = st.text_input("🔍 SOI MÃ CHI TIẾT:", "MWG").upper()
    
    if st.button("🚀 KÍCH HOẠT HỆ THỐNG (MỚI)", use_container_width=True):
        with st.spinner("Đang xây dựng lại dữ liệu sạch..."):
            all_data = []
            for m in ALL_TICKERS:
                t = yf.download(f"{m}.VN", period="1y", interval="1d", progress=False)
                if not t.empty:
                    # Reset index để đưa Date thành cột
                    t = t.reset_index()
                    t['symbol'] = m
                    all_data.append(t)
            
            if all_data:
                full_df = pd.concat(all_data, ignore_index=True)
                full_df.to_csv("master_data.csv", index=False)
                st.success("Hệ thống đã KÍCH HOẠT!")
                st.rerun()

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ FIREANT", "📊 DÒNG TIỀN NGÀNH", "🎯 LỌC SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ CHÍNH ---
if os.path.exists("master_data.csv"):
    hose_df = pd.read_csv("master_data.csv")
    
    if menu == "📈 ĐỒ THỊ FIREANT":
        st.subheader(f"📊 BIỂU ĐỒ KỸ THUẬT: {ticker_input}")
        df_ticker = hose_df[hose_df['symbol'] == ticker_input].copy()
        df_m = calculate_master_signals(df_ticker)
        
        if df_m is not None:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # Tầng 1: Nến & MA
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # Điểm mua
            buy_pts = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=buy_pts['date'], y=buy_pts['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], name="Volume", marker_color='dodgerblue'), row=2, col=1)
            
            # Tầng 3: RSI
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"🚩 {ticker_input} - Điểm: {df_m['total_score'].iloc[-1]}/10")
        else:
            st.warning(f"Không có dữ liệu cho mã {ticker_input}. Hãy nhấn nút KÍCH HOẠT ở sidebar.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN THEO NHÓM NGÀNH")
        summary = []
        for n, mãs in NGANH_MASTER.items():
            pts = []
            for m in mãs:
                subset = hose_df[hose_df['symbol'] == m].copy()
                d = calculate_master_signals(subset)
                if d is not None: pts.append(d['total_score'].iloc[-1])
            avg = np.mean(pts) if pts else 0
            summary.append({"Ngành": n, "Sức Mạnh": round(avg, 1), "Số mã hợp lệ": len(pts)})
        
        st.table(pd.DataFrame(summary).sort_values("Sức Mạnh", ascending=False))

    elif menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.subheader("🚀 SIÊU ĐIỂM MUA: TIỀN VÀO + NỀN GIÁ")
        results = []
        for s in hose_df['symbol'].unique():
            d = calculate_master_signals(hose_df[hose_df['symbol'] == s].copy())
            if d is not None:
                l = d.iloc[-1]
                if l['total_score'] >= 6:
                    results.append({"Mã": s, "Điểm Dòng Tiền": l['total_score'], "RSI": round(l['rsi'],1)})
        
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("Điểm Dòng Tiền", ascending=False), use_container_width=True)
        else:
            st.info("Chưa có mã nào đạt tiêu chuẩn mua mạnh.")
else:
    st.info("Chào mừng! Hãy nhấn '🚀 KÍCH HOẠT HỆ THỐNG (MỚI)' để bắt đầu phân tích.")
