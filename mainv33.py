import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V38.0 - REALTIME TERMINAL", layout="wide")

# --- DANH MỤC NGÀNH ---
NGANH_MASTER = {
    "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
    "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
    "THÉP": ['HPG','NKG','HSG'], 
    "NGÂN HÀNG": ['VCB','TCB','MBB','STB'],
    "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','GEX']
}
ALL_TICKERS = [t for sub in NGANH_MASTER.values() for t in sub]

# --- HÀM XỬ LÝ DỮ LIỆU CHUẨN ---
def process_df(df):
    if df is None or df.empty: return None
    df = df.copy()
    # Fix lỗi cấu trúc Yahoo Finance mới
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if 'date' not in df.columns:
        df = df.reset_index()
    df.columns = [str(col).strip().lower() for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'close']).drop_duplicates(subset=['date'])
    return df.sort_values('date').reset_index(drop=True)

# --- HÀM TÍNH CHỈ BÁO ---
def calculate_indicators(df):
    df = process_df(df)
    if df is None or len(df) < 15: return None
    
    # Chỉ báo kỹ thuật
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # Dòng tiền & Điểm
    df['money_in'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.15)
    
    score = 0
    last = df.iloc[-1]
    if last['close'] > last['ma10']: score += 4
    if last['close'] > last['ma20']: score += 3
    if last['money_in']: score += 3
    df['total_score'] = score
    df['is_buy'] = (last['close'] > last['ma20']) and last['money_in']
    return df

# --- SIDEBAR: CẬP NHẬT CỔ PHIẾU ---
with st.sidebar:
    st.header("⚙️ CONTROL PANEL")
    ticker_input = st.text_input("🔍 SOI MÃ (HPG, MWG...):", "HPG").upper()
    
    if st.button("🔄 CẬP NHẬT CỔ PHIẾU (Real-time)", use_container_width=True):
        with st.spinner("Đang kết nối sàn HOSE..."):
            # 1. Tải VNINDEX
            vni = yf.download("^VNINDEX", period="1y", interval="1d", progress=False)
            vni.to_csv("vnindex.csv")
            
            # 2. Tải danh mục cổ phiếu
            all_list = []
            for m in ALL_TICKERS:
                t = yf.download(f"{m}.VN", period="1y", interval="1d", progress=False)
                if not t.empty:
                    t = t.reset_index()
                    t['symbol'] = m
                    all_list.append(t)
            
            if all_list:
                pd.concat(all_list, ignore_index=True).to_csv("hose.csv", index=False)
                st.success("Đã cập nhật dữ liệu Real-time!")
                st.rerun()

    menu = st.radio("MENU:", ["📈 ĐỒ THỊ FIREANT", "📊 DÒNG TIỀN NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ ---
if os.path.exists("hose.csv"):
    hose_df = pd.read_csv("hose.csv")
    
    if menu == "📈 ĐỒ THỊ FIREANT":
        st.subheader(f"📊 PHÂN TÍCH CHI TIẾT: {ticker_input}")
        data_mã = hose_df[hose_df['symbol'] == ticker_input].copy()
        df_final = calculate_indicators(data_mã)
        
        if df_final is not None:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # Nến & MA
            fig.add_trace(go.Candlestick(x=df_final['date'], open=df_final['open'], high=df_final['high'], low=df_final['low'], close=df_final['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_final['date'], y=df_final['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=df_final['date'], y=df_final['volume'], name="Volume", marker_color='dodgerblue'), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df_final['date'], y=df_final['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)

            fig.update_layout(height=750, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"HPG/Market Score: {df_final['total_score'].iloc[-1]}/10")
        else:
            st.error(f"Lỗi: Mã {ticker_input} không có dữ liệu. Nhấn 'Cập nhật cổ phiếu'.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN (Scale 10)")
        res = []
        for n, mãs in NGANH_MASTER.items():
            scores = []
            for m in mãs:
                d = calculate_indicators(hose_df[hose_df['symbol'] == m])
                if d is not None: scores.append(d['total_score'].iloc[-1])
            res.append({"Ngành": n, "Điểm": round(np.mean(scores),1) if scores else 0, "Số mã": len(scores)})
        st.table(pd.DataFrame(res).sort_values("Điểm", ascending=False))

    elif menu == "🎯 SIÊU ĐIỂM MUA":
        st.subheader("🚀 QUÉT ĐIỂM MUA ĐỘT BIẾN")
        buy_list = []
        for s in hose_df['symbol'].unique():
            d = calculate_indicators(hose_df[hose_df['symbol'] == s])
            if d is not None and d['total_score'].iloc[-1] >= 7:
                buy_list.append({"Mã": s, "Điểm": d['total_score'].iloc[-1], "RSI": round(d['rsi'].iloc[-1],1)})
        st.dataframe(pd.DataFrame(buy_list).sort_values("Điểm", ascending=False), use_container_width=True)
else:
    st.info("Hệ thống chưa có dữ liệu. Vui lòng nhấn nút '🔄 CẬP NHẬT CỔ PHIẾU' ở bên trái.")
