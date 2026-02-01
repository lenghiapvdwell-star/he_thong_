import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import warnings
import os

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
st.set_page_config(page_title="V33.4 - FIREANT STABLE", layout="wide")

# --- HÀM TÍNH TOÁN KỸ THUẬT (Sửa lỗi KeyError & Type) ---
def calculate_pro_signals(df, vni_df):
    if df is None or len(df) < 20: return None
    df = df.copy()
    
    # Ép tên cột về chữ thường để tránh lỗi KeyError
    df.columns = [str(col).lower() for col in df.columns]
    
    # Kiểm tra lại các cột thiết yếu
    required = ['close', 'high', 'low', 'open', 'volume']
    for col in required:
        if col not in df.columns:
            # Nếu thiếu, thử tìm cột có tên tương tự (ví dụ 'adj close')
            alt = [c for c in df.columns if col in c]
            if alt: df[col] = df[alt[0]]
            else: return None

    # Ép kiểu số thực
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['close', 'volume']).reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    # 1. MA & Bollinger
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    df['bb_width'] = (c.rolling(20).std() * 4) / df['ma20']
    
    # 2. RSI
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    
    # 3. RS (Sức mạnh tương đối)
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce') # Lấy cột giá VNI
    df['rs'] = round(((c/c.shift(5)) - (vni_c.iloc[-1]/vni_c.iloc[-5])) * 100, 2)
    
    # 4. Tín hiệu 💣 & 🏹
    df['is_bomb'] = df['bb_width'] <= df['bb_width'].rolling(30).min()
    df['is_buy'] = (c > df['ma20']) & (df['ma20'] >= df['ma50'] * 0.99) & (v > v.rolling(20).mean() * 1.3)
    
    df['target_1'] = round(c * 1.12, 0)
    df['target_2'] = round(c * 1.25, 0)
    df['stop_loss'] = round(df['ma20'] * 0.96, 0)
    return df

# --- SIDEBAR & TẢI DỮ LIỆU ---
with st.sidebar:
    st.header("🚀 FIREANT PRO V33.4")
    if st.button("🔄 CẬP NHẬT DỮ LIỆU"):
        with st.spinner("Đang tải dữ liệu..."):
            # Tải VNI
            vni = yf.download("^VNINDEX", period="2y")
            if isinstance(vni.columns, pd.MultiIndex): vni.columns = vni.columns.get_level_values(0)
            vni.reset_index().to_csv("VNINDEX_local.csv", index=False)
            
            # Tải Danh sách mã
            mã_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','DIG','PDR','VHM','FPT','DGC']
            all_h = []
            for m in mã_list:
                t = yf.download(f"{m}.VN", period="2y", progress=False)
                if isinstance(t.columns, pd.MultiIndex): t.columns = t.columns.get_level_values(0)
                t = t.reset_index()
                t['symbol'] = m
                all_h.append(t)
            pd.concat(all_h).to_csv("hose_local.csv", index=False)
            st.success("✅ Đã cập nhật! Hãy chọn mã.")
            st.rerun()

    ticker = st.text_input("NHẬP MÃ:", "MWG").upper()

# --- HIỂN THỊ CHART ---
if os.path.exists("VNINDEX_local.csv"):
    vni_df = pd.read_csv("VNINDEX_local.csv")
    hose_df = pd.read_csv("hose_local.csv")
    
    df_c = calculate_pro_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
    
    if df_c is not None:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
        
        # Cột ngày tháng: Thường là 'Date' hoặc 'date'
        date_col = 'date' if 'date' in df_c.columns else df_c.columns[0]
        
        # 1. Nến & MA
        fig.add_trace(go.Candlestick(x=df_c[date_col], open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name="Giá"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_c[date_col], y=df_c['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_c[date_col], y=df_c['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
        
        # 💣 Bom & 🏹 Mua
        b = df_c[df_c['is_bomb']]
        fig.add_trace(go.Scatter(x=b[date_col], y=b['high']*1.02, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
        s = df_c[df_c['is_buy']]
        fig.add_trace(go.Scatter(x=s[date_col], y=s['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

        # 2. Volume
        colors = ['red' if r['open'] > r['close'] else 'green' for i, r in df_c.iterrows()]
        fig.add_trace(go.Bar(x=df_c[date_col], y=df_c['volume'], marker_color=colors, name="Vol"), row=2, col=1)

        # 3. RSI & RS
        fig.add_trace(go.Scatter(x=df_c[date_col], y=df_c['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_c[date_col], y=df_c['rs'], line=dict(color='magenta'), name="RS"), row=3, col=1)

        # CẤU HÌNH KÉO DÃN FIREANT
        fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan',
                          yaxis=dict(side='right', fixedrange=False, autorange=True),
                          yaxis2=dict(side='right', fixedrange=False, autorange=True),
                          yaxis3=dict(side='right', fixedrange=False, autorange=True),
                          xaxis=dict(fixedrange=False))
        
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        
        l = df_c.iloc[-1]
        st.success(f"🎯 T1: {int(l['target_1'])} | T2: {int(l['target_2'])} | 🛑 Stop: {int(l['stop_loss'])}")
else:
    st.warning("⚠️ Nhấn 'CẬP NHẬT DỮ LIỆU' để bắt đầu.")
