import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="V33.5 - MOBILE PRO", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU CHUẨN (Fix lỗi hiển thị điện thoại) ---
def clean_and_calculate(df, vni_df):
    if df is None or len(df) < 20: return None
    df = df.copy()
    
    # Làm phẳng dữ liệu (Flatten Multi-Index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).lower() for col in df.columns]
    
    # Ép kiểu số thực để tính toán mượt trên mobile
    cols = ['close', 'open', 'high', 'low', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).reset_index()
    close = df['close']
    
    # Chỉ báo kỹ thuật
    df['ma20'] = close.rolling(20).mean()
    df['ma50'] = close.rolling(50).mean()
    df['rsi'] = 100 - (100 / (1 + (close.diff().where(close.diff() > 0, 0).ewm(14).mean() / 
                                  -close.diff().where(close.diff() < 0, 0).ewm(14).mean())))
    
    # Tín hiệu Mua & Bom (Nén BB)
    std = close.rolling(20).std()
    df['bb_w'] = (std * 4) / df['ma20']
    df['is_bomb'] = df['bb_w'] <= df['bb_w'].rolling(30).min()
    df['is_buy'] = (close > df['ma20']) & (df['ma20'] > df['ma50']) & (df['rsi'] < 75)
    
    return df

# --- SIDEBAR MOBILE ---
with st.sidebar:
    st.title("📱 V33.5 MOBILE")
    # Nút bấm to để dễ nhấn trên điện thoại
    if st.button("🔄 CẬP NHẬT DỮ LIỆU", use_container_width=True):
        with st.spinner("Đang tải..."):
            vni = yf.download("^VNINDEX", period="1y")
            vni.to_csv("vni_mobile.csv")
            
            mã = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','FPT']
            data_all = []
            for m in mã:
                t = yf.download(f"{m}.VN", period="1y", progress=False)
                t['symbol'] = m
                data_all.append(t)
            pd.concat(data_all).to_csv("hose_mobile.csv")
            st.success("Xong! Hãy soi mã.")
            st.rerun()

    ticker = st.text_input("MÃ CẦN SOI:", "MWG").upper()

# --- HIỂN THỊ CHART ---
if os.path.exists("vni_mobile.csv"):
    vni_data = pd.read_csv("vni_mobile.csv", header=[0,1] if "vni" in "" else 0)
    hose_data = pd.read_csv("hose_mobile.csv")
    
    # Lấy dữ liệu riêng cho mã đã chọn
    df_m = clean_and_calculate(hose_data[hose_data['symbol'] == ticker].copy(), None)
    
    if df_m is not None:
        # Giảm chiều cao xuống 600 để vừa màn hình dọc điện thoại
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        
        # 1. Chart Nến & Tín hiệu
        fig.add_trace(go.Candlestick(x=df_m['Date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name="Giá"), row=1, col=1)
        
        # Điểm MUA Mũi tên
        buys = df_m[df_m['is_buy']]
        fig.add_trace(go.Scatter(x=buys['Date'], y=buys['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="BUY"), row=1, col=1)
        
        # 2. Volume
        fig.add_trace(go.Bar(x=df_m['Date'], y=df_m['volume'], name="Vol", marker_color='gray'), row=2, col=1)
        
        # 3. RSI
        fig.add_trace(go.Scatter(x=df_m['Date'], y=df_m['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Cấu hình kéo dãn (Pan/Zoom) mượt cho cảm ứng điện thoại
        fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan',
                          margin=dict(l=5, r=40, t=20, b=20),
                          yaxis=dict(side='right', fixedrange=False),
                          yaxis2=dict(side='right', fixedrange=False),
                          yaxis3=dict(side='right', fixedrange=False))
        
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
        
        # Chỉ số tóm tắt Mobile
        l = df_m.iloc[-1]
        st.markdown(f"**Giá:** {l['close']:.1f} | **RSI:** {l['rsi']:.1f}")
        st.success(f"🎯 Target: {l['close']*1.1:.0f} | 🛑 Stop: {l['ma20']:.0f}")

else:
    st.info("Chưa có dữ liệu. Nhấn nút Cập Nhật ở menu trái (biểu tượng ☰ trên điện thoại).")
