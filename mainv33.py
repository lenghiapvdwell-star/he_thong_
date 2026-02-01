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
st.set_page_config(page_title="V33.3 - FIREANT ULTIMATE FIX", layout="wide")

# --- HÀM TÍNH TOÁN KỸ THUẬT (Đã sửa lỗi TypeError) ---
def calculate_pro_signals(df, vni_df):
    if df is None or len(df) < 50: return None
    df = df.copy()
    
    # Ép kiểu dữ liệu về số thực để tránh lỗi TypeError khi diff()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['close', 'volume']).reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    # 1. MA & Bollinger
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    df['bb_width'] = (c.rolling(20).std() * 4) / df['ma20']
    
    # 2. RSI (Sửa lỗi diff)
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    
    # 3. ADX
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    pdm = pd.Series(np.where((h.diff()>l.shift(1)-l)&(h.diff()>0), h.diff(), 0))
    mdm = pd.Series(np.where((l.shift(1)-l>h.diff())&(l.shift(1)-l>0), l.shift(1)-l, 0))
    pdi = 100 * (pdm.ewm(span=14, adjust=False).mean() / atr)
    mdi = 100 * (mdm.ewm(span=14, adjust=False).mean() / atr)
    df['adx'] = (100 * (abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan))).ewm(span=14, adjust=False).mean()

    # 4. RS (Sức mạnh so với VN-Index)
    vni_c = pd.to_numeric(vni_df['close'] if 'close' in vni_df.columns else vni_df['Close'], errors='coerce')
    df['rs'] = round(((c/c.shift(5)) - (vni_c.iloc[-1]/vni_c.iloc[-5])) * 100, 2)
    
    # 5. Tín hiệu 💣 & 🏹
    df['is_bomb'] = df['bb_width'] <= df['bb_width'].rolling(30).min()
    df['is_buy'] = (c > df['ma20']) & (df['ma20'] >= df['ma50'] * 0.99) & (v > v.rolling(20).mean() * 1.3)
    
    df['target_1'] = round(c * 1.12, 0)
    df['target_2'] = round(c * 1.25, 0)
    df['stop_loss'] = round(df['ma20'] * 0.96, 0)
    return df

# --- SIDEBAR & DATA ENGINE (Xử lý lỗi Multi-Index) ---
with st.sidebar:
    st.header("🚀 FIREANT PRO V33.3")
    if st.button("🔄 CẬP NHẬT DỮ LIỆU"):
        with st.spinner("Đang tải dữ liệu..."):
            # Tải VNI và làm phẳng cột
            vni = yf.download("^VNINDEX", period="2y")
            vni.columns = [col[0] if isinstance(col, tuple) else col for col in vni.columns]
            vni = vni.reset_index()
            vni.to_csv("VNINDEX_local.csv", index=False)
            
            mã_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','DIG','PDR','VHM','FPT','DGC']
            all_h = []
            for m in mã_list:
                t = yf.download(f"{m}.VN", period="2y", progress=False)
                t.columns = [col[0] if isinstance(col, tuple) else col for col in t.columns]
                t = t.reset_index()
                t['symbol'] = m
                all_h.append(t)
            pd.concat(all_h).to_csv("hose_local.csv", index=False)
            st.success("✅ Đã xong! Hãy chọn mã bên dưới.")
            st.rerun()

    mode = st.radio("CHỨC NĂNG:", ["📊 DÒNG TIỀN NGÀNH", "🌟 BỘ LỌC SIÊU SAO", "📈 SOI CHART CHI TIẾT"])
    ticker = st.text_input("NHẬP MÃ:", "MWG").upper()

# --- HIỂN THỊ ---
if not os.path.exists("VNINDEX_local.csv"):
    st.warning("⚠️ Vui lòng nhấn nút 'CẬP NHẬT DỮ LIỆU' để bắt đầu.")
else:
    vni_df = pd.read_csv("VNINDEX_local.csv")
    hose_df = pd.read_csv("hose_local.csv")

    if mode == "📈 SOI CHART CHI TIẾT":
        df_c = calculate_pro_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_c is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # Chart nến, MA, 💣, 🏹
            fig.add_trace(go.Candlestick(x=df_c['Date'], open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
            
            # Tín hiệu
            b = df_c[df_c['is_bomb']]
            fig.add_trace(go.Scatter(x=b['Date'], y=b['high']*1.03, mode='text', text="💣", textfont=dict(size=20), name="Bomb"), row=1, col=1)
            s = df_c[df_c['is_buy']]
            fig.add_trace(go.Scatter(x=s['Date'], y=s['low']*0.97, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # Volume, RSI/RS, ADX
            colors = ['red' if r['open'] > r['close'] else 'green' for i, r in df_c.iterrows()]
            fig.add_trace(go.Bar(x=df_c['Date'], y=df_c['volume'], marker_color=colors, name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['rs'], line=dict(color='magenta'), name="RS"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['adx'], line=dict(color='white'), name="ADX"), row=4, col=1)

            # GIAO DIỆN KÉO DÃN FIREANT
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan',
                              yaxis=dict(side='right', fixedrange=False, autorange=True),
                              yaxis2=dict(side='right', fixedrange=False, autorange=True),
                              yaxis3=dict(side='right', fixedrange=False, autorange=True),
                              yaxis4=dict(side='right', fixedrange=False, autorange=True),
                              xaxis=dict(fixedrange=False))
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            l = df_c.iloc[-1]
            st.success(f"🎯 Target 1: {int(l['target_1'])} | Target 2: {int(l['target_2'])} | 🛑 Stop: {int(l['stop_loss'])}")
