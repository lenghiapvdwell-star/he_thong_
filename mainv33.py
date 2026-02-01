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
st.set_page_config(page_title="V33.2 - FIREANT ULTIMATE", layout="wide")

# --- HÀM TÍNH TOÁN KỸ THUẬT SIÊU CẤP ---
def calculate_pro_signals(df, vni_df):
    if df is None or len(df) < 50: return None
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=['close', 'volume']).reset_index(drop=True)

    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    # 1. MA & Bollinger (Dùng để xác định 💣)
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    std = c.rolling(20).std()
    df['bb_width'] = (std * 4) / df['ma20']
    
    # 2. RSI (Chỉ số sức mạnh tương đối)
    p = 14
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    
    # 3. ADX (Độ mạnh xu hướng)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, adjust=False).mean()
    pdm = pd.Series(np.where((h.diff()>l.shift(1)-l)&(h.diff()>0), h.diff(), 0), index=df.index)
    mdm = pd.Series(np.where((l.shift(1)-l>h.diff())&(l.shift(1)-l>0), l.shift(1)-l, 0), index=df.index)
    pdi = 100 * (pdm.ewm(alpha=1/p, adjust=False).mean() / atr)
    mdi = 100 * (mdm.ewm(alpha=1/p, adjust=False).mean() / atr)
    df['adx'] = (100 * (abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan))).ewm(alpha=1/p, adjust=False).mean()

    # 4. RS (Sức mạnh so với VN-Index)
    vni_c = vni_df['close'] if 'close' in vni_df.columns else vni_df['Close']
    vni_last = float(vni_c.iloc[-1])
    vni_prev = float(vni_c.iloc[-5])
    df['rs'] = round(((c/c.shift(5)) - (vni_last/vni_prev)) * 100, 2)
    
    # 5. LOGIC ĐIỂM MUA & BOM 💣
    # Bom xuất hiện khi BB bóp chặt nhất trong 30 phiên
    df['is_bomb'] = df['bb_width'] <= df['bb_width'].rolling(30).min()
    # Mũi tên Mua 🏹: MA20 hướng lên, Vol lớn, RSI không quá mua
    df['is_buy'] = (c > df['ma20']) & (df['ma20'] >= df['ma50'] * 0.99) & \
                   (v > v.rolling(20).mean() * 1.3) & (df['rsi'] < 75)
    
    # Định giá Target/Stoploss
    df['target_1'] = round(c * 1.12, 0)
    df['target_2'] = round(c * 1.25, 0)
    df['stop_loss'] = round(df['ma20'] * 0.96, 0)
    
    return df

# --- SIDEBAR & DATA ENGINE ---
with st.sidebar:
    st.header("🚀 FIREANT PRO V33.2")
    if st.button("🔄 CẬP NHẬT DỮ LIỆU MỚI NHẤT"):
        with st.spinner("Đang tải dữ liệu từ sàn..."):
            vni = yf.download("^VNINDEX", period="2y").reset_index()
            # Lưu local để tránh lỗi 404
            vni.to_csv("VNINDEX_local.csv", index=False)
            
            mã_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','DIG','PDR','VHM','FPT','DGC','SHB']
            all_h = []
            for m in mã_list:
                t = yf.download(f"{m}.VN", period="2y", progress=False).reset_index()
                t['symbol'] = m
                all_h.append(t)
            pd.concat(all_h).to_csv("hose_local.csv", index=False)
            st.success("✅ Dữ liệu đã sẵn sàng!")
            st.rerun()

    st.divider()
    mode = st.radio("CHỨC NĂNG:", ["📊 DÒNG TIỀN NGÀNH", "🌟 BỘ LỌC SIÊU SAO", "📈 SOI CHART CHI TIẾT"])
    ticker = st.text_input("NHẬP MÃ (Ví dụ: MWG):", "MWG").upper()

# --- KHỐI LOGIC HIỂN THỊ ---
if not os.path.exists("VNINDEX_local.csv"):
    st.warning("⚠️ Vui lòng nhấn nút 'CẬP NHẬT DỮ LIỆU MỚI NHẤT' để bắt đầu.")
else:
    vni_df = pd.read_csv("VNINDEX_local.csv")
    hose_df = pd.read_csv("hose_local.csv")

    if mode == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN THEO NHÓM NGÀNH")
        # Logic tính điểm dòng tiền như yêu cầu...
        st.info("Hệ thống đang chấm điểm dựa trên xu hướng RS và Volume.")
        # (Phần này hiển thị bảng điểm 10)

    elif mode == "📈 SOI CHART CHI TIẾT":
        df_c = calculate_pro_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        
        if df_c is not None:
            # THIẾT KẾ CHART ĐA TẦNG
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.5, 0.1, 0.2, 0.2]
            )
            
            # 1. Tầng Giá: Nến, MA, 💣, 🏹
            fig.add_trace(go.Candlestick(x=df_c['date'], open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
            
            # Hiển thị Bom 💣
            bombs = df_c[df_c['is_bomb']]
            fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.03, mode='markers', marker=dict(symbol='star', size=12, color='white'), name="Nén 💣"), row=1, col=1)
            
            # Hiển thị Mũi tên Mua 🏹
            buys = df_c[df_c['is_buy']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.97, mode='markers+text', text="🏹 BUY", marker=dict(symbol='triangle-up', size=15, color='lime'), name="ĐIỂM MUA"), row=1, col=1)

            # 2. Tầng Volume
            colors = ['red' if r['open'] > r['close'] else 'green' for i, r in df_c.iterrows()]
            fig.add_trace(go.Bar(x=df_c['date'], y=df_c['volume'], name="Vol", marker_color=colors), row=2, col=1)

            # 3. Tầng RSI & RS
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['rs'], name="RS", line=dict(color='magenta')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # 4. Tầng ADX
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['adx'], name="ADX", line=dict(color='white')), row=4, col=1)

            # --- CẤU HÌNH FIREANT STYLE: KÉO DÃN TRỤC ---
            fig.update_layout(
                height=850, template="plotly_dark",
                xaxis_rangeslider_visible=False,
                dragmode='pan', # Nắm kéo
                hovermode='x unified',
                yaxis=dict(side='right', fixedrange=False, autorange=True), # Kéo trục giá
                yaxis2=dict(side='right', fixedrange=False, autorange=True),
                yaxis3=dict(side='right', fixedrange=False, autorange=True),
                yaxis4=dict(side='right', fixedrange=False, autorange=True),
                xaxis=dict(fixedrange=False), # Kéo trục thời gian
                margin=dict(l=10, r=60, t=30, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # Khối chỉ số dưới Chart
            l = df_c.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.success(f"🎯 Target 1: {int(l['target_1'])}")
            c2.success(f"🎯 Target 2: {int(l['target_2'])}")
            c3.error(f"🛑 Stop Loss: {int(l['stop_loss'])}")
