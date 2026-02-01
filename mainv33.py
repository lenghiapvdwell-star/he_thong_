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
GITHUB_USER = "lenghiapvdwell-star"
REPO_NAME = "he_thong_"

st.set_page_config(page_title="V33.1 - MONEY FLOW PRO", layout="wide")

# --- HÀM TÍNH TOÁN KỸ THUẬT SIÊU CẤP ---
def calculate_pro_signals(df, vni_df):
    if df is None or len(df) < 50: return None
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=['close', 'volume']).reset_index(drop=True)

    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    # 1. MA & Bollinger
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    std = c.rolling(20).std()
    df['bb_width'] = (std * 4) / df['ma20']
    
    # 2. RSI & ADX
    p = 14
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, adjust=False).mean()
    pdm = pd.Series(np.where((h.diff()>l.shift(1)-l)&(h.diff()>0), h.diff(), 0), index=df.index)
    mdm = pd.Series(np.where((l.shift(1)-l>h.diff())&(l.shift(1)-l>0), l.shift(1)-l, 0), index=df.index)
    pdi = 100 * (pdm.ewm(alpha=1/p, adjust=False).mean() / atr)
    mdi = 100 * (mdm.ewm(alpha=1/p, adjust=False).mean() / atr)
    df['adx'] = (100 * (abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan))).ewm(alpha=1/p, adjust=False).mean()

    # 3. RS (Sức mạnh giá)
    vni_c = vni_df['close'] if 'close' in vni_df.columns else vni_df['Close']
    # Tính toán RS linh hoạt
    df['rs'] = round(((c/c.shift(5)) - (float(vni_c.iloc[-1])/float(vni_c.iloc[-5]))) * 100, 2)
    
    # 4. Tín hiệu Bom & Mua
    df['is_bomb'] = df['bb_width'] <= df['bb_width'].rolling(30).min()
    df['is_buy'] = (c > df['ma20']) & (df['ma20'] >= df['ma50'] * 0.98) & \
                   (v > v.rolling(20).mean() * 1.3) & (df['rsi'] < 78)
    
    df['target_1'] = round(c * 1.12, 0)
    df['target_2'] = round(c * 1.25, 0)
    df['stop_loss'] = round(df['ma20'] * 0.95, 0)
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("💎 V33.1 PRO")
    if st.button("🔄 CẬP NHẬT REALTIME (BẮT BUỘC)"):
        with st.spinner("Đang tải dữ liệu trực tiếp..."):
            vni = yf.download("^VNINDEX", period="2y").reset_index()
            vni.to_csv("VNINDEX_temp.csv", index=False)
            mã = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','DIG','PDR','VHM','FPT','DGC','SHB']
            all_h = []
            for m in mã:
                t = yf.download(f"{m}.VN", period="2y", progress=False).reset_index()
                t['symbol'] = m
                all_h.append(t)
            pd.concat(all_h).to_csv("hose_temp.csv", index=False)
            st.success("✅ Đã tạo dữ liệu tạm thời!")

    st.divider()
    mode = st.radio("CHỨC NĂNG:", ["📊 DÒNG TIỀN NGÀNH", "🌟 SIÊU SAO THEO DÕI", "📈 SOI CHI TIẾT MÃ"])
    ticker_input = st.text_input("MÃ SOI:", "MWG").upper()

# --- HIỂN THỊ ---
try:
    # Ưu tiên đọc file vừa tải về để tránh lỗi 404 GitHub
    if os.path.exists("VNINDEX_temp.csv"):
        vni_df = pd.read_csv("VNINDEX_temp.csv")
        hose_df = pd.read_csv("hose_temp.csv")
    else:
        vni_df = pd.read_csv(f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/VNINDEX.csv")
        hose_df = pd.read_csv(f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/hose.csv")

    if mode == "📊 DÒNG TIỀN NGÀNH":
        # (Giữ nguyên logic chấm điểm ngành của bạn)
        st.subheader("🌊 BẢNG CHẤM ĐIỂM DÒNG TIỀN")
        # ...

    elif mode == "📈 SOI CHI TIẾT MÃ":
        df_c = calculate_pro_signals(hose_df[hose_df['symbol'] == ticker_input].copy(), vni_df)
        if df_c is not None:
            # TẠO ĐỒ THỊ 4 TẦNG RIÊNG BIỆT
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, 
                vertical_spacing=0.02, 
                row_heights=[0.5, 0.1, 0.2, 0.2]
            )
            
            # TẦNG 1: NẾN, MA, BOM, MŨI TÊN
            fig.add_trace(go.Candlestick(x=df_c['date'], open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name="Nến"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
            
            # 💣 Quả bom (Squeeze)
            b_pts = df_c[df_c['is_bomb']]
            fig.add_trace(go.Scatter(x=b_pts['date'], y=b_pts['high']*1.03, mode='text', text="💣", textfont=dict(size=22), name="Bom"), row=1, col=1)
            
            # 🏹 Mũi tên Mua
            buy_pts = df_c[df_c['is_buy']]
            fig.add_trace(go.Scatter(x=buy_pts['date'], y=buy_pts['low']*0.97, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="ĐIỂM MUA"), row=1, col=1)

            # TẦNG 2: VOLUME
            colors = ['red' if row['open'] > row['close'] else 'green' for i, row in df_c.iterrows()]
            fig.add_trace(go.Bar(x=df_c['date'], y=df_c['volume'], name="Vol", marker_color=colors), row=2, col=1)

            # TẦNG 3: RS & RSI
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['rs'], name="RS (Tím)", line=dict(color='magenta', width=2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['rsi'], name="RSI (Cam)", line=dict(color='orange', width=2)), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # TẦNG 4: ADX
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['adx'], name="ADX", line=dict(color='white')), row=4, col=1)

            # --- CẤU HÌNH KÉO DÃN FIREANT CHUẨN ---
            fig.update_layout(
                height=900, template="plotly_dark",
                xaxis_rangeslider_visible=False,
                dragmode='pan', hovermode='x unified',
                yaxis=dict(side='right', fixedrange=False, autorange=True), # Kéo dãn trục Y
                yaxis2=dict(side='right', fixedrange=False, autorange=True),
                yaxis3=dict(side='right', fixedrange=False, autorange=True),
                yaxis4=dict(side='right', fixedrange=False, autorange=True),
                xaxis=dict(fixedrange=False) # Kéo dãn trục X
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # Target & Stoploss
            l = df_c.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("🎯 TARGET 1", f"{int(l['target_1'])}")
            c2.metric("🎯 TARGET 2", f"{int(l['target_2'])}")
            c3.metric("🛑 STOP LOSS", f"{int(l['stop_loss'])}")

except Exception as e:
    st.warning("⚠️ Vui lòng nhấn nút 'CẬP NHẬT REALTIME' bên trái để khởi tạo dữ liệu.")
    st.error(f"Chi tiết lỗi: {e}")
