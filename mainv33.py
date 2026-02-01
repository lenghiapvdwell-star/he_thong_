import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# --- CẤU HÌNH GITHUB ---
GITHUB_USER = "lenghiapvdwell-star"
REPO_NAME = "he_thong_"

st.set_page_config(page_title="V33 - SIÊU HỆ THỐNG DÒNG TIỀN", layout="wide")

# --- HÀM TÍNH TOÁN KỸ THUẬT V33 CHUẨN ---
def calculate_pro_signals(df, vni_df):
    if df is None or len(df) < 100: return None
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=['close', 'volume']).reset_index(drop=True)

    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    # 1. Đường trung bình & Bollinger
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

    # 3. RS (Sức mạnh tương quan)
    vni_c = vni_df['close'] if 'close' in vni_df.columns else vni_df['Close']
    df['rs'] = round(((c/c.shift(5)) - (vni_c.iloc[-1]/vni_c.iloc[-5])) * 100, 2)
    
    # 4. LOGIC DÒNG TIỀN & NỀN GIÁ
    df['base_volatility'] = (c.rolling(120).max() - c.rolling(120).min()) / c.rolling(120).mean()
    df['is_flat_base'] = df['base_volatility'] < 0.25 # Nền phẳng 6 tháng
    df['is_bomb'] = df['bb_width'] <= df['bb_width'].rolling(30).min()
    df['vol_trend'] = v.rolling(5).mean() > v.shift(5).rolling(5).mean()
    
    # ĐIỂM MUA CHUẨN: Tiền vào + MA20 >= MA50 + Vol đột biến
    df['is_buy'] = (c > df['ma20']) & (df['ma20'] >= df['ma50'] * 0.99) & \
                   (v > v.rolling(20).mean() * 1.3) & (df['rsi'].between(45, 78))
    
    # Target & Stoploss
    df['target_1'] = round(c * 1.12, 0)
    df['target_2'] = round(c * 1.25, 0)
    df['stop_loss'] = round(df['ma50'] * 0.97, 0)
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚡ HỆ THỐNG V33")
    if st.button("🔄 CẬP NHẬT REALTIME"):
        with st.spinner("Đang đồng bộ dữ liệu..."):
            v_df = yf.download("^VNINDEX", period="2y").reset_index()
            v_df.to_csv("VNINDEX.csv", index=False)
            # Quét các mã tiêu biểu
            list_m = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','DIG','PDR','VHM','FPT','DGC','SHB','VNM']
            all_data = []
            for m in list_m:
                t = yf.download(f"{m}.VN", period="2y", progress=False).reset_index()
                t['symbol'] = m
                all_data.append(t)
            pd.concat(all_data).to_csv("hose.csv", index=False)
            st.success("Đã đồng bộ thành công!")

    st.divider()
    mode = st.radio("CHỨC NĂNG:", ["📊 DÒNG TIỀN NGÀNH", "🌟 SIÊU SAO THEO DÕI", "📈 SOI CHI TIẾT MÃ"])
    ticker_input = st.text_input("MÃ SOI:", "MWG").upper()

# --- HIỂN THỊ CHÍNH ---
try:
    vni_df = pd.read_csv("https://raw.githubusercontent.com/" + GITHUB_USER + "/" + REPO_NAME + "/main/VNINDEX.csv")
    hose_df = pd.read_csv("https://raw.githubusercontent.com/" + GITHUB_USER + "/" + REPO_NAME + "/main/hose.csv")

    if mode == "📊 DÒNG TIỀN NGÀNH":
        st.header("🌊 ĐIỂM DÒNG TIỀN NHÓM NGÀNH (Thang điểm 10)")
        sectors = {
            "BÁN LẺ (MWG, FRT, DGW, MSN)": ['MWG', 'FRT', 'DGW', 'MSN'],
            "CHỨNG KHOÁN (SSI, VND, VCI)": ['SSI', 'VND', 'VCI'],
            "THÉP (HPG, NKG, HSG)": ['HPG', 'NKG', 'HSG']
        }
        res = []
        for name, tickers in sectors.items():
            sc = []
            for t in tickers:
                d = calculate_pro_signals(hose_df[hose_df['symbol']==t].copy(), vni_df)
                if d is not None:
                    l = d.iloc[-1]
                    s = (4 if l['is_buy'] else 0) + (3 if l['vol_trend'] else 0) + (3 if l['rs'] > 0 else 0)
                    sc.append(s)
            res.append({"Ngành": name, "Điểm": round(np.mean(sc),1), "Trạng thái": "🔥 DẪN DẮT" if np.mean(sc) > 6 else "Tích lũy"})
        st.table(pd.DataFrame(res).sort_values("Điểm", ascending=False))

    elif mode == "🌟 SIÊU SAO THEO DÕI":
        st.subheader("🚀 BỘ LỌC SIÊU CỔ PHIẾU V33")
        v_list = []
        for s in hose_df['symbol'].unique():
            d = calculate_pro_signals(hose_df[hose_df['symbol']==s].copy(), vni_df)
            if d is not None:
                l = d.iloc[-1]
                if l['is_buy'] or l['is_bomb']:
                    v_list.append({"Mã": s, "Giá": int(l['close']), "Nền 6T": "✅" if l['is_flat_base'] else "❌", "RS": l['rs'], "Tín hiệu": "🏹 MUA" if l['is_buy'] else "💣 BÓ CHẶT"})
        st.dataframe(pd.DataFrame(v_list), use_container_width=True)

    elif mode == "📈 SOI CHI TIẾT MÃ":
        df_c = calculate_pro_signals(hose_df[hose_df['symbol'] == ticker_input].copy(), vni_df)
        if df_c is not None:
            # Layout đa tầng mượt mà
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # Tầng 1: Nến & MA
            fig.add_trace(go.Candlestick(x=df_c['date'], open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name=ticker_input), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
            
            # Quả bom 💣 & Mũi tên Mua 🏹
            b_pts = df_c[df_c['is_bomb']]
            fig.add_trace(go.Scatter(x=b_pts['date'], y=b_pts['high']*1.02, mode='text', text="💣", textfont=dict(size=22), name="Bomb"), row=1, col=1)
            buy_pts = df_c[df_c['is_buy']]
            fig.add_trace(go.Scatter(x=buy_pts['date'], y=buy_pts['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=df_c['date'], y=df_c['volume'], name="Volume"), row=2, col=1)

            # Tầng 3: RS & RSI
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['rs'], name="RS", line=dict(color='magenta')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)

            # Tầng 4: ADX
            fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['adx'], name="ADX", line=dict(color='white')), row=4, col=1)

            # --- CẤU HÌNH KÉO DÃN FIREANT ---
            fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan', hovermode='x unified',
                              yaxis=dict(side='right', fixedrange=False, autorange=True),
                              yaxis2=dict(side='right', fixedrange=False, autorange=True),
                              yaxis3=dict(side='right', fixedrange=False, autorange=True),
                              yaxis4=dict(side='right', fixedrange=False, autorange=True),
                              xaxis=dict(fixedrange=False))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # Khối Khuyến nghị
            l = df_c.iloc[-1]
            st.info(f"🎯 Target 1: {l['target_1']:,} | Target 2: {l['target_2']:,} | 🛑 Stoploss: {l['stop_loss']:,}")

except Exception as e:
    st.error(f"Vui lòng nhấn 'CẬP NHẬT REALTIME' ở Sidebar để tạo dữ liệu. Lỗi: {e}")
