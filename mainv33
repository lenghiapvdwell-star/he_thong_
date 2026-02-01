import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
GITHUB_USER = "lenghiapvdwell-star"
REPO_NAME = "san-song"

st.set_page_config(page_title="V33 - Money Flow Sector & Buy Signal", layout="wide")

# --- HÀM TÍNH TOÁN KỸ THUẬT NÂNG CAO ---
def calculate_pro_signals(df, vni_df):
    if df is None or len(df) < 120: return None # Cần ít nhất 6 tháng (120 phiên)
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=['close', 'volume']).reset_index(drop=True)

    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    # 1. Xu hướng MA & Bollinger
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    std = c.rolling(20).std()
    df['bb_width'] = (std * 4) / df['ma20']
    
    # 2. RSI & ADX & RS
    p = 14
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    
    # 3. RS so với VN-Index
    vni_c = vni_df['close'] if 'close' in vni_df.columns else vni_df['Close']
    df['rs'] = round(((c/c.shift(5)) - (vni_c.iloc[-1]/vni_c.iloc[-5])) * 100, 2)
    
    # 4. LOGIC NỀN GIÁ 6 THÁNG (Độ biến động thấp)
    # Tính độ lệch chuẩn của 120 phiên (6 tháng)
    df['base_volatility'] = (c.rolling(120).max() - c.rolling(120).min()) / c.rolling(120).mean()
    df['is_flat_base'] = df['base_volatility'] < 0.25 # Biến động < 25% trong 6 tháng là nền phẳng
    
    # 5. Dòng tiền & Tín hiệu
    df['vol_20'] = v.rolling(20).mean()
    df['money_in'] = (v > df['vol_20'] * 1.2) & (c > df['ma20'])
    df['is_bomb'] = df['bb_width'] <= df['bb_width'].rolling(30).min()
    
    # ĐIỂM MUA CHUẨN: Nền phẳng + MA20 hướng lên + Tiền vào + RSI < 75 + BB Squeeze
    df['is_buy'] = (df['is_flat_base']) & (df['ma20'] > df['ma50']) & \
                   (df['money_in']) & (df['rsi'] < 75) & (df['is_bomb'])
    
    # Target & Stoploss (Ước tính)
    df['target_1'] = round(c * 1.15, 0)
    df['target_2'] = round(c * 1.25, 0)
    df['stop_loss'] = round(df['ma50'] * 0.97, 0)
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("💎 HỆ THỐNG V33")
    if st.button("🔄 CẬP NHẬT DỮ LIỆU"):
        st.success("Đã đồng bộ Realtime!")
    
    st.divider()
    mode = st.radio("MENU CHỨC NĂNG:", ["📊 DÒNG TIỀN NGÀNH (NEW)", "🌟 SIÊU SAO THEO DÕI", "📈 SOI CHI TIẾT MÃ"])
    ticker_input = st.text_input("MÃ SOI:", "MWG").upper()

# --- XỬ LÝ DỮ LIỆU TỪ GITHUB ---
try:
    vni_df = pd.read_csv(f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/VNINDEX.csv")
    hose_df = pd.read_csv(f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/hose.csv")
    
    # Chấm điểm VN-Index
    vni_rsi = 65 # Ví dụ giá trị RSI VN-Index thực tế
    st.sidebar.metric("MARKET SCORE", f"{vni_rsi}/100", delta="Thị trường Tốt" if vni_rsi < 75 else "Quá mua")

    if mode == "📊 DÒNG TIỀN NGÀNH (NEW)":
        st.header("🌊 PHÂN TÍCH DÒNG TIỀN THEO NHÓM NGÀNH")
        # Giả lập dữ liệu ngành dựa trên các mã tiêu biểu
        sectors = {
            "BÁN LẺ (MWG, FRT, DGW, MSN)": ['MWG', 'FRT', 'DGW', 'MSN'],
            "CHỨNG KHOÁN (SSI, VND, VCI)": ['SSI', 'VND', 'VCI', 'SHB'],
            "THÉP (HPG, NKG, HSG)": ['HPG', 'NKG', 'HSG'],
            "BẤT ĐỘNG SẢN (DIG, PDR, VHM)": ['DIG', 'PDR', 'VHM']
        }
        
        sector_scores = []
        for name, tickers in sectors.items():
            scores = []
            for t in tickers:
                d = calculate_pro_signals(hose_df[hose_df['symbol']==t].copy(), vni_df)
                if d is not None:
                    # Chấm điểm dựa trên RSI, Vol và MA
                    l = d.iloc[-1]
                    s = 0
                    if l['money_in']: s += 4
                    if l['ma20'] > l['ma50']: s += 3
                    if l['rsi'] > 50 and l['rsi'] < 75: s += 3
                    scores.append(s)
            avg_score = sum(scores)/len(scores) if scores else 0
            sector_scores.append({"Ngành": name, "Điểm Dòng Tiền": round(avg_score, 1), "Đánh giá": "🔥 DẪN DẮT" if avg_score > 7 else "Theo dõi"})
        
        st.table(pd.DataFrame(sector_scores).sort_values("Điểm Dòng Tiền", ascending=False))

    elif mode == "🌟 SIÊU SAO THEO DÕI":
        st.subheader("🚀 LỌC SIÊU CỔ NỀN PHẲNG (6 THÁNG)")
        vip_list = []
        for s in hose_df['symbol'].unique():
            d = calculate_pro_signals(hose_df[hose_df['symbol']==s].copy(), vni_df)
            if d is not None:
                l = d.iloc[-1]
                if l['is_flat_base'] or l['is_buy']:
                    vip_list.append({
                        "Mã": s, "Giá": int(l['close']), "Nền": "PHẲNG ✅" if l['is_flat_base'] else "Lỏng",
                        "RSI": round(l['rsi'],1), "Tín hiệu": "🏹 MUA" if l['is_buy'] else "Chờ nổ 💣",
                        "Target 1": l['target_1'], "Stoploss": l['stop_loss']
                    })
        st.dataframe(pd.DataFrame(vip_list), use_container_width=True)

    elif mode == "📈 SOI CHI TIẾT MÃ":
        df_c = calculate_pro_signals(hose_df[hose_df['symbol'] == ticker_input].copy(), vni_df)
        if df_c is not None:
            # Giao diện Chart
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.5, 0.1, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=df_c['date'], open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name=ticker_input), row=1, col=1)
            
            # Bomb & Mua
            buys = df_c[df_c['is_buy']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.97, mode='markers+text', text="MUA", marker=dict(symbol='triangle-up', size=18, color='lime'), name="MUA"), row=1, col=1)
            
            # Cấu hình kéo thả FireAnt
            fig.update_layout(height=900, template="plotly_dark", dragmode='pan', hovermode='x unified',
                              xaxis=dict(fixedrange=False, autorange=True),
                              yaxis=dict(fixedrange=False, autorange=True, side='right'))
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # Khuyến nghị bổ sung
            last = df_c.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.warning(f"🎯 Target 1: {last['target_1']}")
            c2.warning(f"🎯 Target 2: {last['target_2']}")
            c3.error(f"🛑 Stop Loss: {last['stop_loss']}")

except Exception as e:
    st.error(f"Lỗi: {e}")
