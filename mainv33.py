import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V43 - SMART MONEY PRO", layout="wide")

def load_data(file_name):
    if not os.path.exists(file_name): return None
    df = pd.read_csv(file_name)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ['symbol', 'ticker', 'mã']:
        if col in df.columns: df = df.rename(columns={col: 'symbol'}); break
    for col in ['date', 'ngày']:
        if col in df.columns: df = df.rename(columns={col: 'date'}); break
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
    return df.sort_values('date')

# --- TÍNH TOÁN CHỈ BÁO CHI TIẾT ---
def calculate_pro_signals(df, vni_df=None):
    if df is None or len(df) < 20: return None
    df = df.copy().sort_values('date')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close'])

    # 1. Đường trung bình MA
    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(50, min_periods=1).mean()

    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))

    # 3. ADX (Sức mạnh xu hướng)
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = df['low'].diff(-1).clip(lower=0)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift()), 
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    df['adx'] = (abs(plus_dm - minus_dm) / (plus_dm + minus_dm).replace(0, 1) * 100).rolling(14, min_periods=1).mean()

    # 4. RS (Sức mạnh tương quan vs VNI)
    if vni_df is not None:
        vni_df = vni_df.sort_values('date')
        combined = pd.merge(df[['date', 'close']], vni_df[['date', 'close']], on='date', suffixes=('', '_vni'))
        if not combined.empty:
            rs_val = (combined['close'] / combined['close'].shift(20)) / (combined['close_vni'] / combined['close_vni'].shift(20))
            df = pd.merge(df, pd.DataFrame({'date': combined['date'], 'rs': rs_val}), on='date', how='left')
    if 'rs' not in df.columns: df['rs'] = 0

    # 5. ĐIỂM MUA & BOOM TIỀN
    df['vol20'] = df['volume'].rolling(20, min_periods=1).mean()
    # Mũi tên xanh: Giá > MA20 & Vol > 1.2 lần trung bình
    df['buy_signal'] = (df['close'] > df['ma20']) & (df['volume'] > df['vol20'] * 1.2)
    # Quả bom tiền: Vol > 2.0 lần trung bình + Giá tăng > 3%
    df['money_bomb'] = (df['volume'] > df['vol20'] * 2.0) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- GIAO DIỆN ---
hose_df = load_data("hose.csv")
vni_df = load_data("vnindex.csv")

with st.sidebar:
    st.header("🏆 SUPREME V43")
    ticker = st.text_input("🔍 MÃ SOI:", "HPG").upper()
    if st.button("📈 SỨC KHỎE VN-INDEX"):
        if vni_df is not None:
            v_res = calculate_pro_signals(vni_df)
            st.metric("VNI ADX", round(v_res['adx'].iloc[-1], 1))
            st.metric("VNI RSI", round(v_res['rsi'].iloc[-1], 1))
    menu = st.radio("MENU:", ["📈 ĐỒ THỊ CHI TIẾT", "📊 DÒNG TIỀN NGÀNH", "🎯 LỌC ĐIỂM MUA"])

if hose_df is not None:
    if menu == "📈 ĐỒ THỊ CHI TIẾT":
        data = calculate_pro_signals(hose_df[hose_df['symbol'] == ticker], vni_df)
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.15, 0.15, 0.2])
            
            # Tầng 1: Candle + MA + Signals
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
            
            # Mũi tên mua (⬆️)
            buys = data[data['buy_signal']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.98, mode='markers+text', text="⬆️", textposition="bottom center", marker=dict(size=12, color='lime'), name="Điểm mua"), row=1, col=1)
            
            # Quả bom tiền (💣)
            bombs = data[data['money_bomb']]
            fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.02, mode='markers+text', text="💣", textposition="top center", marker=dict(size=15, color='red'), name="BOM TIỀN"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol", marker_color='dodgerblue'), row=2, col=1)
            
            # Tầng 3: RSI & RS
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta'), name="RS (x50)"), row=3, col=1)
            
            # Tầng 4: ADX
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX"), row=4, col=1)

            fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        else:
            st.error("Dữ liệu không đủ hoặc lỗi mã.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        # Tự động quét dòng tiền theo danh mục ngành
        nganh_dict = {"THÉP":['HPG','NKG','HSG'], "BANK":['VCB','TCB','MBB'], "CHỨNG":['SSI','VND','VCI'], "BĐS":['DIG','PDR','VHM'], "BÁN LẺ":['MWG','FRT','MSN']}
        res = []
        for n, mãs in nganh_dict.items():
            scores = []
            for m in mãs:
                d = calculate_pro_signals(hose_df[hose_df['symbol'] == m], vni_df)
                if d is not None:
                    s = 0
                    last = d.iloc[-1]
                    if last['buy_signal']: s += 5
                    if last['money_bomb']: s += 5
                    scores.append(s)
            res.append({"Ngành": n, "Sức Mạnh Dòng Tiền": np.mean(scores) if scores else 0})
        st.table(pd.DataFrame(res).sort_values("Sức Mạnh Dòng Tiền", ascending=False))

    elif menu == "🎯 LỌC ĐIỂM MUA":
        st.subheader("🚀 QUÉT SIÊU ĐIỂM MUA & BOM TIỀN")
        found = []
        for s in hose_df['symbol'].unique():
            d = calculate_pro_signals(hose_df[hose_df['symbol'] == s], vni_df)
            if d is not None:
                last = d.iloc[-1]
                if last['money_bomb'] or last['buy_signal']:
                    found.append({"Mã": s, "Tín hiệu": "💣 BOM TIỀN" if last['money_bomb'] else "⬆️ ĐIỂM MUA", "RSI": round(last['rsi'],1), "Giá": last['close']})
        st.dataframe(pd.DataFrame(found), use_container_width=True)
else:
    st.error("❌ Không tìm thấy hose.csv")
