import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="V50 - SMART MONEY TERMINAL", layout="wide")

# --- 1. ENGINE XỬ LÝ DỮ LIỆU ---
def master_cleaner(df):
    if df is None or df.empty: return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Tìm cột ngày
    for c in ['date', 'datetime', 'ngày', 'index']:
        if c in df.columns:
            df = df.rename(columns={c: 'date'})
            break
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['close', 'date']).sort_values('date')

# --- 2. HÀM TÍNH TOÁN FULL CHỈ BÁO ---
def calculate_supreme(df, vni_df=None):
    df = master_cleaner(df)
    if df is None or len(df) < 30: return None
    
    # MA20, MA50
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX chuẩn (DMI)
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = df['low'].diff(-1).clip(lower=0)
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    df['adx'] = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)).rolling(14).mean()

    # RS (Relative Strength vs VNINDEX)
    if vni_df is not None:
        vni_c = master_cleaner(vni_df)[['date', 'close']].rename(columns={'close': 'v_c'})
        df = pd.merge(df, vni_c, on='date', how='left').ffill()
        df['rs'] = (df['close']/df['close'].shift(20)) / (df['v_c']/df['v_c'].shift(20))
    else: df['rs'] = 1.0

    # Tín hiệu Mua & Bom tiền
    v20 = df['volume'].rolling(20).mean()
    df['buy_arrow'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['money_bomb'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    return df

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🏆 SMART MONEY V50")
    if st.button("🔄 UPDATE DATA REAL-TIME", use_container_width=True):
        with st.spinner("Đang ghi đè dữ liệu..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vnindex.csv")
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','STB','NKG','HSG','PDR','GEX','DGW','FRT','VCI','VIX']
            all_l = []
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y")
                tmp['symbol'] = t
                all_l.append(tmp)
            pd.concat(all_l).to_csv("hose.csv")
            st.success("ĐÃ CẬP NHẬT XONG!")
            st.rerun()

    st.divider()
    ticker = st.text_input("🔍 SOI MÃ (HPG, SSI...):", "HPG").upper()
    
    # NÚT SỨC KHỎE VN-INDEX (CỐ ĐỊNH)
    st.subheader("📊 CHIẾN THUẬT VNI")
    if os.path.exists("vnindex.csv"):
        v_data = calculate_supreme(pd.read_csv("vnindex.csv"))
        if v_data is not None:
            l = v_data.iloc[-1]
            score = 0
            if l['close'] > l['ma20']: score += 3
            if l['close'] > l['ma50']: score += 2
            if l['rsi'] > 50: score += 2
            if l['adx'] > 25: score += 3
            st.metric("SCORE VN-INDEX", f"{score}/10")
            if score >= 7: st.success("🚀 MUA MẠNH")
            elif score >= 5: st.warning("⚖️ QUAN SÁT")
            else: st.error("⚠️ RỦI RO")
    
    menu = st.radio("MENU:", ["📈 ĐỒ THỊ KỸ THUẬT", "📊 DÒNG TIỀN NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- 4. KHÔNG GIAN HIỂN THỊ CHÍNH ---
vni_raw = pd.read_csv("vnindex.csv") if os.path.exists("vnindex.csv") else None
hose_raw = pd.read_csv("hose.csv") if os.path.exists("hose.csv") else None

if hose_raw is not None:
    hose_raw.columns = [str(c).lower() for c in hose_raw.columns]
    
    if menu == "📈 ĐỒ THỊ KỸ THUẬT":
        data = calculate_supreme(hose_raw[hose_raw['symbol'] == ticker], vni_raw)
        if data is not None:
            # Layout 4 tầng riêng biệt: Giá - Vol - RS/RSI - ADX
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.4, 0.15, 0.2, 0.25])
            
            # Tầng 1: Candle + MA
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # Tín hiệu Bom và Mua
            buys = data[data['buy_arrow']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.97, mode='markers+text', text="⬆️", textposition="bottom center", marker=dict(color='lime', size=12), name="MUA"), row=1, col=1)
            bombs = data[data['money_bomb']]
            fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.03, mode='markers+text', text="💣", textposition="top center", marker=dict(color='red', size=15), name="BOM TIỀN"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Khối lượng", marker_color='dodgerblue'), row=2, col=1)
            
            # Tầng 3: RSI & RS (Sức mạnh giá)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta', width=2), name="RS Sức Mạnh"), row=3, col=1)
            
            # Tầng 4: ADX (Sức mạnh xu hướng)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], line=dict(color='white'), fill='tozeroy', name="ADX Trend"), row=4, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="red", row=4, col=1)

            fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        else:
            st.error("Dữ liệu lỗi hoặc thiếu. Hãy nhấn Update Data.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN NGÀNH")
        nganh = {"BANK":['VCB','STB'], "CHỨNG":['SSI','VND','VCI','VIX','FTS'], "THÉP":['HPG','NKG','HSG'], "BĐS":['DIG','PDR','GEX']}
        summary = []
        for n, ms in nganh.items():
            pts = []
            for m in ms:
                d = calculate_supreme(hose_raw[hose_raw['symbol'] == m], vni_raw)
                if d is not None:
                    last = d.iloc[-1]
                    s = 4 if last['close'] > last['ma20'] else 0
                    if last['money_bomb']: s += 6
                    elif last['buy_arrow']: s += 3
                    pts.append(s)
            summary.append({"Ngành": n, "Điểm Dòng Tiền": np.mean(pts) if pts else 0})
        st.table(pd.DataFrame(summary).sort_values("Điểm Dòng Tiền", ascending=False))

    elif menu == "🎯 SIÊU ĐIỂM MUA":
        st.subheader("🚀 QUÉT ĐIỂM MUA & BOM TIỀN")
        results = []
        for m in hose_raw['symbol'].unique():
            d = calculate_supreme(hose_raw[hose_raw['symbol'] == m], vni_raw)
            if d is not None:
                last = d.iloc[-1]
                if last['money_bomb'] or last['buy_arrow']:
                    results.append({"Mã": m, "Tín hiệu": "💣 BOM TIỀN" if last['money_bomb'] else "⬆️ MUA", "RS": round(last['rs'],2), "RSI": round(last['rsi'],1)})
        st.dataframe(pd.DataFrame(results).sort_values("RS", ascending=False), use_container_width=True)
else:
    st.info("Nhấn 'UPDATE DATA REAL-TIME' ở bên trái để bắt đầu.")
