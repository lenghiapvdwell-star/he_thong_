import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V48 - ULTIMATE TERMINAL", layout="wide")

# --- 1. HÀM XỬ LÝ DỮ LIỆU CỰC MẠNH (CHỐNG LỖI) ---
def clean_data(df):
    if df is None or df.empty: return None
    df = df.copy()
    # Phẳng hóa Multi-index nếu tải từ yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # Chuyển tên cột về lowercase và lọc cột cần thiết
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Tìm cột Date
    for c in ['date', 'datetime', 'ngày', 'index']:
        if c in df.columns:
            df = df.rename(columns={c: 'date'})
            break
    
    # Ép kiểu số
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['close', 'date']).sort_values('date')

# --- 2. HÀM TÍNH TOÁN CHỈ BÁO & CHIẾN THUẬT ---
def calculate_all(df, vni_df=None):
    df = clean_data(df)
    if df is None or len(df) < 20: return None
    
    # Chỉ báo xu hướng
    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(50, min_periods=1).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'].shift() * 100).rolling(14).mean() # ADX simplified

    # Dòng tiền & Điểm mua
    df['vol20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['buy_arrow'] = (df['close'] > df['ma20']) & (df['volume'] > df['vol20'] * 1.3)
    df['money_bomb'] = (df['volume'] > df['vol20'] * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    # Điểm RS (Sức mạnh tương quan)
    if vni_df is not None:
        vni_c = vni_df[['date', 'close']].rename(columns={'close': 'v_c'})
        df = pd.merge(df, vni_c, on='date', how='left').ffill()
        df['rs'] = (df['close']/df['close'].shift(20)) / (df['v_c']/df['v_c'].shift(20))
    else: df['rs'] = 1.0
    
    return df

# --- 3. SIDEBAR & CẬP NHẬT ---
with st.sidebar:
    st.header("🏆 SUPREME V48")
    
    if st.button("🔄 UPDATE REAL-TIME & OVERWRITE", use_container_width=True):
        with st.spinner("Đang tải dữ liệu 2026..."):
            # Update VNI
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vnindex.csv")
            
            # Update HOSE (Top mã)
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','STB','NKG','HSG','PDR','GEX','DGW','FRT']
            all_l = []
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y")
                tmp['symbol'] = t
                all_l.append(tmp)
            pd.concat(all_l).to_csv("hose.csv")
            st.success("Đã ghi đè dữ liệu mới nhất!")
            st.rerun()

    st.divider()
    ticker = st.text_input("🔍 SOI MÃ:", "HPG").upper()
    
    # --- KHỐI SỨC KHỎE VN-INDEX ---
    st.subheader("📊 THỊ TRƯỜNG CHUNG")
    if os.path.exists("vnindex.csv"):
        v_raw = pd.read_csv("vnindex.csv")
        v_data = calculate_all(v_raw)
        if v_data is not None:
            l = v_data.iloc[-1]
            score = 0
            if l['close'] > l['ma20']: score += 3
            if l['close'] > l['ma50']: score += 2
            if l['rsi'] > 50: score += 2
            if l['rsi'] < 70: score += 3 # Không quá mua
            st.metric("VNI SCORE", f"{score}/10")
            st.caption(f"RSI: {round(l['rsi'],1)} | MA20: {round(l['ma20'],0)}")
    
    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- 4. XỬ LÝ HIỂN THỊ CHÍNH ---
vni_df = clean_data(pd.read_csv("vnindex.csv")) if os.path.exists("vnindex.csv") else None
hose_raw = pd.read_csv("hose.csv") if os.path.exists("hose.csv") else None

if hose_raw is not None:
    hose_raw.columns = [str(c).lower() for c in hose_raw.columns]
    
    if menu == "📈 ĐỒ THỊ":
        df_m = hose_raw[hose_raw['symbol'] == ticker]
        data = calculate_all(df_m, vni_df)
        if data is not None:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            
            # Icon Mua & Bom
            buys = data[data['buy_arrow']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.98, mode='markers+text', text="⬆️", textposition="bottom center", marker=dict(color='lime', size=12), name="MUA"), row=1, col=1)
            bombs = data[data['money_bomb']]
            fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.02, mode='markers+text', text="💣", textposition="top center", marker=dict(color='red', size=15), name="BOM"), row=1, col=1)
            
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    elif menu == "📊 NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN THEO NHÓM")
        nganh_map = {"THÉP":['HPG','NKG','HSG'], "BANK":['VCB','STB'], "BĐS":['DIG','PDR','GEX'], "CHỨNG":['SSI','VND','FTS'], "BÁN LẺ":['MWG','FRT','DGW']}
        res = []
        for n, mãs in nganh_map.items():
            pts = []
            for m in mãs:
                d = calculate_all(hose_raw[hose_raw['symbol'] == m], vni_df)
                if d is not None:
                    score = 0
                    last = d.iloc[-1]
                    if last['close'] > last['ma20']: score += 4
                    if last['money_bomb']: score += 6
                    elif last['buy_arrow']: score += 3
                    pts.append(score)
            res.append({"Ngành": n, "Sức Mạnh": round(np.mean(pts),1) if pts else 0})
        st.table(pd.DataFrame(res).sort_values("Sức Mạnh", ascending=False))

    elif menu == "🎯 SIÊU ĐIỂM MUA":
        st.subheader("🚀 TÍN HIỆU DÒNG TIỀN TỔ CHỨC (24H)")
        found = []
        for m in hose_raw['symbol'].unique():
            d = calculate_all(hose_raw[hose_raw['symbol'] == m], vni_df)
            if d is not None:
                last = d.iloc[-1]
                if last['money_bomb'] or last['buy_arrow']:
                    found.append({"Mã": m, "Tín hiệu": "💣 BOM TIỀN" if last['money_bomb'] else "⬆️ MUA", "RS": round(last['rs'],2), "RSI": round(last['rsi'],1)})
        st.dataframe(pd.DataFrame(found).sort_values("RS", ascending=False), use_container_width=True)
else:
    st.info("Vui lòng nhấn 'UPDATE REAL-TIME' để khởi tạo dữ liệu!")
