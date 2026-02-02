import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="V60 - SMART MONEY TERMINAL", layout="wide")

# --- 1. ENGINE XỬ LÝ DỮ LIỆU (CHỐNG MỌI LOẠI LỖI) ---
def master_cleaner(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # San phẳng Multi-index của yfinance 2026
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Chuẩn hóa tên cột cốt lõi
    mapping = {
        'date': 'date', 'datetime': 'date', 'index': 'date',
        'close': 'close', 'adj close': 'close',
        'vol': 'volume', 'volume': 'volume',
        'high': 'high', 'low': 'low', 'open': 'open'
    }
    df = df.rename(columns=mapping)
    
    # Lọc cột và ép kiểu số (Fix TypeError)
    needed = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in needed if c in df.columns]]
    
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            # Chỉ lấy cột đầu tiên nếu bị trùng tên
            series = df[c].iloc[:, 0] if isinstance(df[c], pd.DataFrame) else df[c]
            df[c] = pd.to_numeric(series, errors='coerce')
            
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['date', 'close']).drop_duplicates('date').sort_values('date')

# --- 2. HÀM TÍNH TOÁN CHỈ BÁO (RS, ADX, RSI, BOM) ---
def calculate_supreme(df, vni_df=None):
    df = master_cleaner(df)
    if df is None or len(df) < 30: return None
    
    # MA & RSI
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX chuẩn (Độ mạnh xu hướng)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift()), 
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # RS (Sức mạnh tương quan vs VN-Index)
    df['rs'] = 1.0
    if vni_df is not None:
        vni = master_cleaner(vni_df)
        if vni is not None:
            # Đồng bộ hóa ngày (Fix ValueError)
            df = df.set_index('date')
            vni = vni.set_index('date')
            common = df.index.intersection(vni.index)
            if not common.empty:
                rs_val = (df.loc[common, 'close'] / df.loc[common, 'close'].shift(20)) / \
                         (vni.loc[common, 'close'] / vni.loc[common, 'close'].shift(20))
                df.loc[common, 'rs'] = rs_val.ffill()
            df = df.reset_index()

    # Tín hiệu Mua & Bom tiền
    v20 = df['volume'].rolling(20).mean()
    df['buy_sig'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb_sig'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- 3. GIAO DIỆN SIDEBAR ---
with st.sidebar:
    st.header("🏆 SMART MONEY V60")
    if st.button("🔄 UPDATE REAL-TIME (2026)", use_container_width=True):
        with st.spinner("Đang tải dữ liệu vệ tinh..."):
            # Tải VNINDEX
            vni = yf.download("^VNINDEX", period="2y", progress=False)
            vni.to_csv("vni.csv")
            # Danh sách mã soi điểm mua
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','STB','NKG','HSG','PDR','GEX','VCI','VIX']
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y", progress=False)
                if not tmp.empty: tmp.to_csv(f"{t}.csv")
            st.success("ĐÃ CẬP NHẬT XONG!")
            st.rerun()

    ticker = st.text_input("🔍 SOI MÃ CỔ PHIẾU:", "HPG").upper()
    
    # SỨC KHỎE THỊ TRƯỜNG
    if os.path.exists("vni.csv"):
        v_data = calculate_supreme(pd.read_csv("vni.csv"))
        if v_data is not None:
            l = v_data.iloc[-1]
            score = sum([l['close'] > l['ma20'], l['rsi'] > 50, l['adx'] > 15, l['close'] > l['ma50']]) * 2.5
            st.metric("SỨC KHỎE VNI", f"{int(score)}/10")
            st.progress(score/10)

    menu = st.radio("MENU CHÍNH:", ["📈 ĐỒ THỊ KỸ THUẬT", "📊 DÒNG TIỀN NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- 4. HIỂN THỊ CHI TIẾT ---
vni_raw = pd.read_csv("vni.csv") if os.path.exists("vni.csv") else None

if menu == "📈 ĐỒ THỊ KỸ THUẬT":
    path = f"{ticker}.csv"
    if os.path.exists(path):
        data = calculate_supreme(pd.read_csv(path), vni_raw)
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            
            # Tầng 1: Candle & Tín hiệu
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            
            b = data[data['buy_sig']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12), name="MUA"), row=1, col=1)
            bm = data[data['bomb_sig']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.02, mode='markers', marker=dict(symbol='star', color='red', size=15), name="BOM"), row=1, col=1)

            # Tầng 2, 3, 4
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol", marker_color='dodgerblue'), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, name="RS (Sức mạnh)", line=dict(color='magenta', width=2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX (Xu hướng)", line=dict(color='white')), row=4, col=1)
            
            fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else: st.info("Nhấn 'UPDATE REAL-TIME' để tải dữ liệu.")

elif menu == "📊 DÒNG TIỀN NGÀNH":
    st.subheader("📊 XẾP HẠNG DÒNG TIỀN NHÓM NGÀNH")
    nganh = {"BANK":['VCB','STB'], "CHỨNG":['SSI','VND','VCI','VIX'], "THÉP":['HPG','NKG','HSG'], "BĐS":['DIG','PDR','GEX']}
    res = []
    for n, ms in nganh.items():
        pts = [10 if calculate_supreme(pd.read_csv(f"{m}.csv"), vni_raw).iloc[-1]['bomb_sig'] else 0 for m in ms if os.path.exists(f"{m}.csv")]
        res.append({"Ngành": n, "Dòng tiền (%)": np.mean(pts)*10 if pts else 0})
    st.table(pd.DataFrame(res).sort_values("Dòng tiền (%)", ascending=False))

elif menu == "🎯 SIÊU ĐIỂM MUA":
    st.subheader("🎯 BỘ LỌC CỔ PHIẾU MẠNH NHẤT")
    found = []
    for f in os.listdir():
        if f.endswith(".csv") and f != "vni.csv":
            d = calculate_supreme(pd.read_csv(f), vni_raw)
            if d is not None:
                l = d.iloc[-1]
                if l['bomb_sig'] or l['buy_sig']:
                    found.append({"Mã": f.replace(".csv",""), "Tín hiệu": "💣 BOM TIỀN" if l['bomb_sig'] else "⬆️ MUA", "RS": round(l['rs'],2), "RSI": round(l['rsi'],1)})
    st.dataframe(pd.DataFrame(found).sort_values("RS", ascending=False), use_container_width=True)
