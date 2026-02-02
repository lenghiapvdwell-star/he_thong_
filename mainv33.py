import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V55 - ZERO ERROR TERMINAL", layout="wide")

# --- HÀM LÀM SẠCH VÀ CHỐNG TRÙNG (ANTI-DUPLICATE) ---
def clean_and_fix(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # 1. San phẳng Multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 2. Mapping và ép kiểu
    mapping = {'date':'date', 'datetime':'date', 'index':'date', 'close':'close', 'vol':'volume', 'volume':'volume'}
    df = df.rename(columns=mapping)
    
    if 'date' not in df.columns: return None
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 3. XỬ LÝ TRÙNG LẶP (Sửa lỗi ValueError)
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.set_index('date').sort_index()
    
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            s = df[c]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            df[c] = pd.to_numeric(s, errors='coerce')
            
    return df.dropna(subset=['close'])

# --- HÀM TÍNH TOÁN SMART SIGNALS ---
def get_signals(df, vni_df=None):
    df = clean_and_fix(df)
    if df is None or len(df) < 25: return None
    
    # MA & RSI
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    diff = df['close'].diff()
    g = diff.where(diff > 0, 0).rolling(14).mean()
    l = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (g / l.replace(0, 0.001))))
    
    # ADX chuẩn hóa
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # RS (Sửa lỗi chia bằng cách đồng bộ Index)
    df['rs'] = 1.0
    if vni_df is not None:
        vni = clean_and_fix(vni_df)
        if vni is not None:
            # Chỉ lấy những ngày cả 2 cùng có dữ liệu
            common_idx = df.index.intersection(vni.index)
            if not common_idx.empty:
                stock_part = df.loc[common_idx, 'close']
                vni_part = vni.loc[common_idx, 'close']
                rs_val = (stock_part / stock_part.shift(20)) / (vni_part / vni_part.shift(20))
                df.loc[common_idx, 'rs'] = rs_val.ffill()

    # Signals
    v20 = df['volume'].rolling(20).mean()
    df['buy'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    return df.reset_index()

# --- SIDEBAR & CẬP NHẬT ---
with st.sidebar:
    st.header("🏆 SUPREME V55")
    if st.button("🔄 CẬP NHẬT DỮ LIỆU (FIX LỖI)", use_container_width=True):
        with st.spinner("Đang dọn dẹp và tải mới..."):
            vni = yf.download("^VNINDEX", period="2y", progress=False)
            vni.to_csv("vnindex.csv")
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','NKG','HSG','STB','PDR','GEX','VCI','VIX']
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y", progress=False)
                tmp.to_csv(f"{t}.csv")
            st.success("ĐÃ CẬP NHẬT & XÓA TRÙNG!")
            st.rerun()

    ticker = st.text_input("🔍 SOI MÃ:", "HPG").upper()
    
    if os.path.exists("vnindex.csv"):
        v_data = get_signals(pd.read_csv("vnindex.csv"))
        if v_data is not None:
            curr = v_data.iloc[-1]
            score = sum([curr['close'] > curr['ma20'], curr['rsi'] > 50, curr['adx'] > 20, curr['close'] > curr['ma50']]) * 2.5
            st.metric("VNI HEALTH", f"{int(score)}/10")
            st.progress(score/10)
    
    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ ---
vni_global = pd.read_csv("vnindex.csv") if os.path.exists("vnindex.csv") else None

if menu == "📈 ĐỒ THỊ":
    if os.path.exists(f"{ticker}.csv"):
        data = get_signals(pd.read_csv(f"{ticker}.csv"), vni_global)
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            
            # Tín hiệu
            b = data[data['buy']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'), name="MUA"), row=1, col=1)
            bm = data[data['bomb']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.02, mode='markers', marker=dict(symbol='star', size=14, color='red'), name="BOM"), row=1, col=1)

            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta'), name="RS"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX"), row=4, col=1)
            
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else: st.info("Nhấn Cập nhật dữ liệu để bắt đầu.")

elif menu == "📊 NGÀNH":
    st.subheader("📊 SỨC MẠNH NHÓM NGÀNH")
    nganh = {"BANK":['VCB','STB'], "CHỨNG":['SSI','VND','FTS','VCI'], "THÉP":['HPG','NKG','HSG'], "BĐS":['DIG','PDR','GEX']}
    res = []
    for n, ms in nganh.items():
        scs = []
        for m in ms:
            if os.path.exists(f"{m}.csv"):
                d = get_signals(pd.read_csv(f"{m}.csv"), vni_global)
                if d is not None:
                    l = d.iloc[-1]
                    scs.append(10 if l['bomb'] else (5 if l['buy'] else 0))
        res.append({"Ngành": n, "Sức Mạnh": np.mean(scs) if scs else 0})
    st.table(pd.DataFrame(res).sort_values("Sức Mạnh", ascending=False))

elif menu == "🎯 SIÊU ĐIỂM MUA":
    st.subheader("🎯 CỔ PHIẾU BÁO ĐIỂM MUA")
    found = []
    for f in os.listdir():
        if f.endswith(".csv") and f != "vnindex.csv":
            d = get_signals(pd.read_csv(f), vni_global)
            if d is not None:
                l = d.iloc[-1]
                if l['bomb'] or l['buy']:
                    found.append({"Mã": f.replace(".csv",""), "Tín hiệu": "💣 BOM" if l['bomb'] else "⬆️ MUA", "RS": round(l['rs'],2)})
    st.dataframe(pd.DataFrame(found).sort_values("RS", ascending=False), use_container_width=True)
