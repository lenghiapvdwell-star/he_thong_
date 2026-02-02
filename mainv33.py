import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V54 - SUPREME TERMINAL", layout="wide")

# --- HÀM LÀM SẠCH DỮ LIỆU CẤP ĐỘ CAO ---
def ultra_clean(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # San phẳng Multi-index của yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Mapping cột chuẩn
    mapping = {'date':'date', 'datetime':'date', 'index':'date', 'close':'close', 'adj close':'close', 'vol':'volume', 'volume':'volume'}
    df = df.rename(columns=mapping)
    
    # Chỉ lấy các cột cần thiết và ép kiểu Series 1 chiều
    needed = ['date', 'open', 'high', 'low', 'close', 'volume']
    existing = [c for c in needed if c in df.columns]
    df = df[existing]
    
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            # Ép về 1D Series để tránh TypeError
            s = df[c]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            df[c] = pd.to_numeric(s, errors='coerce')
            
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['date', 'close']).sort_values('date')

# --- HÀM TÍNH TOÁN CHỈ BÁO ---
def get_signals(df, vni_df=None):
    df = ultra_clean(df)
    if df is None or len(df) < 20: return None
    
    # Chỉ báo
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI
    diff = df['close'].diff()
    g = (diff.where(diff > 0, 0)).rolling(14).mean()
    l = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (g / l.replace(0, 0.001))))
    
    # ADX (Trend Strength)
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # RS (Relative Strength)
    if vni_df is not None:
        vni = ultra_clean(vni_df)
        if vni is not None:
            v_c = vni[['date', 'close']].rename(columns={'close':'v_c'})
            df = pd.merge(df, v_c, on='date', how='left').ffill()
            df['rs'] = (df['close']/df['close'].shift(20)) / (df['v_c']/df['v_c'].shift(20))
    if 'rs' not in df.columns: df['rs'] = 1.0
    
    # Signal
    v20 = df['volume'].rolling(20).mean()
    df['buy'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    return df

# --- SIDEBAR & CẬP NHẬT ---
with st.sidebar:
    st.header("🏆 SUPREME V54")
    
    if st.button("🔄 CẬP NHẬT REAL-TIME", use_container_width=True):
        with st.spinner("Đang đồng bộ dữ liệu..."):
            # 1. Tải VNI
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vnindex.csv")
            
            # 2. Tải từng mã riêng biệt để tránh lỗi trùng lặp DataFrame
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','NKG','HSG','STB','PDR','GEX','VCI','VIX']
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y")
                tmp.to_csv(f"{t}.csv")
            st.success("ĐÃ CẬP NHẬT XONG!")
            st.rerun()

    ticker = st.text_input("🔍 SOI MÃ:", "HPG").upper()
    
    # --- KHỐI SỨC KHỎE VN-INDEX ---
    if os.path.exists("vnindex.csv"):
        vni_data = get_signals(pd.read_csv("vnindex.csv"))
        if vni_data is not None:
            curr = vni_data.iloc[-1]
            score = sum([curr['close'] > curr['ma20'], curr['rsi'] > 50, curr['adx'] > 20, curr['close'] > curr['ma50']]) * 2.5
            st.metric("VNI HEALTH SCORE", f"{int(score)}/10")
            st.progress(score/10)
    
    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ KỸ THUẬT", "📊 DÒNG TIỀN NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ CHÍNH ---
vni_global = pd.read_csv("vnindex.csv") if os.path.exists("vnindex.csv") else None

if menu == "📈 ĐỒ THỊ KỸ THUẬT":
    file_path = f"{ticker}.csv"
    if os.path.exists(file_path):
        data = get_signals(pd.read_csv(file_path), vni_global)
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            
            # Icons
            b = data[data['buy']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'), name="MUA"), row=1, col=1)
            bm = data[data['bomb']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.02, mode='markers', marker=dict(symbol='star', size=14, color='red'), name="BOM"), row=1, col=1)

            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta'), name="RS"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX"), row=4, col=1)
            
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Chưa có dữ liệu mã {ticker}. Hãy nhấn Update Real-time.")

elif menu == "📊 DÒNG TIỀN NGÀNH":
    st.subheader("📊 SỨC MẠNH NHÓM NGÀNH")
    nganh = {"BANK":['VCB','STB'], "CHỨNG":['SSI','VND','FTS','VCI'], "THÉP":['HPG','NKG','HSG'], "BĐS":['DIG','PDR','GEX']}
    res = []
    for n, ms in nganh.items():
        scores = []
        for m in ms:
            if os.path.exists(f"{m}.csv"):
                d = get_signals(pd.read_csv(f"{m}.csv"), vni_global)
                if d is not None:
                    l = d.iloc[-1]
                    scores.append(10 if l['bomb'] else (5 if l['buy'] else 0))
        res.append({"Ngành": n, "Sức Mạnh": np.mean(scores) if scores else 0})
    st.table(pd.DataFrame(res).sort_values("Sức Mạnh", ascending=False))

elif menu == "🎯 SIÊU ĐIỂM MUA":
    st.subheader("🎯 CỔ PHIẾU CÓ TÍN HIỆU")
    found = []
    for f in os.listdir():
        if f.endswith(".csv") and f != "vnindex.csv" and f != "hose.csv":
            m = f.replace(".csv", "")
            d = get_signals(pd.read_csv(f), vni_global)
            if d is not None:
                l = d.iloc[-1]
                if l['bomb'] or l['buy']:
                    found.append({"Mã": m, "Tín hiệu": "💣 BOM" if l['bomb'] else "⬆️ MUA", "RS": round(l['rs'],2)})
    st.dataframe(pd.DataFrame(found).sort_values("RS", ascending=False), use_container_width=True)
