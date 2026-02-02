import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V57 - THE FINAL ENGINE", layout="wide")

# --- 1. HÀM XỬ LÝ DỮ LIỆU GỐC (CHỐNG LỖI CỘT & TRÙNG) ---
def clean_data_robust(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # San phẳng Multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Tìm cột chuẩn
    rename_dict = {}
    for c in df.columns:
        if any(x in c for x in ['date', 'time', 'index']): rename_dict[c] = 'date'
        elif any(x in c for x in ['close', 'adj']): rename_dict[c] = 'close'
        elif 'open' in c: rename_dict[c] = 'open'
        elif 'high' in c: rename_dict[c] = 'high'
        elif 'low' in c: rename_dict[c] = 'low'
        elif any(x in c for x in ['vol', 'amount']): rename_dict[c] = 'volume'
    
    df = df.rename(columns=rename_dict)
    
    if 'date' not in df.columns or 'close' not in df.columns:
        return None
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'close']).drop_duplicates(subset=['date'])
    
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df.sort_values('date').set_index('date')

# --- 2. HÀM TÍNH TOÁN (ĐỒNG BỘ RS & ADX) ---
def calculate_all_indicators(stock_df, vni_df=None):
    s_df = clean_data_robust(stock_df)
    if s_df is None or len(s_df) < 25: return None
    
    # Đồng bộ hóa với VNI để tính RS
    if vni_df is not None:
        v_df = clean_data_robust(vni_df)
        if v_df is not None:
            # Reindex để s_df và v_df có cùng số dòng, cùng ngày
            combined = s_df.join(v_df[['close']], rsuffix='_vni', how='left').ffill()
            s_df['rs'] = (combined['close'] / combined['close'].shift(20)) / \
                         (combined['close_vni'] / combined['close_vni'].shift(20))
    
    if 'rs' not in s_df.columns: s_df['rs'] = 1.0
    
    # Chỉ báo kỹ thuật
    s_df['ma20'] = s_df['close'].rolling(20).mean()
    s_df['ma50'] = s_df['close'].rolling(50).mean()
    
    # RSI
    delta = s_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    s_df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX chuẩn
    tr = pd.concat([s_df['high'] - s_df['low'], 
                    abs(s_df['high'] - s_df['close'].shift()), 
                    abs(s_df['low'] - s_df['close'].shift())], axis=1).max(axis=1)
    s_df['adx'] = (tr.rolling(14).mean() / s_df['close'] * 500).rolling(14).mean()
    
    # Tín hiệu
    v20 = s_df['volume'].rolling(20).mean()
    s_df['buy'] = (s_df['close'] > s_df['ma20']) & (s_df['volume'] > v20 * 1.3)
    s_df['bomb'] = (s_df['volume'] > v20 * 2.2) & (s_df['close'] > s_df['close'].shift(1) * 1.03)
    
    return s_df.reset_index()

# --- 3. SIDEBAR & DATA ---
with st.sidebar:
    st.header("🏆 TERMINAL V57")
    if st.button("🔄 CẬP NHẬT DỮ LIỆU TỔNG", use_container_width=True):
        with st.spinner("Đang tải dữ liệu 2026..."):
            vni = yf.download("^VNINDEX", period="2y", progress=False)
            vni.to_csv("vnindex.csv")
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','NKG','HSG','STB','PDR','GEX','VCI','VIX']
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y", progress=False)
                if not tmp.empty: tmp.to_csv(f"{t}.csv")
            st.success("XONG! DỮ LIỆU ĐÃ SẴN SÀNG.")
            st.rerun()

    ticker = st.text_input("🔍 NHẬP MÃ:", "HPG").upper()
    
    if os.path.exists("vnindex.csv"):
        v_data = calculate_all_indicators(pd.read_csv("vnindex.csv"))
        if v_data is not None:
            c = v_data.iloc[-1]
            score = sum([c['close'] > c['ma20'], c['rsi'] > 50, c['adx'] > 15, c['close'] > c['ma50']]) * 2.5
            st.metric("SỨC KHỎE VNI", f"{int(score)}/10")

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- 4. HIỂN THỊ ĐỒ THỊ 4 TẦNG ---
vni_raw = pd.read_csv("vnindex.csv") if os.path.exists("vnindex.csv") else None

if menu == "📈 ĐỒ THỊ":
    path = f"{ticker}.csv"
    if os.path.exists(path):
        data = calculate_all_indicators(pd.read_csv(path), vni_raw)
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            # Tầng 1: Candle
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            # Tín hiệu
            b = data[data['buy']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name="MUA"), row=1, col=1)
            bm = data[data['bomb']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.02, mode='markers', marker=dict(symbol='star', color='red', size=14), name="BOM"), row=1, col=1)
            # Các tầng còn lại
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, name="RS", line=dict(color='magenta')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX", line=dict(color='white')), row=4, col=1)
            
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else: st.warning("Hãy nhấn Cập nhật dữ liệu ở sidebar.")

elif menu == "📊 NGÀNH":
    st.subheader("📊 SỨC MẠNH DÒNG TIỀN THEO NGÀNH")
    nganh = {"BANK":['VCB','STB'], "CHỨNG":['SSI','VND','FTS','VCI'], "THÉP":['HPG','NKG','HSG'], "BĐS":['DIG','PDR','GEX']}
    res = []
    for n, ms in nganh.items():
        sc = [10 if calculate_all_indicators(pd.read_csv(f"{m}.csv"), vni_raw).iloc[-1]['bomb'] else 0 for m in ms if os.path.exists(f"{m}.csv")]
        res.append({"Ngành": n, "Sức Mạnh": np.mean(sc) if sc else 0})
    st.table(pd.DataFrame(res).sort_values("Sức Mạnh", ascending=False))

elif menu == "🎯 SIÊU ĐIỂM MUA":
    st.subheader("🎯 BỘ LỌC SMART MONEY")
    found = []
    for f in os.listdir():
        if f.endswith(".csv") and f != "vnindex.csv":
            d = calculate_all_indicators(pd.read_csv(f), vni_raw)
            if d is not None:
                l = d.iloc[-1]
                if l['bomb'] or l['buy']:
                    found.append({"Mã": f.replace(".csv",""), "Tín hiệu": "💣 BOM" if l['bomb'] else "⬆️ MUA", "RS": round(l['rs'],2)})
    st.dataframe(pd.DataFrame(found).sort_values("RS", ascending=False), use_container_width=True)
