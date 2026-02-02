import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="SUPREME V56 - 2026", layout="wide")

# --- 1. BỘ LỌC DỮ LIỆU CỰC ĐOAN (CHỐNG MỌI LOẠI LỖI) ---
def clean_and_fix(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # San phẳng mọi cấu trúc Multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Tìm cột đóng cửa (chống lỗi KeyError 'close')
    possible_close = ['close', 'adj close', 'price', 'đóng cửa']
    found_close = next((c for c in possible_close if c in df.columns), None)
    
    if not found_close: return None
    df = df.rename(columns={found_close: 'close'})
    
    # Tìm cột ngày
    possible_date = ['date', 'datetime', 'ngày', 'index']
    found_date = next((c for c in possible_date if c in df.columns), None)
    if found_date: df = df.rename(columns={found_date: 'date'})
    
    if 'date' not in df.columns: return None
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Loại bỏ trùng lặp và ép kiểu số
    df = df.drop_duplicates(subset=['date']).dropna(subset=['date', 'close'])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df.sort_values('date').set_index('date')

# --- 2. TÍNH TOÁN CHỈ BÁO VÀ TÍN HIỆU ---
def get_signals(df_raw, vni_raw=None):
    df = clean_and_fix(df_raw)
    if df is None or len(df) < 20: return None
    
    # MA và RSI
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX (Simple Trend Strength)
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # RS (Relative Strength vs VNI)
    df['rs'] = 1.0
    if vni_raw is not None:
        vni = clean_and_fix(vni_raw)
        if vni is not None:
            common = df.index.intersection(vni.index)
            if len(common) > 20:
                s_price = df.loc[common, 'close']
                v_price = vni.loc[common, 'close']
                df.loc[common, 'rs'] = (s_price / s_price.shift(20)) / (v_price / v_price.shift(20))

    # Tín hiệu Mua và Bom Tiền
    v20 = df['volume'].rolling(20).mean()
    df['buy'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df.reset_index()

# --- 3. GIAO DIỆN SIDEBAR ---
with st.sidebar:
    st.title("🏆 SUPREME V56")
    st.subheader("Hệ thống Real-time 2026")
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU TỔNG", use_container_width=True):
        with st.spinner("Đang quét thị trường..."):
            # Tải VN-Index
            vni = yf.download("^VNINDEX", period="2y", progress=False)
            vni.to_csv("vnindex.csv")
            # Tải List mã chủ lực
            list_ma = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','NKG','HSG','STB','PDR','GEX','VCI','VIX','DGW','FRT']
            for m in list_ma:
                tmp = yf.download(f"{m}.VN", period="2y", progress=False)
                if not tmp.empty: tmp.to_csv(f"{m}.csv")
            st.success("ĐÃ CẬP NHẬT XONG!")
            st.rerun()

    ticker = st.text_input("🔍 SOI MÃ (VD: HPG):", "HPG").upper()
    
    # HIỂN THỊ SỨC KHỎE VNI
    if os.path.exists("vnindex.csv"):
        v_data = get_signals(pd.read_csv("vnindex.csv"))
        if v_data is not None:
            curr = v_data.iloc[-1]
            score = sum([curr['close'] > curr['ma20'], curr['rsi'] > 50, curr['adx'] > 15, curr['close'] > curr['ma50']]) * 2.5
            st.metric("VNI HEALTH SCORE", f"{int(score)}/10")
            st.progress(score/10)

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- 4. KHÔNG GIAN HIỂN THỊ CHÍNH ---
vni_global = pd.read_csv("vnindex.csv") if os.path.exists("vnindex.csv") else None

if menu == "📈 ĐỒ THỊ":
    f_path = f"{ticker}.csv"
    if os.path.exists(f_path):
        data = get_signals(pd.read_csv(f_path), vni_global)
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            
            # Giá & MA
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            
            # Vẽ Tín hiệu ⬆️ và 💣
            b = data[data['buy']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="MUA"), row=1, col=1)
            bm = data[data['bomb']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.02, mode='markers', marker=dict(symbol='star', size=15, color='red'), name="BOM"), row=1, col=1)

            # Volume, RSI/RS, ADX
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Volume", marker_color='dodgerblue'), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta'), name="RS"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX"), row=4, col=1)
            
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else: st.warning("Vui lòng nhấn 'Cập nhật dữ liệu' ở Sidebar.")

elif menu == "📊 NGÀNH":
    st.subheader("📊 SỨC MẠNH DÒNG TIỀN THEO NGÀNH")
    nganh_dict = {"BANK":['VCB','STB'], "CHỨNG":['SSI','VND','FTS','VCI','VIX'], "THÉP":['HPG','NKG','HSG'], "BĐS":['DIG','PDR','GEX']}
    results = []
    for n, ms in nganh_dict.items():
        scs = []
        for m in ms:
            if os.path.exists(f"{m}.csv"):
                d = get_signals(pd.read_csv(f"{m}.csv"), vni_global)
                if d is not None:
                    l = d.iloc[-1]
                    scs.append(10 if l['bomb'] else (5 if l['buy'] else 0))
        results.append({"Ngành": n, "Sức Mạnh": np.mean(scs) if scs else 0})
    st.table(pd.DataFrame(results).sort_values("Sức Mạnh", ascending=False))

elif menu == "🎯 SIÊU ĐIỂM MUA":
    st.subheader("🎯 DANH SÁCH TÍN HIỆU SIÊU CẤP")
    found_list = []
    for f in os.listdir():
        if f.endswith(".csv") and f != "vnindex.csv":
            d = get_signals(pd.read_csv(f), vni_global)
            if d is not None:
                l = d.iloc[-1]
                if l['bomb'] or l['buy']:
                    found_list.append({"Mã": f.replace(".csv",""), "Tín hiệu": "💣 BOM TIỀN" if l['bomb'] else "⬆️ MUA", "RS": round(l['rs'],2), "RSI": round(l['rsi'],1)})
    if found_list:
        st.dataframe(pd.DataFrame(found_list).sort_values("RS", ascending=False), use_container_width=True)
    else: st.info("Hôm nay chưa có tín hiệu mới.")
