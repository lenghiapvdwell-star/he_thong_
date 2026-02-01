import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V47 - MASTER TERMINAL", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU CHUẨN (FIX LỖI KEYERROR) ---
def process_df(df):
    if df is None or df.empty: return None
    # 1. San phẳng Multi-index nếu có
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Reset index nếu Date đang nằm ở Index
    df = df.reset_index()
    
    # 3. Chuẩn hóa tên cột về chữ thường
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 4. Tìm và đổi tên cột ngày
    for col in ['date', 'datetime', 'ngày', 'time']:
        if col in df.columns:
            df = df.rename(columns={col: 'date'})
            break
            
    if 'date' not in df.columns: return None
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df.dropna(subset=['date', 'close']).sort_values('date')

# --- HÀM TÍNH TOÁN CHỈ BÁO ---
def calculate_pro_signals(df, vni_df=None):
    df = process_df(df)
    if df is None or len(df) < 15: return None
    
    # MA, RSI
    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(50, min_periods=1).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    up = df['high'] - df['high'].shift(1)
    down = df['low'].shift(1) - df['low']
    df['adx'] = (abs(up - down) / (up + down).replace(0, 1) * 100).rolling(14).mean()

    # RS
    if vni_df is not None:
        vni_c = vni_df[['date', 'close']].rename(columns={'close': 'v_c'})
        df = pd.merge(df, vni_c, on='date', how='left').ffill()
        df['rs'] = (df['close']/df['close'].shift(20)) / (df['v_c']/df['v_c'].shift(20))
    else: df['rs'] = 1.0

    df['vol20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['buy_arrow'] = (df['close'] > df['ma20']) & (df['volume'] > df['vol20'] * 1.3)
    df['money_bomb'] = (df['volume'] > df['vol20'] * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- GIAO DIỆN SIDEBAR ---
with st.sidebar:
    st.header("🏆 SUPREME V47")
    
    # NÚT UPDATE REAL-TIME
    if st.button("🔄 UPDATE REAL-TIME (Ghi đè .csv)", use_container_width=True):
        with st.spinner("Đang cập nhật toàn bộ hệ thống..."):
            # Tải VN-INDEX
            vni_new = yf.download("^VNINDEX", period="2y", interval="1d", progress=False)
            vni_new.to_csv("vnindex.csv")
            
            # Tải danh sách hose (Đại diện các mã lớn)
            hose_list = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN']
            all_data = []
            for m in hose_list:
                temp = yf.download(f"{m}.VN", period="2y", interval="1d", progress=False)
                temp = temp.reset_index()
                temp['symbol'] = m
                all_data.append(temp)
            pd.concat(all_data).to_csv("hose.csv", index=False)
            st.success("Đã ghi đè hose.csv và vnindex.csv thành công!")
            st.rerun()

    st.divider()
    ticker = st.text_input("🔍 MÃ SOI:", "HPG").upper()
    
    # KIỂM TRA SỨC KHỎE VNI
    if st.button("📈 SỨC KHỎE VN-INDEX", use_container_width=True):
        if os.path.exists("vnindex.csv"):
            v_raw = pd.read_csv("vnindex.csv")
            v_data = calculate_pro_signals(v_raw)
            if v_data is not None:
                l = v_data.iloc[-1]
                score = 0
                if l['close'] > l['ma20']: score += 3
                if l['rsi'] > 50: score += 2
                if l['adx'] > 20: score += 3
                if l['close'] > l['ma50']: score += 2
                st.metric("SCORE VN-INDEX", f"{score}/10")
                st.info(f"RSI: {round(l['rsi'],1)} | ADX: {round(l['adx'],1)}")
        else: st.error("Hãy nhấn Update Real-time trước!")

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ CHÍNH ---
vni_df = process_df(pd.read_csv("vnindex.csv")) if os.path.exists("vnindex.csv") else None
hose_df = pd.read_csv("hose.csv") if os.path.exists("hose.csv") else None

if menu == "📈 ĐỒ THỊ" and hose_df is not None:
    # Lọc mã thủ công để tránh lỗi KeyError
    hose_df.columns = [str(c).lower() for c in hose_df.columns]
    data_m = hose_df[hose_df['symbol'].str.upper() == ticker]
    data = calculate_pro_signals(data_m, vni_df)
    
    if data is not None:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.15, 0.15, 0.2])
        fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
        
        # Icon Bom & Mũi tên
        buys = data[data['buy_arrow']]
        fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.98, mode='markers+text', text="⬆️", textposition="bottom center", name="MUA"), row=1, col=1)
        bombs = data[data['money_bomb']]
        fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.02, mode='markers+text', text="💣", textposition="top center", name="BOM"), row=1, col=1)

        fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol"), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX"), row=4, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
