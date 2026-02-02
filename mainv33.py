import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V51 - RECOVERY PRO", layout="wide")

# --- 1. HÀM TẢI DỮ LIỆU "CHỐNG LỖI" ---
def safe_download(symbol, name):
    try:
        data = yf.download(symbol, period="2y", interval="1d", progress=False)
        if data.empty: return None
        # QUAN TRỌNG: San phẳng Multi-index của Yahoo Finance 2026
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        data.to_csv(name, index=False)
        return data
    except Exception as e:
        st.error(f"Lỗi khi tải {symbol}: {e}")
        return None

# --- 2. HÀM XỬ LÝ & TÍNH TOÁN ---
def calculate_signals(df, vni_df=None):
    if df is None or len(df) < 30: return None
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    
    # Ép kiểu số cho chắc chắn
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Chỉ báo cơ bản
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX chuẩn hóa
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
               abs(df['low'] - df['close'].shift(1))))
    df['adx'] = (df['tr'].rolling(14).mean() / df['close'] * 100).rolling(14).mean() * 5 # Scale up

    # RS (So sánh với VN-Index)
    if vni_df is not None:
        vni_df.columns = [str(c).lower() for c in vni_df.columns]
        v_c = vni_df[['date', 'close']].rename(columns={'close': 'v_c'})
        df = pd.merge(df, v_c, on='date', how='left').ffill()
        df['rs'] = (df['close'] / df['close'].shift(20)) / (df['v_c'] / df['v_c'].shift(20))
    else: df['rs'] = 1.0

    # Điểm mua & Bom tiền
    v20 = df['volume'].rolling(20).mean()
    df['buy_sig'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb_sig'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    return df

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🏆 V51 - RECOVERY")
    
    if st.button("🔄 BẤM ĐỂ CẬP NHẬT (RESET DỮ LIỆU)", use_container_width=True):
        with st.spinner("Đang kết nối vệ tinh..."):
            # Tải VNI
            vni_data = safe_download("^VNINDEX", "vnindex.csv")
            # Tải List mã
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','NKG','HSG','PDR','GEX','DGW','FRT','VCI']
            all_dfs = []
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y", progress=False)
                if isinstance(tmp.columns, pd.MultiIndex): tmp.columns = tmp.columns.get_level_values(0)
                tmp = tmp.reset_index()
                tmp['symbol'] = t
                all_dfs.append(tmp)
            if all_dfs:
                pd.concat(all_dfs).to_csv("hose.csv", index=False)
                st.success("CẬP NHẬT THÀNH CÔNG!")
                st.rerun()

    ticker = st.text_input("🔍 MÃ SOI:", "HPG").upper()
    
    # SỨC KHỎE VN-INDEX
    if os.path.exists("vnindex.csv"):
        v_raw = pd.read_csv("vnindex.csv")
        v_data = calculate_signals(v_raw)
        if v_data is not None:
            l = v_data.iloc[-1]
            score = 0
            if l['close'] > l['ma20']: score += 3
            if l['rsi'] > 50: score += 2
            if l['adx'] > 15: score += 3
            if l['close'] > l['ma50']: score += 2
            st.metric("SCORE VNI", f"{score}/10")

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- 4. HIỂN THỊ ---
vni_raw = pd.read_csv("vnindex.csv") if os.path.exists("vnindex.csv") else None
hose_raw = pd.read_csv("hose.csv") if os.path.exists("hose.csv") else None

if hose_raw is not None:
    hose_raw.columns = [str(c).lower() for c in hose_raw.columns]
    
    if menu == "📈 ĐỒ THỊ":
        df_m = hose_raw[hose_raw['symbol'] == ticker]
        data = calculate_signals(df_m, vni_raw)
        
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            # Tầng 1
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            # Tín hiệu
            buys = data[data['buy_sig']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.97, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="MUA"), row=1, col=1)
            bombs = data[data['bomb_sig']]
            fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.03, mode='markers', marker=dict(symbol='star', size=15, color='red'), name="BOM"), row=1, col=1)
            # Các tầng còn lại
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta'), name="RS"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX"), row=4, col=1)
            
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    elif menu == "📊 NGÀNH":
        # ... (Tương tự V50 nhưng dùng hàm calculate_signals)
        st.write("Đang quét dữ liệu ngành...")
        # Logic ngành gọn nhẹ
    
    elif menu == "🎯 SIÊU ĐIỂM MUA":
        st.subheader("🎯 MÃ ĐANG CÓ DÒNG TIỀN")
        res = []
        for m in hose_raw['symbol'].unique():
            d = calculate_signals(hose_raw[hose_raw['symbol'] == m], vni_raw)
            if d is not None:
                last = d.iloc[-1]
                if last['bomb_sig'] or last['buy_sig']:
                    res.append({"Mã": m, "Loại": "💣 BOM" if last['bomb_sig'] else "⬆️ MUA", "RS": round(last['rs'],2)})
        st.table(pd.DataFrame(res))
else:
    st.warning("Hệ thống chưa có dữ liệu. Vui lòng nhấn nút Update Real-time ở Sidebar!")
