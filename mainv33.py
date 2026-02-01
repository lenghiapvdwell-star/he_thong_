import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V46 - AUTO DATA ENGINE", layout="wide")

# --- HÀM TẢI DỮ LIỆU (TỰ ĐỘNG TẢI NẾU THIẾU FILE) ---
def get_data(symbol, file_name):
    # Nếu có file local thì dùng, không thì tải mới
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        with st.spinner(f"Đang tải dữ liệu {symbol} từ vệ tinh..."):
            ticker_yf = f"{symbol}.VN" if symbol != "^VNINDEX" else "^VNINDEX"
            df = yf.download(ticker_yf, period="2y", interval="1d", progress=False)
            if df.empty: return None
            df = df.reset_index()
            df.to_csv(file_name, index=False)
    
    # Chuẩn hóa dữ liệu
    df.columns = [str(c).strip().lower() for c in df.columns]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    for col in ['date', 'ngày', 'time']:
        if col in df.columns: df = df.rename(columns={col: 'date'}); break
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df.dropna(subset=['close']).sort_values('date')

# --- HÀM TÍNH TOÁN CHỈ BÁO ---
def calculate_pro_signals(df, vni_df=None):
    if df is None or len(df) < 15: return None
    df = df.copy()
    
    # 1. MA, RSI, ADX
    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(50, min_periods=1).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX Simple
    df['tr'] = np.maximum(df['high'] - df['low'], 
                np.maximum(abs(df['high'] - df['close'].shift(1)), 
                abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / df['atr'])
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / df['atr'])
    df['adx'] = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)).rolling(14).mean()

    # 2. RS (Relative Strength)
    if vni_df is not None:
        vni_c = vni_df[['date', 'close']].rename(columns={'close': 'v_c'})
        df = pd.merge(df, vni_c, on='date', how='left').ffill()
        df['rs'] = (df['close']/df['close'].shift(20)) / (df['v_c']/df['v_c'].shift(20))
    else: df['rs'] = 1.0

    # 3. Signals
    df['vol20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['buy_arrow'] = (df['close'] > df['ma20']) & (df['volume'] > df['vol20'] * 1.3)
    df['money_bomb'] = (df['volume'] > df['vol20'] * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- TẢI DỮ LIỆU HỆ THỐNG ---
vni_raw = get_data("^VNINDEX", "vnindex.csv")

# --- GIAO DIỆN ---
with st.sidebar:
    st.header("🏆 SUPREME V46")
    ticker = st.text_input("🔍 MÃ SOI:", "HPG").upper()
    
    if st.button("📈 KIỂM TRA VN-INDEX", use_container_width=True):
        if vni_raw is not None:
            v = calculate_pro_signals(vni_raw)
            l = v.iloc[-1]
            score = 0
            if l['close'] > l['ma20']: score += 3
            if l['rsi'] > 50: score += 2
            if l['adx'] > 20: score += 3
            if l['close'] > l['ma50']: score += 2
            st.metric("SỨC KHỎE VNI", f"{score}/10")
            st.write(f"ADX: {round(l['adx'],1)} | RSI: {round(l['rsi'],1)}")
            if score >= 7: st.success("THỊ TRƯỜNG CỰC KHỎE")
            else: st.warning("THỊ TRƯỜNG CẨN TRỌNG")

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- XỬ LÝ HIỂN THỊ ---
if menu == "📈 ĐỒ THỊ":
    data_raw = get_data(ticker, f"{ticker}.csv")
    data = calculate_pro_signals(data_raw, vni_raw)
    
    if data is not None:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.15, 0.15, 0.2])
        # Nến & MA
        fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['ma50'], line=dict(color='cyan'), name="MA50"), row=1, col=1)
        
        # Icon
        buys = data[data['buy_arrow']]
        fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.98, mode='markers+text', text="⬆️", textposition="bottom center", name="MUA"), row=1, col=1)
        bombs = data[data['money_bomb']]
        fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.02, mode='markers+text', text="💣", textposition="top center", name="BOM"), row=1, col=1)

        # Vol, RSI, ADX
        fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol"), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta'), name="RS"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX"), row=4, col=1)

        fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    else:
        st.error(f"Không thể lấy dữ liệu cho mã {ticker}")

elif menu == "📊 NGÀNH":
    nganh = {"THÉP":['HPG','NKG','HSG'], "BANK":['VCB','TCB','MBB'], "CHỨNG":['SSI','VND','VCI'], "BĐS":['DIG','PDR','VHM']}
    res = []
    for n, mãs in nganh.items():
        pts = []
        for m in mãs:
            d = calculate_pro_signals(get_data(m, f"{m}.csv"), vni_raw)
            if d is not None: pts.append(10 if d.iloc[-1]['money_bomb'] else (5 if d.iloc[-1]['buy_arrow'] else 0))
        res.append({"Ngành": n, "Sức Mạnh": np.mean(pts) if pts else 0})
    st.table(pd.DataFrame(res).sort_values("Sức Mạnh", ascending=False))

elif menu == "🎯 SIÊU ĐIỂM MUA":
    st.subheader("🚀 QUÉT SIÊU ĐIỂM MUA (Real-time)")
    # Quét danh sách mã mẫu
    list_check = ['HPG','SSI','MWG','VCB','DIG','VND','NKG','VCI','MSN','FTS']
    found = []
    for s in list_check:
        d = calculate_pro_signals(get_data(s, f"{s}.csv"), vni_raw)
        if d is not None and (d.iloc[-1]['money_bomb'] or d.iloc[-1]['buy_arrow']):
            found.append({"Mã": s, "Tín hiệu": "💣 BOM TIỀN" if d.iloc[-1]['money_bomb'] else "⬆️ MUA", "RS": round(d.iloc[-1]['rs'],2)})
    st.dataframe(pd.DataFrame(found), use_container_width=True)
