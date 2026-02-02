import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os
from datetime import datetime

# --- CẤU HÌNH ---
st.set_page_config(page_title="V70 - REAL-TIME TRADING PRO", layout="wide")

# --- 1. HÀM CẬP NHẬT REAL-TIME (LẤY GIÁ MỚI NHẤT GHÉP VÀO CSV) ---
def fetch_realtime_data(symbol, existing_df):
    try:
        # Tải dữ liệu 5 ngày gần nhất để đảm bảo lấy được nến hôm nay
        ticker_yf = f"{symbol}.VN" if symbol != "^VNINDEX" else "^VNINDEX"
        new_data = yf.download(ticker_yf, period="5d", interval="1d", progress=False)
        
        if new_data.empty: return existing_df
        
        # San phẳng dữ liệu yfinance
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)
        new_data = new_data.reset_index()
        new_data.columns = [str(c).lower() for c in new_data.columns]
        new_data = new_data.rename(columns={'date': 'date', 'adj close': 'close'})
        
        # Hợp nhất với dữ liệu cũ, tránh trùng lặp ngày
        combined = pd.concat([existing_df, new_data], ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        return combined.sort_values('date')
    except Exception as e:
        st.warning(f"Không thể cập nhật Real-time cho {symbol}: {e}")
        return existing_df

# --- 2. BỘ GIẢI MÃ CSV VẠN NĂNG ---
def smart_loader(file_path):
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        if df.empty: return None
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        date_col = next((c for c in df.columns if any(k in c for k in ['date', 'ngày', 'time'])), df.columns[0])
        df = df.rename(columns={date_col: 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        sym_col = next((c for c in df.columns if any(k in c for k in ['symbol', 'ticker', 'mã'])), None)
        if sym_col: df = df.rename(columns={sym_col: 'symbol'})

        mapping = {'close':['close','đóng','adj'],'open':['open','mở'],'high':['high','cao'],'low':['low','thấp'],'volume':['vol','khối']}
        for k, v in mapping.items():
            f = next((c for c in df.columns if any(p in c for p in v)), None)
            if f: 
                df[k] = pd.to_numeric(df[f], errors='coerce')
        return df.dropna(subset=['date', 'close']).sort_values('date')
    except: return None

# --- 3. TÍNH TOÁN KỸ THUẬT & TRẠNG THÁI THỊ TRƯỜNG ---
def calculate_all(df, vni_df=None):
    if df is None or len(df) < 30: return None
    df = df.copy()
    
    # Chỉ báo chuẩn
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI & ADX
    change = df['close'].diff()
    df['rsi'] = 100 - (100 / (1 + (change.where(change > 0, 0).rolling(14).mean() / 
                                   -change.where(change < 0, 0).rolling(14).mean().replace(0, 0.001))))
    
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # Nhận diện Rung lắc / Tích lũy
    df['status'] = "Normal"
    df.loc[abs(df['close'] - df['ma20'])/df['ma20'] < 0.015, 'status'] = "Rung lắc / Tích lũy"

    # RS (Sức mạnh so với VNI)
    df['rs'] = 1.0
    if vni_df is not None:
        vni = vni_df.set_index('date')
        df_idx = df.set_index('date')
        common = df_idx.index.intersection(vni.index)
        if not common.empty:
            df_idx.loc[common, 'rs'] = (df_idx.loc[common, 'close']/df_idx.loc[common, 'close'].shift(20)) / \
                                      (vni.loc[common, 'close']/vni.loc[common, 'close'].shift(20))
        df = df_idx.reset_index()

    # Tín hiệu Mua & Bom tiền
    v20 = df['volume'].rolling(20).mean()
    df['buy'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3) & (df['rsi'] > 50)
    df['bomb'] = (df['volume'] > v20 * 2.5) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- GIAO DIỆN ---
with st.sidebar:
    st.header("🏆 V70 REAL-TIME PRO")
    
    # Nút bấm quan trọng nhất: Cập nhật giá Real-time
    update_clicked = st.button("🔄 CẬP NHẬT GIÁ REAL-TIME", use_container_width=True)
    
    vni_raw = smart_loader("vnindex.csv")
    if update_clicked:
        with st.spinner("Đang lấy giá VNI mới nhất..."):
            vni_raw = fetch_realtime_data("^VNINDEX", vni_raw)

    if vni_raw is not None:
        vni = calculate_all(vni_raw)
        curr = vni.iloc[-1]
        score = sum([curr['close'] > curr['ma20'], curr['rsi'] > 55, curr['adx'] > 18, curr['close'] > curr['ma50']]) * 2.5
        st.metric(f"VNI: {curr['close']:,.2f}", f"{score}/10 Health")
        st.write(f"Cập nhật: {curr['date'].strftime('%d/%m/%Y')}")

    ticker = st.text_input("🔍 SOI MÃ CỔ PHIẾU:", "HPG").upper()
    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "🎯 ĐIỂM MUA REAL-TIME"])

hose_raw = smart_loader("hose.csv")

if menu == "📈 ĐỒ THỊ":
    if hose_raw is not None:
        # Tách mã và cập nhật Real-time cho mã đang soi
        stock_df = hose_raw[hose_raw['symbol'] == ticker] if 'symbol' in hose_raw.columns else hose_raw
        
        if update_clicked:
            with st.spinner(f"Đang đồng bộ giá {ticker}..."):
                stock_df = fetch_realtime_data(ticker, stock_df)
        
        data = calculate_all(stock_df, vni_raw)
        
        if data is not None and not data.empty:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='cyan'), name="MA20"), row=1, col=1)
            
            # Tín hiệu
            b = data[data['buy']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.99, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12), name="MUA"), row=1, col=1)
            bm = data[data['bomb']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.01, mode='markers', marker=dict(symbol='star', color='red', size=15), name="BOM"), row=1, col=1)

            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, name="RS", line=dict(color='magenta')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="Trend"), row=4, col=1)
            
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # TƯ VẤN NHANH
            l = data.iloc[-1]
            st.subheader(f"🤖 CHIẾN THUẬT CHO {ticker}")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Giá:** {l['close']:,.0f}")
                st.write(f"**Trạng thái:** {l['status']}")
            with col2:
                st.write(f"**Sức mạnh RS:** {l['rs']:.2f}")
                st.write(f"**Xung lực RSI:** {l['rsi']:.1f}")

elif menu == "🎯 ĐIỂM MUA REAL-TIME":
    st.info("Nhấn 'CẬP NHẬT GIÁ REAL-TIME' ở sidebar trước khi lọc.")
    if hose_raw is not None and 'symbol' in hose_raw.columns:
        res = []
        # Chỉ lọc top các mã phổ biến để tránh làm chậm hệ thống khi update real-time
        common_stocks = hose_raw['symbol'].unique()[:50] 
        for s in common_stocks:
            d = calculate_all(hose_raw[hose_raw['symbol'] == s], vni_raw)
            if d is not None:
                l = d.iloc[-1]
                if l['bomb'] or l['buy']:
                    res.append({"Mã": s, "Tín hiệu": "💣 BOM" if l['bomb'] else "⬆️ MUA", "RS": round(l['rs'],2)})
        st.dataframe(pd.DataFrame(res).sort_values("RS", ascending=False), use_container_width=True)
