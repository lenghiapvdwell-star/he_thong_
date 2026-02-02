import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V69 - MARKET SENTIMENT", layout="wide")

def smart_loader(file_path):
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        if df.empty: return None
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Nhận diện cột ngày
        date_col = next((c for c in df.columns if any(k in c for k in ['date', 'ngày', 'time'])), df.columns[0])
        df = df.rename(columns={date_col: 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Nhận diện cột Mã (Dành cho hose.csv)
        sym_col = next((c for c in df.columns if any(k in c for k in ['symbol', 'ticker', 'mã'])), None)
        if sym_col: df = df.rename(columns={sym_col: 'symbol'})

        # Map giá
        mapping = {'close':['close','đóng','adj'],'open':['open','mở'],'high':['high','cao'],'low':['low','thấp'],'volume':['vol','khối']}
        for k, v in mapping.items():
            f = next((c for c in df.columns if any(p in c for p in v)), None)
            if f: df[k] = pd.to_numeric(df[f], errors='coerce')
            
        return df.dropna(subset=['date', 'close']).sort_values('date')
    except: return None

def get_market_status(df, vni_df=None):
    if df is None or len(df) < 30: return None
    df = df.copy()
    
    # Chỉ báo kỹ thuật nâng cao
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # Tính RSI
    change = df['close'].diff()
    df['rsi'] = 100 - (100 / (1 + (change.where(change > 0, 0).rolling(14).mean() / 
                                   -change.where(change < 0, 0).rolling(14).mean().replace(0, 0.001))))
    
    # Tính ADX (Độ mạnh xu hướng)
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # Nhận diện trạng thái (Tích lũy/Rung lắc)
    # Rung lắc: Khi giá bám sát MA20 và RSI đi ngang vùng 45-55
    df['sideway'] = (abs(df['close'] - df['ma20']) / df['ma20'] < 0.02) & (df['rsi'].between(45, 55))
    
    # RS (Sức mạnh tương quan)
    df['rs'] = 1.0
    if vni_df is not None:
        vni = vni_df.set_index('date')
        df_idx = df.set_index('date')
        common = df_idx.index.intersection(vni.index)
        if not common.empty:
            df_idx.loc[common, 'rs'] = (df_idx.loc[common, 'close']/df_idx.loc[common, 'close'].shift(20)) / \
                                      (vni.loc[common, 'close']/vni.loc[common, 'close'].shift(20))
        df = df_idx.reset_index()

    # Tín hiệu Mua/Bom tiền
    v20 = df['volume'].rolling(20).mean()
    df['buy'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3) & (df['rsi'] > 50)
    df['bomb'] = (df['volume'] > v20 * 2.5) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.header("🏆 V69 MARKET ANALYST")
    vni_raw = smart_loader("vnindex.csv")
    if vni_raw is not None:
        vni = get_market_status(vni_raw)
        curr = vni.iloc[-1]
        
        # Logic chấm điểm mới (Chặt chẽ hơn)
        score = 0
        if curr['close'] > curr['ma20']: score += 2
        if curr['close'] > curr['ma50']: score += 2
        if curr['rsi'] > 55: score += 2  # Nếu RSI < 55 tức là đang yếu/tích lũy
        if curr['adx'] > 20: score += 2  # ADX thấp là tích lũy
        if curr['volume'] > vni['volume'].rolling(20).mean().iloc[-1]: score += 2
        
        # Nhận diện trạng thái thị trường
        status = "BÙNG NỔ" if score >= 8 else "TÍCH LŨY / RUNG LẮC" if score >= 4 else "RỦI RO"
        st.metric("VNI HEALTH", f"{score}/10", help=f"Trạng thái: {status}")
        st.progress(score/10)
        st.write(f"🚩 **Trạng thái:** {status}")

    ticker = st.text_input("🔍 SOI MÃ CỔ PHIẾU:", "HPG").upper()
    menu = st.radio("CHỨC NĂNG:", ["📉 ĐỒ THỊ", "🎯 ĐIỂM MUA DÒNG TIỀN", "💾 RAW DATA HOSE"])

hose_raw = smart_loader("hose.csv")

if menu == "💾 RAW DATA HOSE":
    st.subheader("Kiểm tra file hose.csv")
    if hose_raw is not None:
        st.write("Cột tìm thấy:", list(hose_raw.columns))
        st.dataframe(hose_raw.head(50))
    else: st.error("Không tìm thấy dữ liệu hose.csv")

elif menu == "📉 ĐỒ THỊ":
    if hose_raw is not None:
        # Lọc theo symbol nếu có, không thì lấy cả file
        stock_df = hose_raw[hose_raw['symbol'] == ticker] if 'symbol' in hose_raw.columns else hose_raw
        data = get_market_status(stock_df, vni_raw)
        
        if data is not None and not data.empty:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='cyan'), name="MA20"), row=1, col=1)
            
            # Tín hiệu Mua & Bom
            b = data[data['buy']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12), name="MUA"), row=1, col=1)
            bm = data[data['bomb']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.02, mode='markers', marker=dict(symbol='star', color='red', size=15), name="BOM"), row=1, col=1)

            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, name="Sức mạnh RS", line=dict(color='magenta')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="Lực xu hướng"), row=4, col=1)
            
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tư vấn AI
            st.subheader("🤖 TƯ VẤN CHIẾN THUẬT")
            l = data.iloc[-1]
            if l['bomb']: st.success(f"🔥 {ticker}: DÒNG TIỀN VÀO CỰC MẠNH (BOM). Ưu tiên gia tăng tỷ trọng!")
            elif l['buy']: st.info(f"✅ {ticker}: Có tín hiệu mua chuẩn. Giải ngân từng phần.")
            elif l['sideway']: st.warning(f"⏳ {ticker}: Đang tích lũy/rung lắc quanh MA20. Chờ đợi bùng nổ.")
            else: st.write(f"👉 {ticker}: Chưa có tín hiệu đột biến. Quan sát thêm.")

elif menu == "🎯 ĐIỂM MUA DÒNG TIỀN":
    if hose_raw is not None and 'symbol' in hose_raw.columns:
        res = []
        for s in hose_raw['symbol'].unique():
            d = get_market_status(hose_raw[hose_raw['symbol'] == s], vni_raw)
            if d is not None:
                l = d.iloc[-1]
                if l['bomb'] or l['buy']:
                    res.append({"Mã": s, "Tín hiệu": "💣 BOM TIỀN" if l['bomb'] else "⬆️ MUA", "RS": round(l['rs'],2), "RSI": round(l['rsi'],1)})
        st.dataframe(pd.DataFrame(res).sort_values("RS", ascending=False), use_container_width=True)
