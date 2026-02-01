import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V41 - PRO TERMINAL", layout="wide")

def load_data(file_name):
    if not os.path.exists(file_name): return None
    df = pd.read_csv(file_name)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ['symbol', 'ticker', 'mã']:
        if col in df.columns: df = df.rename(columns={col: 'symbol'})
        break
    for col in ['date', 'ngày']:
        if col in df.columns: df = df.rename(columns={col: 'date'})
        break
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.sort_values('date')

# --- TÍNH TOÁN CHỈ BÁO CHUYÊN SÂU ---
def calculate_pro_signals(df, vni_df=None):
    if df is None or len(df) < 20: return None
    df = df.copy()
    for c in ['close', 'high', 'low', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close'])

    # 1. Đường trung bình
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()

    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))

    # 3. ADX (Sức mạnh xu hướng)
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = df['low'].diff(-1).clip(lower=0)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift()), 
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()

    # 4. RS (Relative Strength vs VNINDEX)
    if vni_df is not None:
        vni_df = vni_df.sort_values('date')
        # Đồng bộ hóa ngày giữa CP và VNI
        combined = pd.merge(df[['date', 'close']], vni_df[['date', 'close']], on='date', suffixes=('', '_vni'))
        if len(combined) > 20:
            rs_val = (combined['close'] / combined['close'].shift(20)) / (combined['close_vni'] / combined['close_vni'].shift(20))
            df = pd.merge(df, pd.DataFrame({'date': combined['date'], 'rs': rs_val}), on='date', how='left')
    
    if 'rs' not in df.columns: df['rs'] = 0

    # 5. ĐIỂM MUA (Dòng tiền + Xu hướng)
    df['vol_20'] = df['volume'].rolling(20).mean()
    # Điều kiện MUA: Giá > MA20, Vol > 1.3 Vol20, RSI > 50
    df['buy_signal'] = (df['close'] > df['ma20']) & (df['volume'] > df['vol_20'] * 1.3) & (df['rsi'] > 50)
    
    return df

# --- GIAO DIỆN ---
hose_df = load_data("hose.csv")
vni_df = load_data("vnindex.csv")

with st.sidebar:
    st.header("🏆 PRO TERMINAL V41")
    ticker = st.text_input("🔍 MÃ CỔ PHIẾU:", "HPG").upper()
    st.divider()
    if st.button("📈 SỨC KHỎE VN-INDEX", use_container_width=True):
        if vni_df is not None:
            v_data = calculate_pro_signals(vni_df)
            last_v = v_data.iloc[-1]
            st.metric("VNI RSI", round(last_v['rsi'],1))
            st.metric("VNI ADX", round(last_v['adx'],1))
            st.write("Xu hướng: " + ("TĂNG" if last_v['close'] > last_v['ma20'] else "GIẢM"))
    
    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ KỸ THUẬT", "📊 DÒNG TIỀN NGÀNH", "🎯 ĐIỂM MUA BÙNG NỔ"])

if hose_df is not None:
    if menu == "📈 ĐỒ THỊ KỸ THUẬT":
        df_t = hose_df[hose_df['symbol'] == ticker]
        data = calculate_pro_signals(df_t, vni_df)
        
        if data is not None:
            # Thiết kế đồ thị 4 tầng
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                               row_heights=[0.5, 0.15, 0.15, 0.2])
            
            # Tầng 1: Candle + MA + Mũi tên Mua
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=1.5), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
            
            # Mũi tên điểm mua
            buys = data[data['buy_signal']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.96, mode='markers', 
                                     marker=dict(symbol='triangle-up', size=15, color='lime', line=dict(width=2, color='white')), 
                                     name="ĐIỂM MUA"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol", marker_color='gray'), row=2, col=1)
            
            # Tầng 3: RSI & RS (Sức mạnh giá)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta', dash='dot'), name="RS (Relative)"), row=3, col=1) # Scale để dễ nhìn
            
            # Tầng 4: ADX
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX (Trend)"), row=4, col=1)

            fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Không có dữ liệu mã này.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        # ... (Giữ logic cũ nhưng dùng calculate_pro_signals)
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN NHÓM NGÀNH")
        # Logic tương tự bản trước
        
    elif menu == "🎯 ĐIỂM MUA BÙNG NỔ":
        st.subheader("🚀 LỌC CỔ PHIẾU CÓ DÒNG TIỀN VÀO + XU HƯỚNG TĂNG")
        results = []
        for s in hose_df['symbol'].unique():
            d = calculate_pro_signals(hose_df[hose_df['symbol'] == s], vni_df)
            if d is not None:
                last = d.iloc[-1]
                if last['buy_signal']:
                    results.append({"Mã": s, "Giá": last['close'], "RSI": round(last['rsi'],1), "ADX": round(last['adx'],1)})
        st.dataframe(pd.DataFrame(results), use_container_width=True)
else:
    st.error("❌ Thiếu file hose.csv!")
