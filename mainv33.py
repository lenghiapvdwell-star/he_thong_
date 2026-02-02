import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V52 - IRONCLAD TERMINAL", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU CỰC ĐOAN (CHỐNG LỖI CỘT) ---
def force_clean_df(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # 1. Xử lý Multi-index (Yahoo Finance hay bị lỗi này)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Reset index để lấy cột Date
    df = df.reset_index()
    
    # 3. Chuyển toàn bộ tên cột về chữ thường và bỏ khoảng trắng
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 4. Tìm và chuẩn hóa cột 'date' và 'close'
    rename_map = {}
    for col in df.columns:
        if col in ['date', 'datetime', 'ngày', 'index']: rename_map[col] = 'date'
        if col in ['close', 'đóng cửa', 'price']: rename_map[col] = 'close'
        if col in ['vol', 'volume', 'khối lượng']: rename_map[col] = 'volume'
        if col in ['high', 'cao']: rename_map[col] = 'high'
        if col in ['low', 'thấp']: rename_map[col] = 'low'
        if col in ['open', 'mở cửa']: rename_map[col] = 'open'
    
    df = df.rename(columns=rename_map)
    
    # Kiểm tra xem có đủ cột tối thiểu không
    required = ['date', 'close']
    if not all(c in df.columns for c in required):
        return None
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df.dropna(subset=['date', 'close']).sort_values('date')

# --- HÀM TÍNH TOÁN (ĐẢM BẢO ADX & RS LUÔN CÓ) ---
def calculate_signals(df, vni_df=None):
    df = force_clean_df(df)
    if df is None or len(df) < 20: return None
    
    # MA
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX (Simple version - Sức mạnh xu hướng)
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
               abs(df['low'] - df['close'].shift(1))))
    df['adx'] = (df['tr'].rolling(14).mean() / df['close'] * 100).rolling(14).mean() * 5

    # RS (Sức mạnh so với VNI)
    if vni_df is not None:
        vni_clean = force_clean_df(vni_df)
        if vni_clean is not None:
            v_c = vni_clean[['date', 'close']].rename(columns={'close': 'v_c'})
            df = pd.merge(df, v_c, on='date', how='left').ffill()
            df['rs'] = (df['close'] / df['close'].shift(20)) / (df['v_c'] / df['v_c'].shift(20))
        else: df['rs'] = 1.0
    else: df['rs'] = 1.0

    # Tín hiệu
    v20 = df['volume'].rolling(20).mean()
    df['buy_sig'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb_sig'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- GIAO DIỆN ---
with st.sidebar:
    st.header("🏆 MASTER V52")
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU MỚI", use_container_width=True):
        with st.spinner("Đang tải dữ liệu..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vnindex.csv")
            
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','NKG','HSG','STB','PDR','GEX','VCI','VIX']
            all_data = []
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y")
                tmp['symbol'] = t
                all_data.append(tmp)
            pd.concat(all_data).to_csv("hose.csv")
            st.success("Đã ghi đè file thành công!")
            st.rerun()

    ticker = st.text_input("🔍 NHẬP MÃ:", "HPG").upper()
    
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
            st.metric("VNI SCORE", f"{score}/10")
            st.progress(score/10)

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ CHÍNH ---
if os.path.exists("vnindex.csv") and os.path.exists("hose.csv"):
    vni_df = pd.read_csv("vnindex.csv")
    hose_all = pd.read_csv("hose.csv")
    hose_all.columns = [str(c).lower() for c in hose_all.columns]
    
    if menu == "📈 ĐỒ THỊ":
        df_m = hose_all[hose_all['symbol'] == ticker]
        data = calculate_signals(df_m, vni_df)
        
        if data is not None:
            # 4 Tầng: Giá, Vol, RS/RSI, ADX
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            
            # Nến & MA
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            
            # Bom và Mũi tên
            buys = data[data['buy_sig']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.97, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="MUA"), row=1, col=1)
            bombs = data[data['bomb_sig']]
            fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.03, mode='markers', marker=dict(symbol='star', size=15, color='red'), name="BOM"), row=1, col=1)

            # Vol
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Khối lượng", marker_color='dodgerblue'), row=2, col=1)
            
            # RSI & RS
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta', width=2), name="RS Sức Mạnh"), row=3, col=1)
            
            # ADX
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX Trend"), row=4, col=1)

            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        else:
            st.error("Không có dữ liệu cho mã này.")

    elif menu == "📊 NGÀNH":
        st.subheader("📊 SỨC MẠNH DÒNG TIỀN NGÀNH")
        nganh = {"BANK":['VCB','STB'], "CHỨNG":['SSI','VND','FTS','VCI','VIX'], "THÉP":['HPG','NKG','HSG'], "BĐS":['DIG','PDR','GEX']}
        summary = []
        for n, ms in nganh.items():
            pts = []
            for m in ms:
                d = calculate_signals(hose_all[hose_all['symbol'] == m], vni_df)
                if d is not None:
                    last = d.iloc[-1]
                    s = 4 if last['close'] > last['ma20'] else 0
                    if last['bomb_sig']: s += 6
                    elif last['buy_sig']: s += 3
                    pts.append(s)
            summary.append({"Ngành": n, "Điểm": np.mean(pts) if pts else 0})
        st.table(pd.DataFrame(summary).sort_values("Điểm", ascending=False))

    elif menu == "🎯 SIÊU ĐIỂM MUA":
        st.subheader("🎯 TÍN HIỆU DÒNG TIỀN TRONG NGÀY")
        res = []
        for m in hose_all['symbol'].unique():
            d = calculate_signals(hose_all[hose_all['symbol'] == m], vni_df)
            if d is not None:
                last = d.iloc[-1]
                if last['bomb_sig'] or last['buy_sig']:
                    res.append({"Mã": m, "Tín hiệu": "💣 BOM TIỀN" if last['bomb_sig'] else "⬆️ MUA", "RS": round(last['rs'],2), "RSI": round(last['rsi'],1)})
        st.dataframe(pd.DataFrame(res).sort_values("RS", ascending=False), use_container_width=True)
else:
    st.info("Vui lòng nhấn nút 'CẬP NHẬT DỮ LIỆU MỚI' để bắt đầu.")
