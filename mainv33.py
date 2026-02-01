import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V37.0 - CHUẨN FIREANT", layout="wide")

# --- DANH MỤC MÃ THEO NGÀNH (Dùng chung cho cả Tải & Lọc) ---
NGANH_MASTER = {
    "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
    "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
    "THÉP": ['HPG','NKG','HSG'], 
    "NGÂN HÀNG": ['VCB','TCB','MBB','STB'],
    "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','GEX']
}
ALL_TICKERS = [ticker for sublist in NGANH_MASTER.values() for ticker in sublist]

# --- HÀM TÍNH TOÁN ---
def calculate_master_signals(df, vni_df):
    if df is None or len(df) < 10: return None
    df = df.copy()
    
    # Chuẩn hóa cột
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    # Chuyển đổi kiểu dữ liệu
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    if len(df) < 10: return None
    
    # Tính toán chỉ báo
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # RS & Money In
    df['money_in'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.1)
    
    # Chấm điểm (Nới lỏng tối đa)
    score = 0
    try:
        last = df.iloc[-1]
        if last['close'] >= last['ma10']: score += 4
        if last['close'] >= last['ma20']: score += 3
        if last['money_in']: score += 3
        if last['rsi'] > 45: score += 2
    except: score = 0
    
    df['total_score'] = score
    df['is_buy'] = (df['close'] > df['ma20']) & (df['money_in'])
    df['is_bomb'] = (df['close'].rolling(10).std() / df['ma20'] < 0.02)
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏆 TERMINAL V37.0")
    ticker_input = st.text_input("🔍 SOI MÃ CHI TIẾT:", "MWG").upper()
    
    if st.button("🔄 LÀM MỚI TOÀN BỘ HỆ THỐNG", use_container_width=True):
        with st.spinner("Đang tải dữ liệu thực tế..."):
            vni = yf.download("^VNINDEX", period="1y")
            vni.to_csv("vni_v37.csv")
            
            all_data = []
            for m in ALL_TICKERS:
                t = yf.download(f"{m}.VN", period="1y", progress=False)
                if not t.empty:
                    t = t.reset_index()
                    t['symbol'] = m
                    all_data.append(t)
            
            if all_data:
                pd.concat(all_data).to_csv("hose_v37.csv", index=False)
                st.success(f"Đã tải {len(all_data)} mã!")
                st.rerun()

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ FIREANT", "📊 DÒNG TIỀN NGÀNH", "🎯 LỌC SIÊU ĐIỂM MUA"])

# --- KIỂM TRA DỮ LIỆU ---
if os.path.exists("vni_v37.csv") and os.path.exists("hose_v37.csv"):
    vni_df = pd.read_csv("vni_v37.csv")
    hose_df = pd.read_csv("hose_v37.csv")
    hose_df['date'] = pd.to_datetime(hose_df['Date'], errors='coerce')

    if menu == "📈 ĐỒ THỊ FIREANT":
        st.subheader(f"📊 PHÂN TÍCH KỸ THUẬT: {ticker_input}")
        # Lọc chính xác mã
        df_ticker = hose_df[hose_df['symbol'] == ticker_input].copy()
        df_m = calculate_master_signals(df_ticker, vni_df)
        
        if df_m is not None:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
            
            # Tầng 1: Giá & MA
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            
            # Tín hiệu
            buy_pts = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=buy_pts['date'], y=buy_pts['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=12, color='lime')), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], name="Volume", marker_color='gray'), row=2, col=1)
            
            # Tầng 3: RSI
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)

            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Điểm kỹ thuật {ticker_input}: {df_m['total_score'].iloc[-1]}/10")
        else:
            st.error(f"Không tìm thấy dữ liệu cho mã {ticker_input}. Hãy nhấn nút Làm mới hệ thống.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN NGÀNH")
        summary = []
        for n, mãs in NGANH_MASTER.items():
            pts = []
            for m in mãs:
                subset = hose_df[hose_df['symbol'] == m].copy()
                d = calculate_master_signals(subset, vni_df)
                if d is not None:
                    pts.append(d['total_score'].iloc[-1])
            
            avg = np.mean(pts) if pts else 0
            summary.append({"Ngành": n, "Sức Mạnh (10)": round(avg, 1), "Số mã": len(pts)})
        
        st.table(pd.DataFrame(summary).sort_values("Sức Mạnh (10)", ascending=False))

    elif menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.write("### 🚀 QUÉT ĐIỂM MUA THEO DÒNG TIỀN")
        results = []
        for s in hose_df['symbol'].unique():
            subset = hose_df[hose_df['symbol'] == s].copy()
            d = calculate_master_signals(subset, vni_df)
            if d is not None:
                l = d.iloc[-1]
                if l['total_score'] >= 7:
                    results.append({"Mã": s, "Điểm": l['total_score'], "RSI": round(l['rsi'],1), "Trạng thái": "🏹 MUA MẠNH"})
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("Điểm", ascending=False), use_container_width=True)
        else:
            st.info("Chưa có mã nào đạt điểm mua tối ưu (>7đ).")
else:
    st.warning("Hệ thống trống. Vui lòng nhấn nút 'LÀM MỚI TOÀN BỘ HỆ THỐNG' bên trái để tải dữ liệu.")
