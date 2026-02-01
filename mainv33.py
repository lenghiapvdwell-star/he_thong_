import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V38.5 - GHOST FIX", layout="wide")

# --- DANH MỤC NGÀNH CHUẨN ---
NGANH_MASTER = {
    "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
    "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
    "THÉP": ['HPG','NKG','HSG'], 
    "NGÂN HÀNG": ['VCB','TCB','MBB','STB'],
    "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','GEX']
}
ALL_TICKERS = [t for sub in NGANH_MASTER.values() for t in sub]

# --- HÀM TÍNH TOÁN AN TOÀN ---
def calculate_indicators(df_raw):
    if df_raw is None or len(df_raw) < 10: return None
    df = df_raw.copy()
    
    # Ép kiểu dữ liệu số cho các cột quan trọng
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['close'])
    if len(df) < 10: return None

    # Tính toán MA & RSI
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    df['money_in'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.1)
    
    # Logic chấm điểm nhạy
    score = 0
    last = df.iloc[-1]
    if last['close'] >= last['ma10']: score += 4
    if last['close'] >= last['ma20']: score += 3
    if last['money_in']: score += 3
    
    df['total_score'] = score
    return df

# --- SIDEBAR: NÚT CẬP NHẬT ---
with st.sidebar:
    st.header("⚙️ HỆ THỐNG V38.5")
    ticker_input = st.text_input("🔍 SOI MÃ (HPG, SSI...):", "HPG").upper()
    
    if st.button("🔄 CẬP NHẬT REAL-TIME", use_container_width=True):
        with st.spinner("Đang quét dữ liệu sàn HOSE..."):
            all_list = []
            for m in ALL_TICKERS:
                # Tải dữ liệu và san phẳng ngay lập tức
                t = yf.download(f"{m}.VN", period="1y", interval="1d", progress=False)
                if not t.empty:
                    if isinstance(t.columns, pd.MultiIndex):
                        t.columns = t.columns.get_level_values(0)
                    t = t.reset_index()
                    t.columns = [str(c).strip().lower() for c in t.columns]
                    t['symbol'] = m # Gán nhãn mã chứng khoán
                    all_list.append(t)
            
            if all_list:
                final_df = pd.concat(all_list, ignore_index=True)
                final_df.to_csv("hose.csv", index=False)
                st.success(f"Đã cập nhật {len(all_list)} mã thành công!")
                st.rerun()

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ FIREANT", "📊 DÒNG TIỀN NGÀNH", "🎯 LỌC ĐIỂM MUA"])

# --- XỬ LÝ HIỂN THỊ ---
if os.path.exists("hose.csv"):
    hose_df = pd.read_csv("hose.csv")
    hose_df['date'] = pd.to_datetime(hose_df['date'], errors='coerce')
    
    # Đảm bảo symbol không bị khoảng trắng
    hose_df['symbol'] = hose_df['symbol'].str.strip()

    if menu == "📈 ĐỒ THỊ FIREANT":
        st.subheader(f"📊 PHÂN TÍCH: {ticker_input}")
        data_m = hose_df[hose_df['symbol'] == ticker_input].copy()
        df_res = calculate_indicators(data_m)
        
        if df_res is not None:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=df_res['date'], open=df_res['open'], high=df_res['high'], low=df_res['low'], close=df_res['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_res['date'], y=df_res['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            fig.add_trace(go.Bar(x=df_res['date'], y=df_res['volume'], name="Volume", marker_color='dodgerblue'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_res['date'], y=df_res['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Điểm Dòng Tiền {ticker_input}: {df_res['total_score'].iloc[-1]}/10")
        else:
            st.error(f"Mã {ticker_input} chưa có dữ liệu. Hãy nhấn 'CẬP NHẬT REAL-TIME'.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN NHÓM NGÀNH")
        res_nganh = []
        for n, mãs in NGANH_MASTER.items():
            pts = []
            for m in mãs:
                subset = hose_df[hose_df['symbol'] == m].copy()
                d = calculate_indicators(subset)
                if d is not None: pts.append(d['total_score'].iloc[-1])
            
            avg = np.mean(pts) if pts else 0
            res_nganh.append({"Ngành": n, "Sức Mạnh": round(avg, 1), "Số mã quét": len(pts)})
        
        st.table(pd.DataFrame(res_nganh).sort_values("Sức Mạnh", ascending=False))

    elif menu == "🎯 LỌC ĐIỂM MUA":
        results = []
        for s in hose_df['symbol'].unique():
            d = calculate_indicators(hose_df[hose_df['symbol'] == s].copy())
            if d is not None:
                l = d.iloc[-1]
                if l['total_score'] >= 7:
                    results.append({"Mã": s, "Điểm": l['total_score'], "RSI": round(l['rsi'],1)})
        st.dataframe(pd.DataFrame(results).sort_values("Điểm", ascending=False), use_container_width=True)
else:
