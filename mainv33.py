import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V36.8 - POWER ENGINE", layout="wide")

# --- HÀM TÍNH TOÁN (TỐI ƯU ĐIỂM SỐ) ---
def calculate_master_signals(df, vni_df):
    if df is None or len(df) < 20: return None 
    df = df.copy()
    
    # Chuẩn hóa cột
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if 'date' not in df.columns: df = df.reset_index().rename(columns={'index':'date'})
    
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    if len(df) < 20: return None
    
    c, v, h, l = df['close'], df['volume'], df['high'], df['low']
    
    # Chỉ báo
    df['ma10'] = c.rolling(10).mean()
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    std = c.rolling(20).std()
    df['bb_w'] = (std * 4) / df['ma20'].replace(0, 0.001)
    
    # RSI & RS
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # RS (Sức mạnh tương quan)
    vni_c_series = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').dropna().reset_index(drop=True)
    if len(vni_c_series) >= 5:
        v_ratio = vni_c_series.iloc[-1] / vni_c_series.iloc[-5]
        df['rs'] = ((c / c.shift(5)) / v_ratio - 1) * 100
    else: df['rs'] = 0
    
    # ADX đơn giản
    df['adx'] = (h - l).rolling(14).mean() / c.rolling(14).mean().replace(0,1) * 1000

    # Tín hiệu Mua & Chấm điểm
    df['money_in'] = (v > v.rolling(20).mean() * 1.1)
    df['is_bomb'] = (df['bb_w'] <= df['bb_w'].rolling(20).min())
    df['is_buy'] = (c > df['ma20']) & (df['money_in'])
    
    # LOGIC CHẤM ĐIỂM MỚI (DỄ CÓ ĐIỂM HƠN)
    score = 0
    try:
        last = df.iloc[-1]
        if last['close'] > last['ma10']: score += 3  # Xu hướng ngắn hạn tốt
        if last['close'] > last['ma20']: score += 2  # Xu hướng trung hạn tốt
        if last['money_in']: score += 3             # Có tiền vào
        if last['rs'] > -1: score += 2              # Không quá yếu so với VNI
    except: score = 0
        
    df['total_score'] = score
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏆 TRADING ENGINE V36.8")
    ticker = st.text_input("🔍 SOI MÃ CHI TIẾT:", "MWG").upper()
    
    if st.button("🔄 LÀM MỚI TOÀN BỘ DỮ LIỆU", use_container_width=True):
        with st.spinner("Đang tải dữ liệu 20 siêu cổ phiếu..."):
            # Xóa file cũ để tránh xung đột
            if os.path.exists("hose_v35.csv"): os.remove("hose_v35.csv")
            
            vni = yf.download("^VNINDEX", period="1y")
            vni.to_csv("vni_v35.csv")
            
            # Danh sách mã được chọn lọc kỹ
            m_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','VIX','FTS','HPG','NKG','HSG','VCB','TCB','MBB','STB','DIG','PDR','VHM','GEX']
            all_d = []
            for m in m_list:
                t = yf.download(f"{m}.VN", period="1y", progress=False)
                if not t.empty:
                    t['symbol'] = m
                    all_d.append(t)
            
            if all_d:
                full_df = pd.concat(all_d)
                full_df.to_csv("hose_v35.csv")
                st.success(f"Đã tải {len(all_d)} mã thành công!")
                st.rerun()

    menu = st.radio("CHỨC NĂNG:", ["📊 DÒNG TIỀN NGÀNH", "🎯 LỌC SIÊU ĐIỂM MUA", "📈 ĐỒ THỊ FIREANT"])

# --- HIỂN THỊ ---
if os.path.exists("vni_v35.csv") and os.path.exists("hose_v35.csv"):
    vni_df = pd.read_csv("vni_v35.csv")
    hose_df = pd.read_csv("hose_v35.csv")
    
    if menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN THEO NHÓM NGÀNH")
        nganh_dict = {
            "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
            "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
            "THÉP": ['HPG','NKG','HSG'], 
            "NGÂN HÀNG": ['VCB','TCB','MBB','STB'],
            "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','GEX']
        }
        
        summary = []
        for n, mãs in nganh_dict.items():
            pts = []
            for m in mãs:
                subset = hose_df[hose_df['symbol'] == m].copy()
                if not subset.empty:
                    d = calculate_master_signals(subset, vni_df)
                    if d is not None: pts.append(d['total_score'].iloc[-1])
            
            # Tính trung bình cộng của ngành
            avg = np.mean(pts) if len(pts) > 0 else 0
            summary.append({"Ngành": n, "Điểm Dòng Tiền": round(avg, 1), "Số mã quét": len(pts)})
        
        res_df = pd.DataFrame(summary).sort_values("Điểm Dòng Tiền", ascending=False)
        st.table(res_df)
        st.info("💡 Nếu ngành có Điểm = 0, hãy nhấn 'LÀM MỚI TOÀN BỘ DỮ LIỆU' để tải lại danh mục mã.")

    elif menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.write("### 🚀 Tín hiệu Mua: Giá > MA20 + Tiền vào")
        results = []
        for s in hose_df['symbol'].unique():
            subset = hose_df[hose_df['symbol'] == s].copy()
            d = calculate_master_signals(subset, vni_df)
            if d is not None:
                l = d.iloc[-1]
                if l['is_buy'] or l['is_bomb']:
                    results.append({"Mã": s, "Điểm": l['total_score'], "RSI": round(l['rsi'],1), "Tín hiệu": "🏹 MUA" if l['is_buy'] else "💣 NÉN"})
        
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("Điểm", ascending=False), use_container_width=True)
        else: st.info("Hệ thống đang chờ tín hiệu bùng nổ...")

    elif menu == "📈 ĐỒ THỊ FIREANT":
        df_m = calculate_master_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.1, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # Tín hiệu
            b = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=b['date'], y=b['high']*1.02, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            s = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=s['date'], y=s['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            v_colors = ['red' if df_m['close'].iloc[i] < df_m['open'].iloc[i] else 'green' for i in range(len(df_m))]
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], marker_color=v_colors, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], line=dict(color='magenta'), name="RS"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['adx'], line=dict(color='white'), name="ADX"), row=4, col=1)

            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            fig.update_yaxes(side="right", fixedrange=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
else:
    st.warning("⚠️ Không tìm thấy dữ liệu. Hãy nhấn 'LÀM MỚI TOÀN BỘ DỮ LIỆU' ở cột trái.")
