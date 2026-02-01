import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os
import warnings

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
st.set_page_config(page_title="V33.8 - FIREANT ULTIMATE", layout="wide")

# --- HÀM TÍNH TOÁN KỸ THUẬT SIÊU CẤP ---
def calculate_pro_signals(df, vni_df):
    if df is None or len(df) < 30: return None
    df = df.copy()
    
    # 1. Làm phẳng dữ liệu và chuẩn hóa tên cột
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).lower() for col in df.columns]
    
    # 2. Ép kiểu số và Reset Index để tránh lỗi ValueError khi so sánh
    for c in ['close', 'open', 'high', 'low', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    # 3. Tính toán các chỉ báo
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # RSI
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss.replace(0, 1))))
    
    # ADX
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    pdm = pd.Series(np.where((h.diff()>l.shift(1)-l)&(h.diff()>0), h.diff(), 0))
    mdm = pd.Series(np.where((l.shift(1)-l>h.diff())&(l.shift(1)-l>0), l.shift(1)-l, 0))
    pdi = 100 * (pdm.ewm(span=14, adjust=False).mean() / atr)
    mdi = 100 * (mdm.ewm(span=14, adjust=False).mean() / atr)
    df['adx'] = (100 * (abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan))).ewm(span=14, adjust=False).mean()

    # RS (Sức mạnh giá so với VNI) - Fix lỗi Index VNI
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').reset_index(drop=True)
    # Lấy 5 phiên gần nhất của VNI để so sánh tương quan
    vni_ratio = vni_c.iloc[-1] / vni_c.iloc[-5] if len(vni_c) > 5 else 1
    df['rs'] = ((c / c.shift(5)) / vni_ratio - 1) * 100
    
    # 4. Tín hiệu Bom 💣 & Mũi tên 🏹 (Đã fix lỗi so sánh Series)
    std = c.rolling(20).std()
    df['bb_w'] = (std * 4) / df['ma20']
    df['is_bomb'] = df['bb_w'] <= df['bb_w'].rolling(30).min()
    
    # So sánh giá trị thay vì Series
    df['is_buy'] = (df['close'] > df['ma20']) & (df['ma20'] > df['ma50']) & (df['volume'] > v.rolling(20).mean() * 1.3)
    
    # 5. Điểm dòng tiền
    score = 0
    if len(df) > 0:
        last = df.iloc[-1]
        if last['close'] > last['ma20']: score += 2
        if last['ma20'] > last['ma50']: score += 2
        if last['rs'] > 0: score += 3
        if last['volume'] > v.rolling(20).mean().iloc[-1]: score += 3
    df['total_score'] = score
    
    return df

# --- SIDEBAR (KHÔI PHỤC ĐẦY ĐỦ) ---
with st.sidebar:
    st.title("🛡️ HỆ THỐNG V33.8")
    
    # Ô NHẬP MÃ ĐẶT LÊN ĐẦU
    ticker = st.text_input("🔍 NHẬP MÃ SOI:", "MWG").upper()
    
    st.divider()
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU", use_container_width=True):
        with st.spinner("Đang tải dữ liệu..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vni_v33.csv")
            
            nganh_list = {
                'BAN_LE': ['MWG','FRT','DGW','MSN'],
                'CHUNG_KHOAN': ['SSI','VND','VCI','HCM','VIX'],
                'THEP': ['HPG','NKG','HSG'],
                'BDS': ['DIG','PDR','VHM','DXG','CEO'],
                'BANK': ['VCB','TCB','MBB','STB','LPB']
            }
            all_mã = [m for n in nganh_list.values() for m in n]
            data_all = []
            for m in all_mã:
                try:
                    t = yf.download(f"{m}.VN", period="2y", progress=False)
                    t['symbol'] = m
                    data_all.append(t)
                except: continue
            pd.concat(data_all).to_csv("hose_v33.csv")
            st.success("✅ Đã cập nhật!")
            st.rerun()

    mode = st.radio("CHẾ ĐỘ XEM:", ["📈 SOI CHI TIẾT MÃ", "📊 DÒNG TIỀN NGÀNH"])

# --- HIỂN THỊ CHÍNH ---
if os.path.exists("vni_v33.csv") and os.path.exists("hose_v33.csv"):
    vni_df = pd.read_csv("vni_v33.csv")
    hose_df = pd.read_csv("hose_v33.csv")

    if mode == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN THEO NHÓM NGÀNH")
        nganh_dict = {
            "BÁN LẺ": ['MWG','FRT','DGW','MSN'],
            "CHỨNG KHOÁN": ['SSI','VND','VCI','HCM','VIX'],
            "THÉP": ['HPG','NKG','HSG'],
            "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','DXG','CEO'],
            "NGÂN HÀNG": ['VCB','TCB','MBB','STB','LPB']
        }
        summary = []
        for ten, dsm in nganh_dict.items():
            scores = []
            for m in dsm:
                # Lọc dữ liệu và reset index để tránh lỗi ValueError
                m_data = hose_df[hose_df['symbol'] == m].copy().reset_index(drop=True)
                res = calculate_pro_signals(m_data, vni_df)
                if res is not None: scores.append(res['total_score'].iloc[-1])
            avg = np.mean(scores) if scores else 0
            tt = "🔥 DẪN DẮT" if avg >= 7 else "✅ TÍCH CỰC" if avg >= 5 else "☁️ TÍCH LŨY"
            summary.append({"Nhóm Ngành": ten, "Điểm": round(avg, 1), "Trạng Thái": tt})
        
        st.table(pd.DataFrame(summary).sort_values("Điểm", ascending=False))

    elif mode == "📈 SOI CHI TIẾT MÃ":
        m_data = hose_df[hose_df['symbol'] == ticker].copy().reset_index(drop=True)
        df_m = calculate_pro_signals(m_data, vni_df)
        
        if df_m is not None:
            # CHART 4 TẦNG
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # Ngày tháng chuẩn
            date_col = 'Date' if 'Date' in df_m.columns else 'date'
            
            # Tầng 1: Giá
            fig.add_trace(go.Candlestick(x=df_m[date_col], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m[date_col], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m[date_col], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # 💣 Bom & 🏹 Mua
            bombs = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=bombs[date_col], y=bombs['high']*1.03, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            buys = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=buys[date_col], y=buys['low']*0.97, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=df_m[date_col], y=df_m['volume'], name="Vol", marker_color='gray'), row=2, col=1)
            # Tầng 3: RSI & RS
            fig.add_trace(go.Scatter(x=df_m[date_col], y=df_m['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m[date_col], y=df_m['rs'], name="RS", line=dict(color='magenta')), row=3, col=1)
            # Tầng 4: ADX
            fig.add_trace(go.Scatter(x=df_m[date_col], y=df_m['adx'], name="ADX", line=dict(color='white')), row=4, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan',
                              yaxis=dict(side='right', fixedrange=False, autorange=True),
                              xaxis=dict(fixedrange=False))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            l = df_m.iloc[-1]
            st.success(f"🎯 Target: {l['close']*1.12:,.0f} | 🛑 Stoploss: {l['ma20']:,.0f} | 📊 Điểm: {l['total_score']}/10")
else:
    st.info("👋 Chào mừng! Hãy nhấn 'CẬP NHẬT DỮ LIỆU' ở cột bên trái để bắt đầu soi mã.")
