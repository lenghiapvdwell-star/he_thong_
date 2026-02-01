import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V36.0 - PRO TERMINAL", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU & TÍNH TOÁN CHỈ BÁO ---
def calculate_master_signals(df, vni_df):
    if df is None or len(df) < 130: return None # Cần ít nhất 6 tháng (130 phiên)
    df = df.copy()
    
    # Chuẩn hóa cột
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if 'date' not in df.columns: df = df.reset_index().rename(columns={'index':'date'})
    df['date'] = pd.to_datetime(df['date'])
    
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    
    c, v, h, l = df['close'], df['volume'], df['high'], df['low']
    
    # 1. Chỉ báo xu hướng
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # 2. Bollinger Band (Độ hẹp)
    std = c.rolling(20).std()
    df['bb_w'] = (std * 4) / df['ma20']
    
    # 3. RSI & ADX & RS
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # RS so với VNIndex
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').reset_index(drop=True)
    df['rs'] = ((c / c.shift(10)) / (vni_c / vni_c.shift(10)) - 1) * 100
    
    # ADX đơn giản
    df['adx'] = (c.diff().abs().rolling(14).mean() / c.rolling(14).mean()) * 1000

    # 4. LOGIC ĐIỂM MUA (NỀN GIÁ 6 THÁNG)
    # Độ biến động thấp trong 120 phiên (6 tháng)
    df['range_6m'] = (h.rolling(120).max() - l.rolling(120).min()) / l.rolling(120).min()
    df['vol_dry'] = v < v.rolling(120).mean() * 0.8 # Vol cạn kiệt
    
    # Tín hiệu nổ: Vol đột biến + Giá vượt MA20 + BB mở rộng
    df['is_bomb'] = (df['bb_w'] <= df['bb_w'].rolling(30).min())
    df['is_buy'] = (v > v.rolling(20).mean() * 1.5) & (c > df['ma20']) & (df['ma20'] > df['ma50']) & (df['rsi'] < 75)
    
    # 5. CHẤM ĐIỂM (Thang 10)
    score = 0
    last = df.iloc[-1]
    if last['ma20'] > last['ma50']: score += 2
    if last['rs'] > 0: score += 3
    if last['money_in'] := (last['volume'] > v.rolling(20).mean().iloc[-1]): score += 3
    if last['bb_w'] < 0.05: score += 2 # Nén cực chặt
    df['total_score'] = score

    return df

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏆 V36.0 TERMINAL")
    ticker = st.text_input("🔍 SOI MÃ CHI TIẾT:", "MWG").upper()
    
    if st.button("🚀 CẬP NHẬT DỮ LIỆU CSV", use_container_width=True):
        with st.spinner("Đang ghi đè dữ liệu..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vni_v35.csv")
            
            m_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','DIG','PDR','VHM','VCB','TCB','MBB','STB']
            all_data = []
            for m in m_list:
                t = yf.download(f"{m}.VN", period="2y", progress=False)
                t['symbol'] = m
                all_data.append(t)
            pd.concat(all_data).to_csv("hose_v35.csv")
            st.success("Đã ghi đè vni_v35.csv & hose_v35.csv")
            st.rerun()

    st.divider()
    menu = st.sidebar.selectbox("DANH MỤC CHỨNG NĂNG", ["📊 DÒNG TIỀN NGÀNH", "🎯 LỌC SIÊU ĐIỂM MUA", "📈 ĐỒ THỊ FIREANT"])

# --- HIỂN THỊ CHÍNH ---
if os.path.exists("vni_v35.csv") and os.path.exists("hose_v35.csv"):
    vni_df = pd.read_csv("vni_v35.csv")
    hose_df = pd.read_csv("hose_v35.csv")
    
    # Đánh giá VN-Index
    vni_close = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce')
    vni_rsi = 100 - (100 / (1 + (vni_close.diff().where(vni_close.diff() > 0, 0).rolling(14).mean() / -vni_close.diff().where(vni_close.diff() < 0, 0).rolling(14).mean().replace(0,1))))
    v_status = "🔥 TỐT" if vni_rsi.iloc[-1] < 70 else "⚠️ QUÁ MUA"
    st.metric("SỨC MẠNH VN-INDEX", f"{vni_close.iloc[-1]:.2f}", f"RSI: {vni_rsi.iloc[-1]:.1f} ({v_status})")

    if menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 PHÂN TÍCH DÒNG TIỀN THEO NHÓM NGÀNH")
        nganh_map = {
            "BÁN LẺ": ['MWG','FRT','DGW','MSN'],
            "CHỨNG KHOÁN": ['SSI','VND','VCI'],
            "THÉP": ['HPG','NKG','HSG'],
            "BANK": ['VCB','TCB','MBB','STB']
        }
        sector_results = []
        for n, mãs in nganh_map.items():
            s_scores = []
            for m in mãs:
                d = calculate_master_signals(hose_df[hose_df['symbol'] == m].copy(), vni_df)
                if d is not None: s_scores.append(d['total_score'].iloc[-1])
            avg = np.mean(s_scores) if s_scores else 0
            sector_results.append({"Ngành": n, "Điểm Dòng Tiền": round(avg, 1), "Trạng Thái": "🔥 DẪN DẮT" if avg > 7 else "🔵 TÍCH CỰC"})
        st.table(pd.DataFrame(sector_results).sort_values("Điểm Dòng Tiền", ascending=False))

    elif menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.subheader("🚀 CỔ PHIẾU NỀN GIÁ 6 THÁNG + TIỀN VÀO")
        best_stocks = []
        for s in hose_df['symbol'].unique():
            d = calculate_master_signals(hose_df[hose_df['symbol'] == s].copy(), vni_df)
            if d is not None:
                l = d.iloc[-1]
                # Điều kiện: Nền giá 6 tháng (biên độ < 25%) + BB thắt chặt hoặc Tiền vào
                if l['range_6m'] < 0.25 and (l['is_buy'] or l['is_bomb']):
                    best_stocks.append({"Mã": s, "Điểm": l['total_score'], "RSI": round(l['rsi'],1), "Lý do": "Nền chặt + Vol nổ" if l['is_buy'] else "Nén BB"})
        st.dataframe(pd.DataFrame(best_stocks).sort_values("Điểm", ascending=False))

    elif menu == "📈 ĐỒ THỊ FIREANT":
        df_m = calculate_master_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # 1. NẾN & CHỈ BÁO XU HƯỚNG
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # Tín hiệu 💣 & 🏹
            bomb = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=bomb['date'], y=bomb['high']*1.03, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            buy = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=buy['date'], y=buy['low']*0.97, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # 2. DÒNG TIỀN (VOLUME)
            colors = ['red' if df_m['close'].iloc[i] < df_m['open'].iloc[i] else 'green' for i in range(len(df_m))]
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], marker_color=colors, name="Volume"), row=2, col=1)
            
            # 3. RS & RSI
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], line=dict(color='magenta'), name="RS (Sức mạnh)"), row=3, col=1)
            
            # 4. ADX (Dòng tiền định hướng)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['adx'], line=dict(color='white'), name="ADX"), row=4, col=1)

            # ĐIỀU CHỈNH KÉO DÃN NHƯ FIREANT
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan',
                              yaxis=dict(side='right', fixedrange=False), yaxis2=dict(side='right', fixedrange=False),
                              yaxis3=dict(side='right', fixedrange=False), yaxis4=dict(side='right', fixedrange=False))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'modeBarButtonsToAdd':['drawline','drawrect']})
            
            # QUYẾN NGHỊ
            l = df_m.iloc[-1]
            st.info(f"🚩 **KHUYẾN NGHỊ {ticker}:** Target 1: {l['close']*1.1:,.0f} (+10%) | Target 2: {l['close']*1.2:,.0f} (+20%) | Stoploss: {l['ma50']:,.0f}")
else:
    st.warning("Vui lòng nhấn 'CẬP NHẬT DỮ LIỆU CSV' để khởi tạo hệ thống.")
