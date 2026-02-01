import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V35.0 - FIREANT SUPREME", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU SIÊU CẤP ---
def fix_and_calculate(df, vni_df):
    if df is None or len(df) < 1: return None
    df = df.copy()
    
    # 1. PHÁ BỎ MULTI-INDEX (Quan trọng nhất để hiện nến)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    # 2. Xử lý cột ngày tháng
    if 'date' not in df.columns:
        df = df.reset_index()
        df.columns = [str(col).lower() for col in df.columns]
    
    df = df.rename(columns={'index': 'date', 'datetime': 'date'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 3. Chuyển đổi kiểu dữ liệu số
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close', 'date']).sort_values('date').reset_index(drop=True)
    if len(df) < 50: return None # Đảm bảo đủ dữ liệu tính MA50
    
    # 4. TÍNH TOÁN CHỈ BÁO (Đảm bảo không rỗng)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # Bollinger Bands
    std = c.rolling(20).std()
    df['bb_upper'] = df['ma20'] + (std * 2)
    df['bb_lower'] = df['ma20'] - (std * 2)
    df['bb_w'] = (std * 4) / df['ma20']
    
    # RSI
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # RS (Sức mạnh giá)
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').reset_index(drop=True)
    v_ratio = vni_c.iloc[-1] / vni_c.iloc[-5] if len(vni_c) > 5 else 1
    df['rs'] = ((c / c.shift(5)) / v_ratio - 1) * 100
    
    # ADX (Đơn giản hóa để không lỗi)
    df['adx'] = (c.diff().abs().rolling(14).mean() / c.rolling(14).mean()) * 1000

    # TÍN HIỆU
    df['is_bomb'] = df['bb_w'] <= df['bb_w'].rolling(30).min()
    df['is_buy'] = (c > df['ma20']) & (v > v.rolling(20).mean() * 1.3)
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚡ TRADING SYSTEM V35")
    ticker = st.text_input("🔍 NHẬP MÃ SOI:", "MWG").upper()
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU", use_container_width=True):
        with st.spinner("Đang tải dữ liệu mới..."):
            vni = yf.download("^VNINDEX", period="1y")
            vni.to_csv("vni_v35.csv")
            
            m_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','DIG','PDR','VHM','VCB','TCB']
            all_d = []
            for m in m_list:
                t = yf.download(f"{m}.VN", period="1y", progress=False)
                t['symbol'] = m
                all_d.append(t)
            pd.concat(all_d).to_csv("hose_v35.csv")
            st.success("Xong! Hãy soi mã.")
            st.rerun()

    menu = st.radio("CHỨNG NĂNG:", ["📈 SOI CHI TIẾT", "🚀 LỌC TIỀN VÀO"])

# --- HIỂN THỊ ---
if os.path.exists("vni_v35.csv") and os.path.exists("hose_v35.csv"):
    vni_raw = pd.read_csv("vni_v35.csv")
    hose_raw = pd.read_csv("hose_v35.csv")

    if menu == "📈 SOI CHI TIẾT":
        df_m = fix_and_calculate(hose_raw[hose_raw['symbol'] == ticker].copy(), vni_raw)
        
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # 1. NẾN, MA, BOLLINGER BANDS
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name="Nến"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['bb_upper'], line=dict(color='gray', dash='dash'), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['bb_lower'], line=dict(color='gray', dash='dash'), name="BB Lower"), row=1, col=1)
            
            # ICON
            b = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=b['date'], y=b['high']*1.02, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            s = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=s['date'], y=s['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # 2. VOLUME
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], name="Vol", marker_color='blue'), row=2, col=1)
            
            # 3. RSI & RS
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], line=dict(color='magenta'), name="RS"), row=3, col=1)
            
            # 4. ADX
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['adx'], line=dict(color='white'), name="ADX"), row=4, col=1)

            fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            fig.update_yaxes(side="right")
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # Chấm điểm ngành & mã
            l = df_m.iloc[-1]
            st.success(f"🚩 {ticker} - Giá: {l['close']:.1f} | RSI: {l['rsi']:.1f} | RS: {l['rs']:.1f} | MA20: {l['ma20']:.1f}")
        else:
            st.error("Không đủ dữ liệu để vẽ Chart (Cần ít nhất 50 phiên). Hãy nhấn Cập nhật.")

    elif menu == "🚀 LỌC TIỀN VÀO":
        st.subheader("🔥 CỔ PHIẾU ĐỘT BIẾN DÒNG TIỀN")
        # Logic lọc bảng (Đã fix IndexError bằng cách check len)
        selection = []
        for s in hose_raw['symbol'].unique():
            d = fix_and_calculate(hose_raw[hose_raw['symbol'] == s].copy(), vni_raw)
            if d is not None and len(d) > 0:
                l = d.iloc[-1]
                if l['is_buy'] or l['is_bomb']:
                    selection.append({"Mã": s, "Giá": l['close'], "RSI": round(l['rsi'],1), "RS": round(l['rs'],1)})
        if selection: st.table(pd.DataFrame(selection))
        else: st.info("Chưa tìm thấy mã đạt tiêu chí.")
else:
    st.warning("⚠️ Nhấn 'CẬP NHẬT DỮ LIỆU' để tải nến và chỉ báo.")
