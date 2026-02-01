import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V34.2 - IRONCLAD PRO", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU THÔNG MINH ---
def fix_and_calculate(df, vni_df):
    if df is None or len(df) < 30: return None
    df = df.copy()
    
    # 1. Phá bỏ Multi-Index và đưa về chữ thường
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    # 2. CƠ CHẾ DÒ CỘT DATE (Sửa lỗi KeyError)
    # Thử tìm các tên phổ biến, nếu không thấy thì reset_index
    possible_date_cols = ['date', 'datetime', 'index', 'unnamed: 0']
    found_date = False
    for col in possible_date_cols:
        if col in df.columns:
            df = df.rename(columns={col: 'date'})
            found_date = True
            break
    
    if not found_date:
        df = df.reset_index().rename(columns={df.index.name if df.index.name else 'index': 'date'})
        df.columns = [str(col).lower() for col in df.columns]

    # 3. Ép kiểu và chuẩn hóa dữ liệu
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # 4. Chỉ báo kỹ thuật
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # Bollinger Band Width
    std = c.rolling(20).std()
    df['bb_w'] = (std * 4) / df['ma20']
    
    # RSI & RS
    delta = c.diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).ewm(14).mean() / 
                                  -delta.where(delta < 0, 0).ewm(14).mean().replace(0, 1))))
    
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').reset_index(drop=True)
    vni_change = vni_c.iloc[-1] / vni_c.iloc[-5] if len(vni_c) > 5 else 1
    df['rs'] = ((c / c.shift(5)) / vni_change - 1) * 100
    
    # 5. Logic lọc Siêu Cổ (Money In & BB Squeeze)
    df['is_bomb'] = df['bb_w'] <= df['bb_w'].rolling(30).min()
    df['money_in'] = (v > v.rolling(20).mean() * 1.3) & (c > c.shift(1))
    df['is_buy'] = (c > df['ma20']) & (df['ma20'] > df['ma50']) & (df['rsi'] > 45)
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚡ IRONCLAD V34.2")
    ticker = st.text_input("🔍 NHẬP MÃ SOI:", "MWG").upper()
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU", use_container_width=True):
        with st.spinner("Đang tải dữ liệu..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vni_v34.csv")
            
            nganh = {
                'BAN_LE': ['MWG','FRT','DGW','MSN','PNJ'],
                'CHUNG_KHOAN': ['SSI','VND','VCI','VIX','HCM','FTS'],
                'THEP': ['HPG','NKG','HSG'],
                'BDS': ['DIG','PDR','VHM','DXG','CEO','NLG'],
                'BANK': ['VCB','TCB','MBB','STB','LPB','CTG']
            }
            all_m = [m for sub in nganh.values() for m in sub]
            data = []
            for m in all_m:
                t = yf.download(f"{m}.VN", period="2y", progress=False)
                t['symbol'] = m
                data.append(t)
            pd.concat(data).to_csv("hose_v34.csv")
            st.success("✅ Đã cập nhật xong!")
            st.rerun()

    menu = st.radio("CHẾ ĐỘ XEM:", ["📈 SOI CHI TIẾT", "🚀 LỌC SIÊU CỔ PHIẾU", "📊 DÒNG TIỀN NGÀNH"])

# --- HIỂN THỊ ---
if os.path.exists("vni_v34.csv") and os.path.exists("hose_v34.csv"):
    vni_raw = pd.read_csv("vni_v34.csv")
    hose_raw = pd.read_csv("hose_v34.csv")

    if menu == "🚀 LỌC SIÊU CỔ PHIẾU":
        st.subheader("🎯 DANH SÁCH CỔ PHIẾU CÓ DÒNG TIỀN & NÉN CHẶT")
        findings = []
        for s in hose_raw['symbol'].unique():
            df_s = fix_and_calculate(hose_raw[hose_raw['symbol'] == s].copy(), vni_raw)
            if df_s is not None:
                l = df_s.iloc[-1]
                if l['money_in'] or l['is_bomb'] or l['is_buy']:
                    label = []
                    if l['money_in']: label.append("💰 Tiền vào")
                    if l['is_bomb']: label.append("💣 Nén chặt")
                    if l['is_buy']: label.append("✅ Xu hướng tốt")
                    findings.append({"Mã": s, "Giá": l['close'], "RSI": round(l['rsi'],1), "RS": round(l['rs'],1), "Tín hiệu": " + ".join(label)})
        st.dataframe(pd.DataFrame(findings).sort_values("RS", ascending=False), use_container_width=True)

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("📊 SỨC MẠNH DÒNG TIỀN THEO NHÓM NGÀNH")
        # Code tính điểm ngành tương tự logic trên...
        st.info("Bảng điểm đang được cập nhật dựa trên RS và Volume...")

    elif menu == "📈 SOI CHI TIẾT":
        df_m = fix_and_calculate(hose_raw[hose_raw['symbol'] == ticker].copy(), vni_raw)
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            
            # Tín hiệu đồ họa
            b = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=b['date'], y=b['high']*1.02, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            s = df_m[df_m['is_buy'] & df_m['money_in']]
            fig.add_trace(go.Scatter(x=s['date'], y=s['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], name="Vol", marker_color='gray'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], name="RS", line=dict(color='magenta')), row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan',
                              yaxis=dict(side='right', fixedrange=False, autorange=True), xaxis=dict(fixedrange=False))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            l = df_m.iloc[-1]
            st.success(f"Mã: {ticker} | RSI: {l['rsi']:.1f} | RS: {l['rs']:.1f} | Trạng thái: {'🏹 ĐIỂM MUA' if l['is_buy'] else 'Quan sát'}")
else:
    st.warning("⚠️ Nhấn 'CẬP NHẬT DỮ LIỆU' để bắt đầu.")
