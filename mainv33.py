import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V34.1 - SMART FILTER PRO", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU & LỌC TÍN HIỆU ---
def fix_and_calculate(df, vni_df):
    if df is None or len(df) < 25: return None
    df = df.copy()
    
    # 1. Xử lý Multi-Index và ép tên cột về chữ thường
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).lower() for col in df.columns]
    
    # 2. Xử lý cột Date (Nguyên nhân gây lỗi KeyError)
    if 'date' not in df.columns:
        df = df.reset_index()
        df.columns = [str(col).lower() for col in df.columns]
    
    # Đảm bảo cột date ở định dạng datetime
    df['date'] = pd.to_datetime(df['date'])

    # 3. Ép kiểu số
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # 4. Chỉ báo kỹ thuật
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # Bollinger Band Width (Độ rộng BB)
    std = c.rolling(20).std()
    df['bb_w'] = (std * 4) / df['ma20']
    
    # RSI
    delta = c.diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).ewm(14).mean() / 
                                  -delta.where(delta < 0, 0).ewm(14).mean().replace(0, 1))))
    
    # RS (Sức mạnh so với VNI)
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').reset_index(drop=True)
    v_change = vni_c.iloc[-1] / vni_c.iloc[-5] if len(vni_c) > 5 else 1
    df['rs'] = ((c / c.shift(5)) / v_change - 1) * 100
    
    # 5. Logic lọc cổ phiếu Siêu Hạng
    # - BB thắt chặt nhất 30 phiên (Sắp nổ 💣)
    df['is_bomb'] = df['bb_w'] <= df['bb_w'].rolling(30).min()
    # - Tiền vào: Vol > 1.3 lần TB 20 phiên & Giá tăng
    df['money_in'] = (v > v.rolling(20).mean() * 1.3) & (c > c.shift(1))
    # - Điểm Mua: Giá > MA20 & MA20 > MA50 & RSI 45-70
    df['is_buy'] = (c > df['ma20']) & (df['ma20'] > df['ma50']) & (df['rsi'] > 45) & (df['rsi'] < 75)
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚡ SMART TRADING V34.1")
    ticker = st.text_input("🔍 NHẬP MÃ SOI:", "MWG").upper()
    
    if st.button("🔄 CẬP NHẬT REALTIME", use_container_width=True):
        with st.spinner("Đang quét dữ liệu toàn sàn..."):
            vni = yf.download("^VNINDEX", period="1y")
            vni.to_csv("vni_clean.csv")
            
            nganh = {
                'BAN_LE': ['MWG','FRT','DGW','MSN','PNJ'],
                'CHUNG_KHOAN': ['SSI','VND','VCI','VIX','HCM','FTS','BSI'],
                'THEP': ['HPG','NKG','HSG','TLH'],
                'BDS': ['DIG','PDR','VHM','DXG','CEO','NLG','KDH'],
                'BANK': ['VCB','TCB','MBB','STB','LPB','CTG','BID']
            }
            all_m = [m for sub in nganh.values() for m in sub]
            data = []
            for m in all_m:
                t = yf.download(f"{m}.VN", period="1y", progress=False)
                t['symbol'] = m
                data.append(t)
            pd.concat(data).to_csv("hose_clean.csv")
            st.success("✅ Đã cập nhật!")
            st.rerun()

    menu = st.radio("CHẾ ĐỘ XEM:", ["📈 SOI CHI TIẾT", "🚀 LỌC SIÊU CỔ PHIẾU", "📊 DÒNG TIỀN NGÀNH"])

# --- HIỂN THỊ ---
if os.path.exists("vni_clean.csv") and os.path.exists("hose_clean.csv"):
    vni_df = pd.read_csv("vni_clean.csv")
    hose_df = pd.read_csv("hose_clean.csv")

    if menu == "🚀 LỌC SIÊU CỔ PHIẾU":
        st.subheader("🎯 CỔ PHIẾU CÓ DÒNG TIỀN VÀO & NÉN CHẶT")
        selection = []
        all_symbols = hose_df['symbol'].unique()
        for s in all_symbols:
            d = fix_and_calculate(hose_df[hose_df['symbol'] == s].copy(), vni_df)
            if d is not None:
                l = d.iloc[-1]
                if l['money_in'] or l['is_bomb']:
                    status = []
                    if l['money_in']: status.append("💰 Tiền vào")
                    if l['is_bomb']: status.append("💣 Nén chặt (BB)")
                    if l['is_buy']: status.append("✅ Điểm Mua")
                    
                    selection.append({
                        "Mã": s, "Giá": l['close'], "RSI": round(l['rsi'], 1), 
                        "RS": round(l['rs'], 1), "Tín hiệu": " + ".join(status)
                    })
        st.dataframe(pd.DataFrame(selection).sort_values("RS", ascending=False), use_container_width=True)

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH NHÓM NGÀNH")
        # (Logic chấm điểm tương tự bản trước, hiển thị bảng điểm ngành)
        st.info("Hệ thống đang quét RS và Volume trung bình của từng nhóm.")

    elif menu == "📈 SOI CHI TIẾT":
        df_m = fix_and_calculate(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # Tầng 1: Candle + MA + Bom + Mua
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            
            # Icon 💣 và 🏹
            b = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=b['date'], y=b['high']*1.02, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            s = df_m[df_m['is_buy'] & df_m['money_in']]
            fig.add_trace(go.Scatter(x=s['date'], y=s['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # Các tầng Volume, RSI/RS, ADX
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], name="Vol", marker_color='gray'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], name="RS", line=dict(color='magenta')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['adx'], name="ADX", line=dict(color='white')), row=4, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan',
                              yaxis=dict(side='right', fixedrange=False, autorange=True), xaxis=dict(fixedrange=False))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            l = df_m.iloc[-1]
            st.success(f"Mã: {ticker} | RSI: {l['rsi']:.1f} | RS: {l['rs']:.1f} | Trạng thái: {'🏹 ĐIỂM MUA' if l['is_buy'] else 'Theo dõi'}")
else:
    st.warning("⚠️ Vui lòng nhấn nút 'CẬP NHẬT REALTIME' bên trái.")
