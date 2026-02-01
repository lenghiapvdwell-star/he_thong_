import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V34.0 - MONEY FLOW PRO", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU CỰC MẠNH ---
def fix_and_calculate(df, vni_df):
    if df is None or len(df) < 10: return None
    df = df.copy()
    
    # 1. Xử lý Multi-Index và ép tên cột về chữ thường
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).lower() for col in df.columns]
    
    # 2. Đảm bảo có cột Date
    if 'date' not in df.columns:
        df = df.reset_index()
        df.columns = [str(col).lower() for col in df.columns]

    # 3. Ép kiểu số cho các cột chính
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # 4. Chỉ báo kỹ thuật
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # RSI
    delta = c.diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).ewm(14).mean() / 
                                  -delta.where(delta < 0, 0).ewm(14).mean().replace(0, 1))))
    
    # RS (Sức mạnh giá)
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').reset_index(drop=True)
    vni_change = vni_c.iloc[-1] / vni_c.iloc[-5] if len(vni_c) > 5 else 1
    df['rs'] = ((c / c.shift(5)) / vni_change - 1) * 100
    
    # ADX
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(14).mean()
    pdm = pd.Series(np.where((h.diff()>l.shift(1)-l)&(h.diff()>0), h.diff(), 0))
    pdi = 100 * (pdm.ewm(14).mean() / atr)
    mdm = pd.Series(np.where((l.shift(1)-l>h.diff())&(l.shift(1)-l>0), l.shift(1)-l, 0))
    mdi = 100 * (mdm.ewm(14).mean() / atr)
    df['adx'] = (100 * (abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan))).ewm(14).mean()

    # 5. Tín hiệu Bom & Mua
    std = c.rolling(20).std()
    df['is_bomb'] = ((std * 4) / df['ma20']) <= ((std * 4) / df['ma20']).rolling(30).min()
    df['is_buy'] = (c > df['ma20']) & (df['ma20'] > df['ma50']) & (v > v.rolling(20).mean() * 1.3)
    
    # 6. Chấm điểm
    score = 0
    last = df.iloc[-1]
    if last['close'] > last['ma20']: score += 2
    if last['ma20'] > last['ma50']: score += 2
    if last['rs'] > 0: score += 3
    if last['volume'] > v.rolling(20).mean().iloc[-1]: score += 3
    df['score'] = score
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("💎 HỆ THỐNG V34.0")
    ticker = st.text_input("🔍 NHẬP MÃ SOI:", "MWG").upper()
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU", use_container_width=True):
        with st.spinner("Đang quét thị trường..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vni_clean.csv")
            
            # Danh sách ngành
            nganh = {
                'BAN_LE': ['MWG','FRT','DGW','MSN'],
                'CHUNG_KHOAN': ['SSI','VND','VCI','VIX','HCM'],
                'THEP': ['HPG','NKG','HSG'],
                'BDS': ['DIG','PDR','VHM','DXG','CEO'],
                'BANK': ['VCB','TCB','MBB','STB','LPB']
            }
            all_m = [m for sub in nganh.values() for m in sub]
            data = []
            for m in all_m:
                t = yf.download(f"{m}.VN", period="2y", progress=False)
                t['symbol'] = m
                data.append(t)
            pd.concat(data).to_csv("hose_clean.csv")
            st.success("✅ Đã xong!")
            st.rerun()

    menu = st.radio("CHẾ ĐỘ:", ["📈 SOI CHI TIẾT", "📊 DÒNG TIỀN NGÀNH"])

# --- HIỂN THỊ ---
if os.path.exists("vni_clean.csv"):
    vni_df = pd.read_csv("vni_clean.csv")
    hose_df = pd.read_csv("hose_clean.csv")

    if menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 BẢNG CHẤM ĐIỂM DÒNG TIỀN NGÀNH")
        nganh_dict = {"BÁN LẺ": ['MWG','FRT','DGW','MSN'], "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','HCM'], "THÉP": ['HPG','NKG','HSG'], "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','DXG','CEO'], "NGÂN HÀNG": ['VCB','TCB','MBB','STB','LPB']}
        
        results = []
        for n_name, mãs in nganh_dict.items():
            pts = []
            for m in mãs:
                m_data = fix_and_calculate(hose_df[hose_df['symbol'] == m].copy(), vni_df)
                if m_data is not None: pts.append(m_data['score'].iloc[-1])
            avg = np.mean(pts) if pts else 0
            tt = "🔥 MẠNH" if avg >= 7 else "✅ KHÁ" if avg >= 5 else "☁️ YẾU"
            results.append({"Ngành": n_name, "Điểm": round(avg, 1), "Trạng Thái": tt})
        
        st.table(pd.DataFrame(results).sort_values("Điểm", ascending=False))

    elif menu == "📈 SOI CHI TIẾT":
        df_m = fix_and_calculate(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # Tầng 1: Giá, Bom, Mua
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            
            # Tín hiệu
            b = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=b['date'], y=b['high']*1.03, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            s = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=s['date'], y=s['low']*0.97, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # Tầng 2: Vol
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], name="Volume", marker_color='gray'), row=2, col=1)
            # Tầng 3: RSI & RS
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], name="RS", line=dict(color='magenta')), row=3, col=1)
            # Tầng 4: ADX
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['adx'], name="ADX", line=dict(color='white')), row=4, col=1)

            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan',
                              yaxis=dict(side='right', fixedrange=False, autorange=True), xaxis=dict(fixedrange=False))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            last = df_m.iloc[-1]
            st.success(f"🚩 Mã: {ticker} | Điểm: {last['score']}/10 | Target: {last['close']*1.12:,.0f}")
else:
    st.info("👋 Nhấn 'CẬP NHẬT DỮ LIỆU' bên trái để bắt đầu.")
