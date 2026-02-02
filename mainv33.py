import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V53 - SMART MONEY STABLE", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU CỰC MẠNH (ÉP VỀ 1 CHIỀU) ---
def force_clean_df(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # 1. San phẳng Multi-index (Cốt lõi sửa lỗi TypeError)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Loại bỏ các cột trùng tên (Nếu có)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 3. Đưa Date ra khỏi Index
    df = df.reset_index()
    
    # 4. Chuẩn hóa tên cột
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    rename_map = {
        'date': 'date', 'datetime': 'date', 'ngày': 'date', 'index': 'date',
        'close': 'close', 'adj close': 'close', 'price': 'close',
        'vol': 'volume', 'volume': 'volume',
        'high': 'high', 'low': 'low', 'open': 'open'
    }
    df = df.rename(columns=rename_map)
    
    # Lấy danh sách các cột thực sự tồn tại để ép kiểu số
    cols_to_fix = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    for c in cols_to_fix:
        # Ép về Series 1 chiều để tránh TypeError
        df[c] = pd.to_numeric(df[c].iloc[:, 0] if isinstance(df[c], pd.DataFrame) else df[c], errors='coerce')
            
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['date', 'close']).sort_values('date')

# --- HÀM TÍNH TOÁN (ADX, RSI, RS, SIGNALS) ---
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
    
    # ADX (Simple Strength Index)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift(1)), 
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # RS (Sức mạnh so với VNI)
    if vni_df is not None:
        vni_clean = force_clean_df(vni_df)
        if vni_clean is not None:
            v_c = vni_clean[['date', 'close']].rename(columns={'close': 'v_c'})
            df = pd.merge(df, v_c, on='date', how='left').ffill()
            df['rs'] = (df['close'] / df['close'].shift(20)) / (df['v_c'] / df['v_c'].shift(20))
        else: df['rs'] = 1.0
    else: df['rs'] = 1.0

    # Tín hiệu Mua (⬆️) và Bom tiền (💣)
    v20 = df['volume'].rolling(20).mean()
    df['buy_sig'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb_sig'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- SIDEBAR & DATA UPDATE ---
with st.sidebar:
    st.header("🏆 SUPREME V53")
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU REAL-TIME", use_container_width=True):
        with st.spinner("Đang tải dữ liệu từ Yahoo Finance..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vnindex.csv")
            
            tickers = ['HPG','SSI','MWG','VCB','DIG','VND','FTS','MSN','NKG','HSG','STB','PDR','GEX','VCI','VIX']
            all_data = []
            for t in tickers:
                tmp = yf.download(f"{t}.VN", period="2y")
                tmp['symbol'] = t
                all_data.append(tmp)
            pd.concat(all_data).to_csv("hose.csv")
            st.success("Cập nhật thành công!")
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
            if l['adx'] > 20: score += 3
            if l['close'] > l['ma50']: score += 2
            st.metric("VNI SCORE", f"{score}/10")
            st.progress(score/10)
    
    menu = st.radio("CHUYÊN MỤC:", ["📈 ĐỒ THỊ KỸ THUẬT", "📊 DÒNG TIỀN NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ CHÍNH ---
if os.path.exists("vnindex.csv") and os.path.exists("hose.csv"):
    vni_df = pd.read_csv("vnindex.csv")
    hose_all = pd.read_csv("hose.csv")
    hose_all.columns = [str(c).lower() for c in hose_all.columns]
    
    if menu == "📈 ĐỒ THỊ KỸ THUẬT":
        df_m = hose_all[hose_all['symbol'] == ticker]
        data = calculate_signals(df_m, vni_df)
        
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            
            # Tầng 1: Giá & MA
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            
            # Icon ⬆️ và 💣
            buys = data[data['buy_sig']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="MUA"), row=1, col=1)
            bombs = data[data['bomb_sig']]
            fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.02, mode='markers', marker=dict(symbol='star', size=15, color='red'), name="BOM TIỀN"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Khối lượng", marker_color='dodgerblue'), row=2, col=1)
            
            # Tầng 3: RSI & RS
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta', width=2), name="RS Sức Mạnh"), row=3, col=1)
            
            # Tầng 4: ADX
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX Xu hướng", line=dict(color='white')), row=4, col=1)

            fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Dữ liệu mã này bị lỗi. Vui lòng Update lại.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("📊 PHÂN TÍCH NHÓM DẪN DẮT")
        nganh = {"BANK":['VCB','STB'], "CHỨNG KHOÁN":['SSI','VND','FTS','VCI','VIX'], "THÉP":['HPG','NKG','HSG'], "BĐS":['DIG','PDR','GEX']}
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
            summary.append({"Ngành": n, "Sức Mạnh": np.mean(pts) if pts else 0})
        st.table(pd.DataFrame(summary).sort_values("Sức Mạnh", ascending=False))

    elif menu == "🎯 SIÊU ĐIỂM MUA":
        st.subheader("🎯 BỘ LỌC ĐIỂM MUA TRONG NGÀY")
        res = []
        for m in hose_all['symbol'].unique():
            d = calculate_signals(hose_all[hose_all['symbol'] == m], vni_df)
            if d is not None:
                last = d.iloc[-1]
                if last['bomb_sig'] or last['buy_sig']:
                    res.append({"Mã": m, "Tín hiệu": "💣 BOM TIỀN" if last['bomb_sig'] else "⬆️ MUA", "RS": round(last['rs'],2), "RSI": round(last['rsi'],1)})
        st.dataframe(pd.DataFrame(res).sort_values("RS", ascending=False), use_container_width=True)
else:
    st.info("Chào mừng! Hãy nhấn 'CẬP NHẬT DỮ LIỆU REAL-TIME' để bắt đầu soi mã.")
