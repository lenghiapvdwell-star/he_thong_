import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V45 - MASTER STRATEGY", layout="wide")

# --- HÀM TẢI DỮ LIỆU THÔNG MINH ---
def load_data(file_name):
    if not os.path.exists(file_name): return None
    try:
        df = pd.read_csv(file_name)
        df.columns = [str(c).strip().lower() for c in df.columns]
        # Tìm cột Symbol/Ticker
        for col in ['symbol', 'ticker', 'mã', 'ma']:
            if col in df.columns: 
                df = df.rename(columns={col: 'symbol'})
                break
        # Tìm cột Date
        for col in ['date', 'ngày', 'time']:
            if col in df.columns: 
                df = df.rename(columns={col: 'date'})
                break
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Ép kiểu số ngay từ đầu cho các cột quan trọng
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df = df.dropna(subset=['date', 'close']).sort_values('date')
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
        return df
    except:
        return None

# --- HÀM TÍNH TOÁN CHỈ BÁO CHI TIẾT ---
def calculate_pro_signals(df, vni_df=None):
    if df is None or len(df) < 10: return None
    df = df.copy().sort_values('date')

    # 1. Đường trung bình MA
    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(50, min_periods=1).mean()

    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))

    # 3. ADX (Sức mạnh xu hướng)
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift()), 
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)) * 100
    df['adx'] = dx.rolling(14, min_periods=1).mean()

    # 4. RS (Sức mạnh tương quan vs VNI)
    if vni_df is not None:
        # Làm sạch dữ liệu VNI để merge
        vni_clean = vni_df[['date', 'close']].rename(columns={'close': 'vni_close'})
        df = pd.merge(df, vni_clean, on='date', how='left')
        df['vni_close'] = df['vni_close'].ffill()
        # Công suất RS = (Giá CP / Giá VNI)
        df['rs'] = (df['close'] / df['close'].shift(20)) / (df['vni_close'] / df['vni_close'].shift(20))
    if 'rs' not in df.columns: df['rs'] = 1.0

    # 5. Tín hiệu đặc biệt
    df['vol20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['buy_signal'] = (df['close'] > df['ma20']) & (df['volume'] > df['vol20'] * 1.3)
    df['money_bomb'] = (df['volume'] > df['vol20'] * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- TẢI DỮ LIỆU ---
hose_df = load_data("hose.csv")
vni_df = load_data("vnindex.csv")

# --- GIAO DIỆN SIDEBAR ---
with st.sidebar:
    st.header("🏆 SUPREME V45")
    ticker = st.text_input("🔍 MÃ SOI:", "HPG").upper()
    
    st.divider()
    st.subheader("📊 SỨC KHỎE THỊ TRƯỜNG")
    btn_vni = st.button("📈 KIỂM TRA VN-INDEX", use_container_width=True)
    
    if btn_vni:
        if vni_df is not None:
            v_data = calculate_pro_signals(vni_df)
            if v_data is not None:
                last_v = v_data.iloc[-1]
                # Chấm điểm VNI thang 10
                v_score = 0
                if last_v['close'] > last_v['ma20']: v_score += 3
                if last_v['close'] > last_v['ma50']: v_score += 2
                if last_v['rsi'] > 50: v_score += 2
                if last_v['adx'] > 25: v_score += 3
                
                st.metric("CHẤM ĐIỂM VNI", f"{v_score}/10")
                st.write(f"**RSI:** {round(last_v['rsi'], 1)}")
                st.write(f"**ADX:** {round(last_v['adx'], 1)}")
                
                if v_score >= 7: st.success("🚀 THỊ TRƯỜNG MẠNH: Ưu tiên mua.")
                elif v_score >= 5: st.warning("⚖️ THỊ TRƯỜNG TÍCH LŨY: Mua thăm dò.")
                else: st.error("⚠️ THỊ TRƯỜNG YẾU: Đứng ngoài.")
            else: st.error("Lỗi tính toán dữ liệu VNI.")
        else:
            st.error("❌ Không tìm thấy file 'vnindex.csv'!")
    
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ CHI TIẾT", "📊 DÒNG TIỀN NGÀNH", "🎯 LỌC SIÊU ĐIỂM MUA"])

# --- HIỂN THỊ CHÍNH ---
if hose_df is not None:
    if menu == "📈 ĐỒ THỊ CHI TIẾT":
        ticker_data = hose_df[hose_df['symbol'] == ticker]
        data = calculate_pro_signals(ticker_data, vni_df)
        
        if data is not None:
            st.subheader(f"📊 PHÂN TÍCH SMART MONEY: {ticker}")
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                               row_heights=[0.5, 0.15, 0.15, 0.2])
            
            # Tầng 1: Candle + MA + Icons
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], 
                                         low=data['low'], close=data['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
            
            # Vẽ Mũi tên và Bom
            buys = data[data['buy_signal']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.98, mode='markers+text', 
                                     text="⬆️", textposition="bottom center", 
                                     marker=dict(size=12, color='lime'), name="MUA"), row=1, col=1)
            
            bombs = data[data['money_bomb']]
            fig.add_trace(go.Scatter(x=bombs['date'], y=bombs['high']*1.02, mode='markers+text', 
                                     text="💣", textposition="top center", 
                                     marker=dict(size=15, color='red'), name="BOM TIỀN"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Khối lượng", marker_color='rgba(100, 149, 237, 0.6)'), row=2, col=1)
            
            # Tầng 3: RSI & RS
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, line=dict(color='magenta'), name="RS (Relative)"), row=3, col=1)
            
            # Tầng 4: ADX
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX (Trend Strength)"), row=4, col=1)

            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        else:
            st.error(f"Dữ liệu mã {ticker} bị thiếu hoặc file hose.csv sai định dạng.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN NHÓM NGÀNH")
        nganh_dict = {
            "THÉP":['HPG','NKG','HSG'], 
            "BANK":['VCB','TCB','MBB','STB'], 
            "CHỨNG KHOÁN":['SSI','VND','VCI','VIX'], 
            "BĐS":['DIG','PDR','VHM','GEX'], 
            "BÁN LẺ":['MWG','FRT','DGW','MSN']
        }
        summary = []
        for n, mãs in nganh_dict.items():
            scores = []
            for m in mãs:
                d = calculate_pro_signals(hose_df[hose_df['symbol'] == m], vni_df)
                if d is not None:
                    s = 0
                    l = d.iloc[-1]
                    if l['close'] > l['ma20']: s += 5
                    if l['money_bomb'] or l['buy_signal']: s += 5
                    scores.append(s)
            summary.append({"Ngành": n, "Sức Mạnh (0-10)": round(np.mean(scores), 1) if scores else 0})
        st.table(pd.DataFrame(summary).sort_values("Sức Mạnh (0-10)", ascending=False))

    elif menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.subheader("🚀 QUÉT CỔ PHIẾU CÓ DÒNG TIỀN TỔ CHỨC")
        results = []
        unique_mãs = hose_df['symbol'].unique()
        for s in unique_mãs:
            d = calculate_pro_signals(hose_df[hose_df['symbol'] == s], vni_df)
            if d is not None:
                l = d.iloc[-1]
                if l['money_bomb'] or l['buy_signal']:
                    results.append({
                        "Mã": s, 
                        "Tín hiệu": "💣 BOM TIỀN" if l['money_bomb'] else "⬆️ ĐIỂM MUA", 
                        "RSI": round(l['rsi'], 1), 
                        "RS": round(l['rs'], 2)
                    })
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("RS", ascending=False), use_container_width=True)
        else:
            st.info("Chưa tìm thấy mã nào bùng nổ hôm nay.")
else:
    st.error("❌ Không tìm thấy file 'hose.csv'! Vui lòng kiểm tra lại thư mục lưu trữ.")
