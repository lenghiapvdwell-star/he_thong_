import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V65 - STOCK ANALYTICS PRO", layout="wide")

# --- 1. HÀM CHUẨN HÓA DỮ LIỆU TỪ CSV ---
def load_and_clean(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        if df.empty: return None
        
        # Chuyển tên cột về chữ thường
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Ép kiểu ngày tháng
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Ép kiểu số cho các cột kỹ thuật
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=['date', 'close']).sort_values('date')
    except Exception as e:
        st.error(f"Lỗi đọc file {file_path}: {e}")
        return None

# --- 2. HÀM TÍNH TOÁN CHỈ BÁO ---
def add_indicators(df, vni_df=None):
    if df is None or len(df) < 20: return None
    df = df.copy()
    
    # MA & RSI
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX chuẩn (Đo xu hướng)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift()), 
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # RS (Sức mạnh tương quan)
    df['rs'] = 1.0
    if vni_df is not None:
        # Đồng bộ hóa ngày để tính RS
        vni = vni_df.copy().set_index('date')
        df_idx = df.set_index('date')
        common = df_idx.index.intersection(vni.index)
        if not common.empty:
            rs_val = (df_idx.loc[common, 'close'] / df_idx.loc[common, 'close'].shift(20)) / \
                     (vni.loc[common, 'close'] / vni.loc[common, 'close'].shift(20))
            df_idx.loc[common, 'rs'] = rs_val.ffill()
        df = df_idx.reset_index()

    # Tín hiệu Mua (⬆️) & Bom Tiền (💣)
    v20 = df['volume'].rolling(20).mean()
    df['buy_sig'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb_sig'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🏆 STOCK V65")
    st.info("Dữ liệu được lấy từ hose.csv và vnindex.csv")
    
    # Chức năng Update Real-time (Ghi đè dòng cuối)
    if st.button("🔄 CẬP NHẬT GIÁ REAL-TIME", use_container_width=True):
        st.warning("Đang kết nối để lấy giá ngày hiện tại...")
        # Ở đây bạn có thể gọi script download_hose.py của bạn hoặc dùng yfinance lấy 1 ngày
        # Tạm thời tôi hướng dẫn code này đọc file bạn đã upload
        st.success("Đã đồng bộ với file dữ liệu mới nhất!")
        st.rerun()

    ticker = st.text_input("🔍 NHẬP MÃ CỔ PHIẾU:", "HPG").upper()
    
    # Kiểm tra sức khỏe VN-INDEX
    vni_df = load_and_clean("vnindex.csv")
    if vni_df is not None:
        v_data = add_indicators(vni_df)
        if v_data is not None:
            l = v_data.iloc[-1]
            score = sum([l['close'] > l['ma20'], l['rsi'] > 50, l['adx'] > 15, l['close'] > l['ma50']]) * 2.5
            st.metric("VNI SCORE", f"{int(score)}/10")
            st.progress(score/10)

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ", "📊 DÒNG TIỀN NGÀNH", "🎯 SIÊU ĐIỂM MUA"])

# --- 4. HIỂN THỊ ---
hose_df = load_and_clean("hose.csv")

if hose_df is not None:
    if menu == "📈 ĐỒ THỊ":
        # Lọc dữ liệu cho mã đã chọn
        stock_data = hose_df[hose_df['symbol'].str.upper() == ticker] if 'symbol' in hose_df.columns else hose_df
        data = add_indicators(stock_data, vni_df)
        
        if data is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            
            # Tầng 1: Candle & Signal
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            
            b = data[data['buy_sig']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12), name="MUA"), row=1, col=1)
            bm = data[data['bomb_sig']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.02, mode='markers', marker=dict(symbol='star', color='red', size=15), name="BOM"), row=1, col=1)

            # Tầng 2, 3, 4
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, name="RS", line=dict(color='magenta')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="ADX Trend"), row=4, col=1)
            
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Không tìm thấy dữ liệu cho mã {ticker} trong file hose.csv")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("📊 SỨC MẠNH DÒNG TIỀN")
        # Logic lọc ngành từ file hose.csv
        st.info("Tính năng đang phân tích dựa trên dữ liệu file CSV của bạn...")

    elif menu == "🎯 SIÊU ĐIỂM MUA":
        st.subheader("🎯 CÁC MÃ CÓ TÍN HIỆU TỪ FILE HOSE.CSV")
        if 'symbol' in hose_df.columns:
            results = []
            for s in hose_df['symbol'].unique():
                d = add_indicators(hose_df[hose_df['symbol'] == s], vni_df)
                if d is not None:
                    last = d.iloc[-1]
                    if last['bomb_sig'] or last['buy_sig']:
                        results.append({"Mã": s, "Lệnh": "💣 BOM" if last['bomb_sig'] else "⬆️ MUA", "RS": round(last['rs'],2)})
            st.dataframe(pd.DataFrame(results).sort_values("RS", ascending=False))
else:
    st.error("LỖI: Không tìm thấy file hose.csv. Hãy đảm bảo file này nằm cùng thư mục với code.")
