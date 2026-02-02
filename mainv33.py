import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V67 - FINAL ALIGN 2026", layout="wide")

# --- 1. BỘ GIẢI MÃ CSV VẠN NĂNG ---
def universal_loader(file_path):
    if not os.path.exists(file_path):
        st.error(f"Không tìm thấy file: {file_path}")
        return None
    try:
        # Đọc file với cơ chế tự đoán dấu phân cách
        df = pd.read_csv(file_path, sep=None, engine='python')
        if df.empty: return None
        
        # Xử lý Multi-index (nếu có tầng 0 là Ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
            
        # Chuẩn hóa tên cột: viết thường, xóa khoảng trắng
        df.columns = [str(c).strip().lower() for c in df.columns]

        # --- TÌM CỘT NGÀY THÁNG ---
        # Ưu tiên các cột có tên phổ biến, nếu không lấy cột 0
        date_candidates = ['date', 'datetime', 'ngày', 'time', 'timestamp']
        target_date_col = next((c for c in df.columns if any(p in c for p in date_candidates)), df.columns[0])
        df = df.rename(columns={target_date_col: 'date'})
        
        # Chuyển đổi ngày (xử lý cả định dạng số timestamp hoặc chuỗi)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # --- TÌM CỘT GIÁ ---
        # Tự động map các cột dựa trên từ khóa
        col_map = {
            'close': ['close', 'đóng', 'last', 'adj'],
            'open': ['open', 'mở'],
            'high': ['high', 'cao'],
            'low': ['low', 'thấp'],
            'volume': ['vol', 'khối', 'amount']
        }
        
        for standard_name, keywords in col_map.items():
            found_col = next((c for c in df.columns if any(k in c for k in keywords)), None)
            if found_col:
                df = df.rename(columns={found_col: standard_name})
                df[standard_name] = pd.to_numeric(df[standard_name], errors='coerce')

        return df.dropna(subset=['date', 'close']).sort_values('date').drop_duplicates('date')
    except Exception as e:
        st.error(f"Lỗi cấu trúc file {file_path}: {e}")
        return None

# --- 2. HÀM TÍNH TOÁN SMART SIGNALS ---
def compute_signals(stock_df, vni_df=None):
    if stock_df is None or len(stock_df) < 20: return None
    df = stock_df.copy()
    
    # Chỉ báo xu hướng
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    
    # RSI
    diff = df['close'].diff()
    gain = (diff.where(diff > 0, 0)).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # ADX (Đo lực xu hướng)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift()), 
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['adx'] = (tr.rolling(14).mean() / df['close'] * 500).rolling(14).mean()

    # RS (Sức mạnh tương quan - CỰC QUAN TRỌNG)
    df['rs'] = 1.0
    if vni_df is not None:
        vni = vni_df.copy().set_index('date')
        df_idx = df.set_index('date')
        common = df_idx.index.intersection(vni.index)
        if not common.empty:
            # So sánh hiệu suất 20 phiên của CP so với VNI
            s_perf = df_idx.loc[common, 'close'] / df_idx.loc[common, 'close'].shift(20)
            v_perf = vni.loc[common, 'close'] / vni.loc[common, 'close'].shift(20)
            df_idx.loc[common, 'rs'] = (s_perf / v_perf).ffill()
        df = df_idx.reset_index()

    # Tín hiệu Mua & Bom tiền (Dòng tiền vào mạnh)
    v20 = df['volume'].rolling(20).mean()
    df['buy'] = (df['close'] > df['ma20']) & (df['volume'] > v20 * 1.3)
    df['bomb'] = (df['volume'] > v20 * 2.2) & (df['close'] > df['close'].shift(1) * 1.03)
    
    return df

# --- 3. SIDEBAR & ĐIỀU KHIỂN ---
with st.sidebar:
    st.header("🏆 SMART MONEY PRO")
    st.markdown("---")
    
    # Đọc VNINDEX trước để lấy thông tin thị trường
    vni_data_raw = universal_loader("vnindex.csv")
    vni_final = compute_signals(vni_data_raw)
    
    if vni_final is not None:
        curr = vni_final.iloc[-1]
        score = sum([curr['close'] > curr['ma20'], curr['rsi'] > 50, curr['adx'] > 15, curr['close'] > curr['ma50']]) * 2.5
        st.metric("VNI HEALTH SCORE", f"{int(score)}/10")
        st.progress(score/10)
    
    ticker = st.text_input("🔍 SOI MÃ (VD: HPG, SSI):", "HPG").upper()
    menu = st.radio("CHUYÊN MỤC:", ["📈 ĐỒ THỊ KỸ THUẬT", "🎯 SIÊU ĐIỂM MUA"])

# --- 4. KHÔNG GIAN HIỂN THỊ CHÍNH ---
hose_data_raw = universal_loader("hose.csv")

if hose_data_raw is not None:
    if menu == "📈 ĐỒ THỊ KỸ THUẬT":
        # Tách dữ liệu mã cổ phiếu
        if 'symbol' in hose_data_raw.columns:
            stock_df = hose_data_raw[hose_data_raw['symbol'].str.upper() == ticker]
        else:
            stock_df = hose_data_raw # Giả sử file chỉ có 1 mã nếu ko có cột symbol
            
        data = compute_signals(stock_df, vni_data_raw)
        
        if data is not None and not data.empty:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.2, 0.25])
            
            # Tầng 1: Candle & MA
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            
            # Tín hiệu
            b = data[data['buy']]; fig.add_trace(go.Scatter(x=b['date'], y=b['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12), name="MUA"), row=1, col=1)
            bm = data[data['bomb']]; fig.add_trace(go.Scatter(x=bm['date'], y=bm['high']*1.02, mode='markers', marker=dict(symbol='star', color='red', size=15), name="BOM"), row=1, col=1)

            # Tầng 2, 3, 4
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Volume", marker_color='dodgerblue'), row=2, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], name="RSI", line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['rs']*50, name="Sức mạnh RS", line=dict(color='magenta', width=2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['adx'], fill='tozeroy', name="Lực xu hướng ADX", line=dict(color='white')), row=4, col=1)
            
            fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Dữ liệu mã {ticker} không khả dụng. Kiểm tra file hose.csv")

    elif menu == "🎯 SIÊU ĐIỂM MUA":
        if 'symbol' in hose_data_raw.columns:
            st.subheader("🎯 BỘ LỌC CỔ PHIẾU CÓ DÒNG TIỀN ĐỘT BIẾN")
            findings = []
            for s in hose_data_raw['symbol'].unique():
                d = compute_signals(hose_data_raw[hose_data_raw['symbol'] == s], vni_data_raw)
                if d is not None:
                    last = d.iloc[-1]
                    if last['bomb'] or last['buy']:
                        findings.append({"Mã": s, "Tín hiệu": "💣 BOM TIỀN" if last['bomb'] else "⬆️ MUA", "RS": round(last['rs'],2), "RSI": round(last['rsi'],1)})
            st.dataframe(pd.DataFrame(findings).sort_values("RS", ascending=False), use_container_width=True)
else:
    st.info("💡 Đang chờ dữ liệu từ file hose.csv và vnindex.csv...")
