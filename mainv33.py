import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V42 - FIREANT PRO CLONE", layout="wide")

# --- HÀM TẢI DỮ LIỆU ---
def load_data(file_name):
    if not os.path.exists(file_name): return None
    df = pd.read_csv(file_name)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Tìm cột Symbol
    for col in ['symbol', 'ticker', 'mã']:
        if col in df.columns: 
            df = df.rename(columns={col: 'symbol'})
            break
    # Tìm cột Date
    for col in ['date', 'ngày']:
        if col in df.columns: 
            df = df.rename(columns={col: 'date'})
            break
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
    return df.sort_values('date')

# --- HÀM TÍNH TOÁN (ĐẢM BẢO KHÔNG LỖI) ---
def calculate_indicators(df):
    if df is None or len(df) < 5: return None
    df = df.copy().sort_values('date')
    
    # Ép kiểu số để vẽ MA và điểm mua
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close'])
    
    # 1. Các đường MA
    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(50, min_periods=1).mean()
    
    # 2. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    # 3. Dòng tiền (Money In)
    df['vol20'] = df['volume'].rolling(20).mean()
    # Tín hiệu MUA: Giá > MA20 & Vol > 1.3 lần trung bình
    df['is_buy'] = (df['close'] > df['ma20']) & (df['volume'] > df['vol20'] * 1.3)
    
    # 4. Chấm điểm
    score = 0
    last = df.iloc[-1]
    if last['close'] > last['ma20']: score += 4
    if last['volume'] > last['vol20']: score += 4
    if last['rsi'] > 50: score += 2
    df['total_score'] = score
    
    return df

# --- SIDEBAR ---
hose_df = load_data("hose.csv")
vni_df = load_data("vnindex.csv")

with st.sidebar:
    st.header("🏆 FIREANT PRO V42")
    ticker = st.text_input("🔍 NHẬP MÃ (HPG, SSI...):", "HPG").upper()
    
    st.divider()
    if st.button("📈 SỨC KHỎE VN-INDEX", use_container_width=True):
        if vni_df is not None:
            v_res = calculate_indicators(vni_df)
            st.metric("VNI SCORE", f"{v_res['total_score'].iloc[-1]}/10")
            st.write("Xu hướng: " + ("BẮT ĐẦU TĂNG" if v_res['close'].iloc[-1] > v_res['ma20'].iloc[-1] else "TÍCH LŨY/GIẢM"))
    
    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ KỸ THUẬT", "📊 DÒNG TIỀN NGÀNH", "🎯 ĐIỂM MUA TỔ CHỨC"])

# --- HIỂN THỊ CHÍNH ---
if hose_df is not None:
    if menu == "📈 ĐỒ THỊ KỸ THUẬT":
        st.subheader(f"📊 PHÂN TÍCH KỸ THUẬT CHI TIẾT: {ticker}")
        df_m = hose_df[hose_df['symbol'] == ticker]
        data = calculate_indicators(df_m)
        
        if data is not None:
            # Layout 3 tầng
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # Tầng 1: Candle + MA20 + MA50
            fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma50'], line=dict(color='cyan', width=1.5), name="MA50"), row=1, col=1)
            
            # ĐIỂM MUA (Mũi tên xanh)
            buys = data[data['is_buy']]
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.97, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name="TIỀN VÀO"), row=1, col=1)

            # Tầng 2: Volume
            fig.add_trace(go.Bar(x=data['date'], y=data['volume'], name="Khối lượng", marker_color='dodgerblue'), row=2, col=1)
            
            # Tầng 3: RSI
            fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)

            # CẤU HÌNH ZOOM & KÉO THẢ (GIỐNG FIREANT)
            fig.update_layout(
                height=800, 
                template="plotly_dark", 
                xaxis_rangeslider_visible=False,
                dragmode='pan', # Cho phép kéo chuột để xem quá khứ
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True}) # Bật cuộn chuột để Zoom
            st.success(f"Điểm Dòng Tiền: {data['total_score'].iloc[-1]}/10")
        else:
            st.error(f"Không tìm thấy dữ liệu cho mã {ticker}")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN NGÀNH (Scale 10)")
        nganh_master = {
            "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
            "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'], 
            "THÉP": ['HPG','NKG','HSG'], 
            "BANK": ['VCB','TCB','MBB','STB'],
            "BĐS": ['DIG','PDR','VHM','GEX']
        }
        res = []
        for n, mãs in nganh_master.items():
            pts = []
            for m in mãs:
                d = calculate_indicators(hose_df[hose_df['symbol'] == m])
                if d is not None: pts.append(d['total_score'].iloc[-1])
            res.append({"Ngành": n, "Sức Mạnh": round(np.mean(pts),1) if pts else 0, "Số mã": len(pts)})
        
        st.table(pd.DataFrame(res).sort_values("Sức Mạnh", ascending=False))

    elif menu == "🎯 ĐIỂM MUA TỔ CHỨC":
        st.subheader("🚀 DANH SÁCH MÃ CÓ DÒNG TIỀN ĐỘT BIẾN")
        found = []
        for s in hose_df['symbol'].unique():
            d = calculate_indicators(hose_df[hose_df['symbol'] == s])
            if d is not None and d['is_buy'].iloc[-1]:
                found.append({"Mã": s, "Điểm": d['total_score'].iloc[-1], "RSI": round(d['rsi'].iloc[-1],1)})
        if found:
            st.dataframe(pd.DataFrame(found).sort_values("Điểm", ascending=False), use_container_width=True)
        else:
            st.info("Hôm nay chưa có mã nào bùng nổ Vol.")
else:
    st.error("❌ Thiếu file hose.csv! Hãy upload file vào thư mục app.")
