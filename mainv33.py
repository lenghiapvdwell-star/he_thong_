import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V33.6 - DÒNG TIỀN NGÀNH", layout="wide")

# --- HÀM TÍNH TOÁN KỸ THUẬT SIÊU CẤP ---
def calculate_pro_signals(df, vni_df):
    if df is None or len(df) < 30: return None
    df = df.copy()
    
    # Làm phẳng dữ liệu
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).lower() for col in df.columns]
    
    for c in ['close', 'open', 'high', 'low', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).reset_index(drop=True)
    c, v = df['close'], df['volume']
    
    # Chỉ báo
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    df['rsi'] = 100 - (100 / (1 + (c.diff().where(c.diff() > 0, 0).ewm(14).mean() / 
                                  -c.diff().where(c.diff() < 0, 0).ewm(14).mean().replace(0, 1))))
    
    # Tính RS (Sức mạnh so với thị trường)
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce')
    df['rs'] = ((c/c.shift(5)) / (vni_c/vni_c.shift(5)) - 1) * 100
    
    # Logic tính điểm dòng tiền (Thang điểm 10)
    # 1. Điểm xu hướng (4đ): Giá trên MA20 và MA20 trên MA50
    # 2. Điểm sức mạnh (3đ): RS > 0 (Mạnh hơn VNI)
    # 3. Điểm dòng tiền (3đ): Volume 5 phiên gần nhất tăng so với trung bình
    score = 0
    l = df.iloc[-1]
    if l['close'] > l['ma20']: score += 2
    if l['ma20'] > l['ma50']: score += 2
    if l['rs'] > 0: score += 3
    if l['volume'] > v.rolling(20).mean().iloc[-1]: score += 3
    
    df['total_score'] = score
    return df

# --- SIDEBAR & DATA ---
with st.sidebar:
    st.header("⚡ HỆ THỐNG V33.6")
    if st.button("🔄 CẬP NHẬT DỮ LIỆU", use_container_width=True):
        with st.spinner("Đang tải dữ liệu..."):
            vni = yf.download("^VNINDEX", period="1y")
            vni.to_csv("vni_v33.csv")
            
            # Danh sách mã theo ngành
            nganh_list = {
                'BAN_LE': ['MWG','FRT','DGW','MSN'],
                'CHUNG_KHOAN': ['SSI','VND','VCI','HCM'],
                'THEP': ['HPG','NKG','HSG'],
                'BDS': ['DIG','PDR','VHM','DXG'],
                'BANK': ['VCB','TCB','MBB','STB']
            }
            all_mã = [m for n in nganh_list.values() for m in n]
            data_all = []
            for m in all_mã:
                t = yf.download(f"{m}.VN", period="1y", progress=False)
                t['symbol'] = m
                data_all.append(t)
            pd.concat(data_all).to_csv("hose_v33.csv")
            st.success("Đã cập nhật xong!")
            st.rerun()

    mode = st.radio("MENU:", ["📊 DÒNG TIỀN NGÀNH", "📈 SOI CHI TIẾT"])
    ticker = st.text_input("MÃ SOI:", "MWG").upper()

# --- XỬ LÝ HIỂN THỊ ---
if os.path.exists("vni_v33.csv"):
    vni_df = pd.read_csv("vni_v33.csv")
    hose_df = pd.read_csv("hose_v33.csv")

    if mode == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN THEO NHÓM NGÀNH")
        nganh_dict = {
            "BÁN LẺ": ['MWG','FRT','DGW','MSN'],
            "CHỨNG KHOÁN": ['SSI','VND','VCI','HCM'],
            "THÉP": ['HPG','NKG','HSG'],
            "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM','DXG'],
            "NGÂN HÀNG": ['VCB','TCB','MBB','STB']
        }
        
        summary = []
        for ten_nganh, dsm in nganh_dict.items():
            diem_nganh = []
            for m in dsm:
                data_m = calculate_pro_signals(hose_df[hose_df['symbol'] == m].copy(), vni_df)
                if data_m is not None:
                    diem_nganh.append(data_m['total_score'].iloc[-1])
            
            tb_diem = np.mean(diem_nganh) if diem_nganh else 0
            trang_thai = "🔥 DẪN DẮT" if tb_diem >= 7 else "✅ TÍCH CỰC" if tb_diem >= 5 else "☁️ ĐANG TÍCH LŨY"
            summary.append({"Nhóm Ngành": ten_nganh, "Điểm Dòng Tiền": round(tb_diem, 1), "Trạng Thái": trang_thai})
        
        df_view = pd.DataFrame(summary).sort_values(by="Điểm Dòng Tiền", ascending=False)
        st.table(df_view)
        st.info("💡 Điểm > 7: Ưu tiên giải ngân mạnh. Điểm < 4: Đứng ngoài quan sát.")

    elif mode == "📈 SOI CHI TIẾT":
        df_m = calculate_pro_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_m is not None:
            # Code vẽ chart (giữ nguyên sự mượt mà của bản trước)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df_m['Date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['Date'], y=df_m['ma20'], line=dict(color='yellow'), name="MA20"), row=1, col=1)
            fig.add_trace(go.Bar(x=df_m['Date'], y=df_m['volume'], name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_m['Date'], y=df_m['rsi'], name="RSI"), row=3, col=1)
            
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # Target/Stoploss
            l = df_m.iloc[-1]
            st.success(f"🎯 Target: {l['close']*1.12:,.0f} | 🛑 Stop: {l['ma20']:,.0f}")

else:
    st.warning("⚠️ Nhấn 'CẬP NHẬT DỮ LIỆU' ở menu bên trái để hệ thống tính toán điểm.")
