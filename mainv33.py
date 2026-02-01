import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V36.5 - BULLETPROOF", layout="wide")

# --- HÀM TÍNH TOÁN AN TOÀN TUYỆT ĐỐI ---
def calculate_master_signals(df, vni_df):
    if df is None or len(df) < 50: return None # Kiểm tra độ dài tối thiểu
    df = df.copy()
    
    # Chuẩn hóa cột
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if 'date' not in df.columns: df = df.reset_index().rename(columns={'index':'date'})
    
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    if len(df) < 50: return None # Kiểm tra lại sau khi dropna
    
    c, v = df['close'], df['volume']
    
    # 1. Chỉ báo xu hướng
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    std = c.rolling(20).std()
    df['bb_w'] = (std * 4) / df['ma20'].replace(0, 0.001)
    
    # 2. RSI & RS & ADX
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    vni_c_series = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').dropna().reset_index(drop=True)
    if len(vni_c_series) >= 10:
        v_ratio = vni_c_series.iloc[-1] / vni_c_series.iloc[-10]
        df['rs'] = ((c / c.shift(10)) / v_ratio - 1) * 100
    else: df['rs'] = 0
    df['adx'] = (c.diff().abs().rolling(14).mean() / c.rolling(14).mean().replace(0,1)) * 1000

    # 3. TÍN HIỆU
    df['money_in'] = (v > v.rolling(20).mean() * 1.2)
    df['is_bomb'] = (df['bb_w'] <= df['bb_w'].rolling(20).min())
    df['is_buy'] = (df['ma20'] > df['ma50']) & (c > df['ma20']) & (df['money_in'])
    
    # --- PHẦN SỬA LỖI INDEXERROR (BỌC BẢO VỆ) ---
    score = 0
    try:
        # Lấy giá trị cuối cùng một cách an toàn
        m20_last = df['ma20'].iloc[-1]
        m50_last = df['ma50'].iloc[-1]
        rs_last = df['rs'].iloc[-1]
        mi_last = df['money_in'].iloc[-1]
        bb_last = df['bb_w'].iloc[-1]

        if m20_last > m50_last: score += 3
        if rs_last > 0: score += 2
        if mi_last: score += 3
        if bb_last < 0.08: score += 2
    except Exception:
        score = 0 # Trả về 0 nếu có bất kỳ lỗi truy cập index nào
        
    df['total_score'] = score
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏆 TERMINAL V36.5")
    ticker = st.text_input("🔍 SOI MÃ CHI TIẾT:", "MWG").upper()
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU CSV", use_container_width=True):
        with st.spinner("Đang tải dữ liệu..."):
            vni = yf.download("^VNINDEX", period="1y")
            vni.to_csv("vni_v35.csv")
            # Danh sách mã bao quát các ngành
            m_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','VIX','HPG','NKG','HSG','DIG','PDR','VHM','VCB','TCB','MBB','STB','FTS','VDS','CTS']
            all_d = []
            for m in m_list:
                t = yf.download(f"{m}.VN", period="1y", progress=False)
                if not t.empty:
                    t['symbol'] = m
                    all_d.append(t)
            pd.concat(all_d).to_csv("hose_v35.csv")
            st.success("Xong!")
            st.rerun()

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ FIREANT", "🎯 LỌC SIÊU ĐIỂM MUA", "📊 DÒNG TIỀN NGÀNH"])

# --- HIỂN THỊ CHÍNH ---
if os.path.exists("vni_v35.csv") and os.path.exists("hose_v35.csv"):
    vni_df = pd.read_csv("vni_v35.csv")
    hose_df = pd.read_csv("hose_v35.csv")
    
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').dropna()
    if not vni_c.empty:
        vni_curr = vni_c.iloc[-1]
        vni_ma20 = vni_c.rolling(20).mean().iloc[-1] if len(vni_c) >= 20 else vni_curr
        v_status = "🟢 THUẬN LỢI" if vni_curr > vni_ma20 else "🔴 RỦI RO"
        st.subheader(f"🌐 VN-INDEX: {vni_curr:,.2f} | {v_status}")

    if menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.write("### 🎯 Điểm Mua: MA20 > MA50 & Dòng Tiền Mạnh")
        results = []
        for s in hose_df['symbol'].unique():
            subset = hose_df[hose_df['symbol'] == s].copy()
            d = calculate_master_signals(subset, vni_df)
            if d is not None:
                l = d.iloc[-1]
                if l['is_buy'] or l['is_bomb']:
                    results.append({
                        "Mã": s, "Điểm": l['total_score'], "RSI": round(l['rsi'], 1), 
                        "RS": round(l['rs'], 1), "Tín hiệu": "🏹 MUA" if l['is_buy'] else "💣 NÉN"
                    })
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("Điểm", ascending=False), use_container_width=True)
        else: st.info("Không có mã nào đạt tiêu chí.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 SỨC MẠNH DÒNG TIỀN NGÀNH (Thang 10)")
        nganh_dict = {
            "BÁN LẺ": ['MWG','FRT','DGW','MSN'], 
            "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS','CTS'], 
            "THÉP": ['HPG','NKG','HSG'], 
            "BANK": ['VCB','TCB','MBB','STB']
        }
        summary = []
        for n, mãs in nganh_dict.items():
            pts = []
            for m in mãs:
                subset = hose_df[hose_df['symbol'] == m].copy()
                if not subset.empty:
                    d = calculate_master_signals(subset, vni_df)
                    if d is not None: pts.append(d['total_score'].iloc[-1])
            avg = np.mean(pts) if pts else 0
            summary.append({"Ngành": n, "Điểm": round(avg, 1), "Đánh giá": "⭐ DẪN SÓNG" if avg > 6 else "⚪ ĐỢI TIỀN"})
        st.table(pd.DataFrame(summary).sort_values("Điểm", ascending=False))

    elif menu == "📈 ĐỒ THỊ FIREANT":
        df_m = calculate_master_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # Tín hiệu 💣/🏹
            b = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=b['date'], y=b['high']*1.02, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            s = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=s['date'], y=s['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # Volume & Chỉ báo
            v_colors = ['red' if df_m['close'].iloc[i] < df_m['open'].iloc[i] else 'green' for i in range(len(df_m))]
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], marker_color=v_colors, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], line=dict(color='magenta'), name="RS"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['adx'], line=dict(color='white'), name="ADX"), row=4, col=1)

            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            fig.update_yaxes(side="right", fixedrange=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            st.success(f"🚩 **{ticker}** | Giá: {df_m['close'].iloc[-1]:,.1f} | Điểm: {df_m['total_score'].iloc[-1]}/10")
else:
    st.info("Nhấn 'CẬP NHẬT DỮ LIỆU CSV' để bắt đầu.")
