import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V36.3 - PRO TERMINAL", layout="wide")

# --- HÀM TÍNH TOÁN (GIỮ NGUYÊN LOGIC CHART & CHỈ BÁO) ---
def calculate_master_signals(df, vni_df):
    if df is None or len(df) < 20: return None
    df = df.copy()
    
    # Chuẩn hóa cột
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if 'date' not in df.columns: df = df.reset_index().rename(columns={'index':'date'})
    
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    if len(df) < 20: return None
    
    c, v, h, l = df['close'], df['volume'], df['high'], df['low']
    
    # 1. Chỉ báo xu hướng & Bollinger Bands
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
    if len(vni_c_series) > 10:
        v_ratio = vni_c_series.iloc[-1] / vni_c_series.iloc[-10]
        df['rs'] = ((c / c.shift(10)) / v_ratio - 1) * 100
    else: df['rs'] = 0
        
    df['adx'] = (c.diff().abs().rolling(14).mean() / c.rolling(14).mean().replace(0,1)) * 1000

    # 3. LOGIC NỀN GIÁ 6 THÁNG
    df['range_6m'] = (h.rolling(120).max() - l.rolling(120).min()) / l.rolling(120).min().replace(0,1)

    # 4. TÍN HIỆU
    df['is_bomb'] = (df['bb_w'] <= df['bb_w'].rolling(30).min())
    df['money_in'] = (v > v.rolling(20).mean() * 1.5)
    df['is_buy'] = (df['money_in']) & (c > df['ma20']) & (df['ma20'] > df['ma50']) & (df['rsi'] < 80)
    
    # Điểm số (0-10)
    last_idx = len(df) - 1
    score = 0
    if df['ma20'].iloc[last_idx] > df['ma50'].iloc[last_idx]: score += 2
    if df['rs'].iloc[last_idx] > 0: score += 2
    if df['money_in'].iloc[last_idx]: score += 3
    if df['bb_w'].iloc[last_idx] < 0.06: score += 3
    df['total_score'] = score
    
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏆 V36.3 TERMINAL")
    ticker = st.text_input("🔍 SOI MÃ CHI TIẾT:", "MWG").upper()
    
    if st.button("🔄 CẬP NHẬT DỮ LIỆU CSV", use_container_width=True):
        with st.spinner("Đang ghi đè dữ liệu..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vni_v35.csv")
            m_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','VIX','HPG','NKG','HSG','DIG','PDR','VHM','VCB','TCB','MBB','STB']
            all_d = []
            for m in m_list:
                t = yf.download(f"{m}.VN", period="2y", progress=False)
                if not t.empty:
                    t['symbol'] = m
                    all_d.append(t)
            pd.concat(all_d).to_csv("hose_v35.csv")
            st.success("Đã cập nhật file CSV!")
            st.rerun()

    menu = st.radio("CHỨC NĂNG:", ["📈 ĐỒ THỊ FIREANT", "🎯 LỌC SIÊU ĐIỂM MUA", "📊 DÒNG TIỀN NGÀNH"])

# --- HIỂN THỊ ---
if os.path.exists("vni_v35.csv") and os.path.exists("hose_v35.csv"):
    vni_df = pd.read_csv("vni_v35.csv")
    hose_df = pd.read_csv("hose_v35.csv")
    
    # Highlight Sức mạnh VNINDEX
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').dropna()
    if not vni_c.empty:
        vni_last = vni_c.iloc[-1]
        vni_ma20 = vni_c.rolling(20).mean().iloc[-1]
        v_status = "TỐT (Dòng tiền khỏe)" if vni_last > vni_ma20 else "YẾU (Cẩn trọng)"
        st.subheader(f"🌐 VN-INDEX: {vni_last:,.2f} | Trạng thái: {v_status}")

    if menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.write("### 🔍 Siêu phẩm: Nền 6 tháng + Dòng tiền bùng nổ")
        results = []
        for s in hose_df['symbol'].unique():
            d = calculate_master_signals(hose_df[hose_df['symbol'] == s].copy(), vni_df)
            if d is not None:
                l = d.iloc[-1]
                # Lọc: Nền 6 tháng < 25% + (Tiền vào hoặc BB nén)
                if l['range_6m'] < 0.25 and (l['money_in'] or l['is_bomb']):
                    results.append({
                        "Mã": s, "Điểm": l['total_score'], "Biên độ nền": f"{l['range_6m']*100:.1f}%",
                        "RSI": round(l['rsi'], 1), "Tín hiệu": "🏹 MUA" if l['is_buy'] else "💣 NÉN CHẶT"
                    })
        
        if results:
            df_res = pd.DataFrame(results)
            st.dataframe(df_res.sort_values("Điểm", ascending=False), use_container_width=True)
        else:
            st.info("Chưa tìm thấy mã thỏa mãn nền giá 6 tháng. Hãy thử cập nhật dữ liệu mới.")

    elif menu == "📊 DÒNG TIỀN NGÀNH":
        st.subheader("🌊 XẾP HẠNG DÒNG TIỀN NHÓM NGÀNH")
        nganh_dict = {"BÁN LẺ": ['MWG','FRT','DGW','MSN'], "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX'], "THÉP": ['HPG','NKG','HSG'], "BANK": ['VCB','TCB','MBB','STB']}
        summary = []
        for n, mãs in nganh_dict.items():
            pts = []
            for m in mãs:
                d = calculate_master_signals(hose_df[hose_df['symbol'] == m].copy(), vni_df)
                if d is not None: pts.append(d['total_score'].iloc[-1])
            avg = np.mean(pts) if pts else 0
            summary.append({"Ngành": n, "Điểm (10)": round(avg, 1), "Dòng tiền": "🔥 MẠNH" if avg > 7 else "🔵 ỔN ĐỊNH"})
        st.table(pd.DataFrame(summary).sort_values("Điểm (10)", ascending=False))

    elif menu == "📈 ĐỒ THỊ FIREANT":
        df_m = calculate_master_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # --- TẦNG 1: GIÁ & BB & MA ---
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # Icon tín hiệu
            b = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=b['date'], y=b['high']*1.02, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            s = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=s['date'], y=s['low']*0.98, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # --- TẦNG 2: VOLUME ---
            v_colors = ['red' if df_m['close'].iloc[i] < df_m['open'].iloc[i] else 'green' for i in range(len(df_m))]
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], marker_color=v_colors, name="Volume"), row=2, col=1)
            
            # --- TẦNG 3: RSI & RS ---
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], line=dict(color='magenta'), name="RS"), row=3, col=1)
            
            # --- TẦNG 4: ADX ---
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['adx'], line=dict(color='white'), name="ADX"), row=4, col=1)

            # Tối ưu giao diện FireAnt
            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            fig.update_yaxes(side="right", fixedrange=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            l = df_m.iloc[-1]
            st.success(f"🎯 **{ticker}** | Giá: {l['close']:.1f} | Điểm: {l['total_score']}/10 | Target 1: {l['close']*1.1:,.0f} | SL: {l['ma50']:.1f}")
else:
    st.info("👋 Nhấn nút 'CẬP NHẬT DỮ LIỆU CSV' để bắt đầu.")
