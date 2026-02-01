import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="V36.1 - PRO TERMINAL", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU & TÍNH TOÁN ---
def calculate_master_signals(df, vni_df):
    if df is None or len(df) < 130: return None
    df = df.copy()
    
    # Chuẩn hóa cột
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if 'date' not in df.columns: df = df.reset_index().rename(columns={'index':'date'})
    df['date'] = pd.to_datetime(df['date'])
    
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    
    c, v, h, l = df['close'], df['volume'], df['high'], df['low']
    
    # 1. Chỉ báo xu hướng
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # 2. Bollinger Band
    std = c.rolling(20).std()
    df['bb_w'] = (std * 4) / df['ma20'].replace(0, 0.001)
    
    # 3. RSI & RS & ADX
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').reset_index(drop=True)
    df['rs'] = ((c / c.shift(10)) / (vni_c / vni_c.shift(10).replace(0, 1)) - 1) * 100
    df['adx'] = (c.diff().abs().rolling(14).mean() / c.rolling(14).mean().replace(0,1)) * 1000

    # 4. LOGIC NỀN GIÁ & DÒNG TIỀN
    # Nền giá 6 tháng (biên độ hẹp < 25%)
    df['range_6m'] = (h.rolling(120).max() - l.rolling(120).min()) / l.rolling(120).min().replace(0,1)
    
    # Tín hiệu
    df['is_bomb'] = (df['bb_w'] <= df['bb_w'].rolling(30).min())
    df['money_in'] = (v > v.rolling(20).mean() * 1.5)
    df['is_buy'] = (df['money_in']) & (c > df['ma20']) & (df['ma20'] > df['ma50']) & (df['rsi'] < 80)
    
    # 5. CHẤM ĐIỂM (Sửa lỗi gán :=)
    last_idx = len(df) - 1
    score = 0
    if df['ma20'][last_idx] > df['ma50'][last_idx]: score += 2
    if df['rs'][last_idx] > 0: score += 3
    if df['money_in'][last_idx]: score += 3
    if df['bb_w'][last_idx] < 0.06: score += 2
    
    df['total_score'] = score
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏆 TRADING V36.1")
    ticker = st.text_input("🔍 SOI MÃ:", "MWG").upper()
    
    if st.button("🔄 UPDATE DATA (OVERWRITE CSV)", use_container_width=True):
        with st.spinner("Đang tải dữ liệu thực..."):
            vni = yf.download("^VNINDEX", period="2y")
            vni.to_csv("vni_v35.csv")
            
            # Danh sách mã mở rộng theo yêu cầu
            m_list = ['MWG','FRT','DGW','MSN','SSI','VND','VCI','HPG','NKG','HSG','DIG','PDR','VHM','VCB','TCB','MBB','STB','FTS','VIX','MWG','PNJ']
            all_data = []
            for m in list(set(m_list)):
                t = yf.download(f"{m}.VN", period="2y", progress=False)
                t['symbol'] = m
                all_data.append(t)
            pd.concat(all_data).to_csv("hose_v35.csv")
            st.success("Đã ghi đè file CSV thành công!")
            st.rerun()

    menu = st.radio("MENU CHÍNH:", ["📊 DÒNG TIỀN NGÀNH", "🎯 LỌC SIÊU ĐIỂM MUA", "📈 CHART FIREANT PRO"])

# --- XỬ LÝ HIỂN THỊ ---
if os.path.exists("vni_v35.csv") and os.path.exists("hose_v35.csv"):
    vni_df = pd.read_csv("vni_v35.csv")
    hose_df = pd.read_csv("hose_v35.csv")
    
    # Đo lường sức mạnh thị trường
    vni_c = pd.to_numeric(vni_df.iloc[:, 1], errors='coerce').dropna()
    vni_rsi_val = 100 - (100 / (1 + (vni_c.diff().where(vni_c.diff() > 0, 0).rolling(14).mean() / -vni_c.diff().where(vni_c.diff() < 0, 0).rolling(14).mean().replace(0,1)))).iloc[-1]
    
    st.subheader(f"🌐 VN-INDEX: {vni_c.iloc[-1]:,.2f} | RSI: {vni_rsi_val:.1f}")
    if vni_rsi_val > 70: st.warning("⚠️ Thị trường vùng Quá Mua - Hạn chế giải ngân đuổi!")
    elif vni_rsi_val < 35: st.success("✅ Thị trường vùng Chiết khấu - Cơ hội săn hàng nền tốt.")
    else: st.info("💎 Thị trường trạng thái ổn định - Tập trung lọc cổ phiếu có nền.")

    if menu == "📊 DÒNG TIỀN NGÀNH":
        nganh_dict = {
            "BÁN LẺ": ['MWG','FRT','DGW','MSN','PNJ'],
            "CHỨNG KHOÁN": ['SSI','VND','VCI','VIX','FTS'],
            "THÉP": ['HPG','NKG','HSG'],
            "BẤT ĐỘNG SẢN": ['DIG','PDR','VHM'],
            "NGÂN HÀNG": ['VCB','TCB','MBB','STB']
        }
        summary = []
        for n, mãs in nganh_dict.items():
            pts = []
            for m in mãs:
                d = calculate_master_signals(hose_df[hose_df['symbol'] == m].copy(), vni_df)
                if d is not None: pts.append(d['total_score'].iloc[-1])
            avg = np.mean(pts) if pts else 0
            summary.append({"Ngành": n, "Điểm (10)": round(avg, 1), "Dòng tiền": "🔥 MẠNH" if avg > 7 else "🔵 ỔN ĐỊNH"})
        st.table(pd.DataFrame(summary).sort_values("Điểm (10)", ascending=False))

    elif menu == "🎯 LỌC SIÊU ĐIỂM MUA":
        st.write("### 🔍 Cổ phiếu Nền > 6 tháng + Chớm nổ Dòng tiền")
        results = []
        for s in hose_df['symbol'].unique():
            d = calculate_master_signals(hose_df[hose_df['symbol'] == s].copy(), vni_df)
            if d is not None:
                l = d.iloc[-1]
                # Lọc: Nền 6 tháng (range < 25%) + Có tiền vào HOẶC BB thắt chặt
                if l['range_6m'] < 0.25 and (l['money_in'] or l['is_bomb']):
                    results.append({
                        "Mã": s, "Điểm": l['total_score'], "Biên nền 6th": f"{l['range_6m']*100:.1f}%",
                        "RSI": round(l['rsi'], 1), "Tín hiệu": "🏹 ĐIỂM MUA" if l['is_buy'] else "💣 ĐANG NÉN"
                    })
        st.dataframe(pd.DataFrame(results).sort_values("Điểm", ascending=False), use_container_width=True)

    elif menu == "📈 CHART FIREANT PRO":
        df_m = calculate_master_signals(hose_df[hose_df['symbol'] == ticker].copy(), vni_df)
        if df_m is not None:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.1, 0.2, 0.2])
            
            # Trục Y bên phải cho giống FireAnt
            fig.add_trace(go.Candlestick(x=df_m['date'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'], name=ticker), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma20'], line=dict(color='yellow', width=2), name="MA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['ma50'], line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
            
            # Hiển thị Bom và Mũi tên
            b = df_m[df_m['is_bomb']]
            fig.add_trace(go.Scatter(x=b['date'], y=b['high']*1.03, mode='text', text="💣", textfont=dict(size=20), name="Nén"), row=1, col=1)
            s = df_m[df_m['is_buy']]
            fig.add_trace(go.Scatter(x=s['date'], y=s['low']*0.97, mode='markers+text', text="🏹 MUA", marker=dict(symbol='triangle-up', size=15, color='lime'), name="MUA"), row=1, col=1)

            # Volume tô màu
            v_colors = ['red' if df_m['close'][i] < df_m['open'][i] else 'green' for i in range(len(df_m))]
            fig.add_trace(go.Bar(x=df_m['date'], y=df_m['volume'], marker_color=v_colors, name="Volume"), row=2, col=1)
            
            # Các chỉ báo phụ
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rsi'], line=dict(color='orange'), name="RSI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['rs'], line=dict(color='magenta'), name="RS"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_m['date'], y=df_m['adx'], line=dict(color='white'), name="ADX"), row=4, col=1)

            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
            fig.update_yaxes(side="right", fixedrange=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            l = df_m.iloc[-1]
            st.success(f"🎯 **{ticker}** | Target 1: {l['close']*1.1:,.0f} | Target 2: {l['close']*1.22:,.0f} | Stoploss: {l['ma50']:,.0f}")
else:
    st.info("Nhấn nút 'CẬP NHẬT DATA' ở Sidebar để khởi tạo hệ thống.")
