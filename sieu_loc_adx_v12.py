import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import webbrowser

# --- CÁC HÀM TÍNH TOÁN CHUẨN WILDER (Triệt tiêu ADX ảo) ---
def tinh_adx_chuan(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    up_move = high.diff(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1/period, adjust=False).mean()

def tinh_rsi_chuan(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))

# --- HÀM LỌC MÃ TỪ CSV (Săn siêu phẩm ADX > 20) ---
def quet_ma_tu_csv(file_path, limit=100):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file {file_path}"); return []
    try:
        df_all = pd.read_csv(file_path)
        danh_sach = df_all['symbol'].unique().tolist()[:limit]
        ket_qua = []
        print(f"🔄 Đang quét Real-time {len(danh_sach)} mã đầu tiên...")
        for ticker in danh_sach:
            df = yf.download(f"{ticker}.VN", period="60d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 30: continue
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.columns = [col.lower() for col in df.columns]
            
            # Tính ADX nhạy cho phiên hiện tại
            adx_series = tinh_adx_chuan(df)
            last_adx = adx_series.iloc[-1]
            
            # Thêm điều kiện Quả bom cho bộ lọc
            sma20 = df['close'].rolling(20).mean()
            bb_w = (df['close'].rolling(20).std() * 4) / sma20
            is_bomb = bb_w.iloc[-1] <= bb_w.rolling(20).min().iloc[-1]
            
            if last_adx > 20:
                trang_thai = "💣 ĐANG NÉN" if is_bomb else "🚀 ĐANG CHẠY"
                ket_qua.append({
                    'Mã': ticker, 
                    'ADX': round(last_adx, 2), 
                    'Giá': int(df['close'].iloc[-1]),
                    'Trạng thái': trang_thai
                })
        return ket_qua
    except Exception as e:
        print(f"⚠️ Lỗi quét: {e}"); return []

# --- HÀM VẼ ĐỒ THỊ VÀ BÁO ĐIỂM MUA NHẠY TRONG PHIÊN ---
def ve_do_thi_v14(ticker_input):
    try:
        end_d = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        start_d = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        df = yf.download(f"{ticker_input}.VN", start=start_d, end=end_d, progress=False, auto_adjust=True)
        if df.empty: return
        df = df.reset_index()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [col.lower() for col in df.columns]

        # 1. TÍNH TOÁN CHỈ BÁO
        df['adx'] = tinh_adx_chuan(df)
        df['rsi'] = tinh_rsi_chuan(df)
        df['sma20'] = df['close'].rolling(20).mean()
        df['bb_w'] = (df['close'].rolling(20).std() * 4) / df['sma20']
        df['bomb'] = df['bb_w'] <= df['bb_w'].rolling(20).min()
        df['vol_sma10'] = df['volume'].rolling(10).mean()
        
        # ĐIỂM MUA NHẠY TRONG PHIÊN: Vol > 80% trung bình + Giá xanh + ADX ngóc lên
        df['is_buy'] = (df['volume'] > df['vol_sma10'] * 0.8) & (df['close'] > df['open']) & (df['adx'] > 20)
        df['is_test'] = (df['close'] < df['open']) & (df['volume'] < df['vol_sma10'] * 0.6)

        # 2. VẼ SUBPLOTS
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                           subplot_titles=(f'PHÂN TÍCH {ticker_input} - ĐIỂM NỔ 💣', 'DÒNG TIỀN (VOLUME)', 'RSI (Cam) & ADX (Xanh lơ)'),
                           row_width=[0.25, 0.2, 0.55])

        # Candle & Symbols
        fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Giá'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df[df['is_buy']]['date'], y=df[df['is_buy']]['low']*0.98, mode='text', text="<b>MUA</b>", textfont=dict(color="lime"), name='MUA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df[df['is_test']]['date'], y=df[df['is_test']]['high']*1.02, mode='text', text="<b>TEST</b>", textfont=dict(color="yellow"), name='TEST'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df[df['bomb']]['date'].tail(7), y=df[df['bomb']]['high'].tail(7)*1.05, mode='text', text="💣", textfont=dict(size=18), name='BOM'), row=1, col=1)

        # Volume & Indicators
        v_color = ['red' if r['open'] > r['close'] else 'green' for _, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=v_color, name='Vol'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], line=dict(color='orange', width=2), name='RSI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['adx'], line=dict(color='cyan', width=2.5), name='ADX'), row=3, col=1)
        
        # Vạch kẻ 23 (Vạch kẻ bạn yêu cầu)
        fig.add_hline(y=23, line_dash="dash", line_color="white", row=3, col=1, annotation_text="Sóng (23)")

        # 3. BẢNG THÔNG SỐ LỀ PHẢI
        last = df.iloc[-1]
        trang_thai_adx = "VÀO SÓNG 🌊" if last['adx'] > 23 else "TÍCH LŨY 💤"
        info = (f"<b>MÃ: {ticker_input}</b><br>Giá: {int(last['close']):,}<br>ADX: {round(last['adx'],1)} ({trang_thai_adx})<br>----------<br>"
                f"🎯 Chốt T1 (7%): {int(last['close']*1.07):,}<br>🎯 Chốt T2 (15%): {int(last['close']*1.15):,}<br>"
                f"🛑 Cắt lỗ (6%): {int(last['close']*0.94):,}<br>----------<br>"
                f"🟠 RSI | 🔵 ADX")
        
        fig.add_annotation(xref="paper", yref="paper", x=1.22, y=0.5, text=info, showarrow=False, align="left", bgcolor="rgba(30,30,30,0.95)", bordercolor="gray", borderwidth=1, font=dict(color="white", size=12))
        fig.update_layout(height=850, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(r=280))
        
        f_name = f"chart_{ticker_input}.html"
        fig.write_html(f_name)
        webbrowser.open('file://' + os.path.realpath(f_name))
        print(f"✅ Đã phân tích xong {ticker_input}. Xem trên trình duyệt.")

    except Exception as e: print(f"⚠️ Lỗi vẽ: {e}")

# --- KHỞI CHẠY ---
if __name__ == "__main__":
    print("\n" + "⭐"*20)
    print("HỆ THỐNG PHÂN TÍCH SÓNG ADX V14")
    print("⭐"*20)
    
    # Bước 1: Lọc mã đẹp từ file CSV của bạn
    list_dep = quet_ma_tu_csv("hose.csv", limit=100)
    if list_dep:
        print("\n🔥 DANH SÁCH MÃ TIỀM NĂNG (ADX > 20):")
        print(pd.DataFrame(list_dep).sort_values(by='ADX', ascending=False).to_string(index=False))
    
    # Bước 2: Chế độ soi tay
    while True:
        ma = input("\n🔍 Nhập mã muốn soi chi tiết (hoặc 'exit'): ").upper().strip()
        if ma in ['EXIT', 'STOP', '']: break
        ve_do_thi_v14(ma)