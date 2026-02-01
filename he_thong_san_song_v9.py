import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import webbrowser
import os

def he_thong_san_song_v9():
    print("="*50)
    print("🚀 HE THONG CHUNG KHOAN V9 - CHAY TREN MAY TINH")
    print("👉 Go 'exit' de dung chuong trinh")
    print("="*50)
    
    while True:
        try:
            ticker_input = input("\n🔍 Nhap ma co phieu (VD: VGI, HPG, SSI): ").upper().strip()
            if ticker_input in ['EXIT', 'THOAT', 'STOP', '']: break

            # 1. TAI DU LIEU REAL-TIME
            end_d = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            start_d = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            df = yf.download(f"{ticker_input}.VN", start=start_d, end=end_d, progress=False, auto_adjust=True)
            if df.empty: 
                print(f"❌ Khong tim thay ma {ticker_input}. Vui long kiem tra lai!"); continue

            # Chuan hoa du lieu
            df = df.reset_index()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.columns = [col.lower() for col in df.columns]

            # 2. TINH TOAN CHI BAO
            # RSI (Cam)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))

            # ADX (Xanh lo)
            high, low, close = df['high'], df['low'], df['close']
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            plus_dm = high.diff().where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0).rolling(14).mean()
            minus_dm = low.diff().abs().where((low.diff().abs() > high.diff()) & (low.diff().abs() > 0), 0).rolling(14).mean()
            df['adx'] = (abs(plus_dm - minus_dm) / (plus_dm + minus_dm).replace(0, np.nan)) * 100
            df['adx'] = df['adx'].rolling(14).mean()

            # Qua bom 💣 & Tin hieu Mua/Test
            df['sma20'] = df['close'].rolling(20).mean()
            df['bb_w'] = (df['close'].rolling(20).std() * 4) / df['sma20']
            df['bomb'] = df['bb_w'] <= df['bb_w'].rolling(20).min()
            df['vol_sma10'] = df['volume'].rolling(10).mean()
            df['is_buy'] = (df['volume'] > df['vol_sma10'] * 1.5) & (df['close'] > df['open'])
            df['is_test'] = (df['close'] < df['open']) & (df['volume'] < df['vol_sma10'] * 0.6)

            # 3. KIEM TRA SONG ADX 23
            last_adx = df['adx'].iloc[-1]
            prev_adx = df['adx'].iloc[-2]
            adx_signal = "BINH THUONG"
            if last_adx > 23 and prev_adx <= 23: adx_signal = "BAT DAU VAO SONG 🌊"
            elif last_adx > 23: adx_signal = "DANG TRONG SONG 💪"

            # 4. VE DO THI
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                               subplot_titles=('GIA & QUA BOM 💣', 'DONG TIEN (VOLUME)', 'RSI (Cam) & ADX (Xanh lo)'),
                               row_width=[0.25, 0.2, 0.55])

            fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Gia'), row=1, col=1)
            
            # Ky hieu chu MUA/TEST/BOM
            fig.add_trace(go.Scatter(x=df[df['is_buy']]['date'], y=df[df['is_buy']]['low']*0.97, mode='text', text="<b>MUA</b>", textfont=dict(color="lime"), name='MUA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df[df['is_test']]['date'], y=df[df['is_test']]['high']*1.03, mode='text', text="<b>TEST</b>", textfont=dict(color="yellow"), name='TEST'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df[df['bomb']]['date'].tail(5), y=df[df['bomb']]['high'].tail(5)*1.07, mode='text', text="💣", textfont=dict(size=20), name='BOM'), row=1, col=1)

            # Volume
            v_color = ['red' if r['open'] > r['close'] else 'green' for _, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=v_color, name='Vol'), row=2, col=1)
            
            # RSI & ADX
            fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], line=dict(color='orange', width=2), name='RSI'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['adx'], line=dict(color='cyan', width=2.5), name='ADX'), row=3, col=1)
            
            fig.add_hline(y=23, line_dash="dash", line_color="white", row=3, col=1, annotation_text="Song (23)")

            # 5. BANG THONG SO (BEN HONG)
            last = df.iloc[-1]
            info = (f"<b>MA: {ticker_input}</b><br>Gia: {int(last['close']):,}<br>----------<br>"
                    f"Song ADX: <b>{adx_signal}</b><br>"
                    f"ADX HT: {round(last_adx, 2)}<br>----------<br>"
                    f"🎯 T1 (7%): {int(last['close']*1.07):,}<br>"
                    f"🎯 T2 (15%): {int(last['close']*1.15):,}<br>"
                    f"🛑 SL (6%): {int(last['close']*0.94):,}<br>----------<br>"
                    f"🟠 RSI | 🔵 ADX")

            fig.add_annotation(xref="paper", yref="paper", x=1.22, y=0.5, text=info, showarrow=False,
                               align="left", bgcolor="rgba(30,30,30,0.95)", bordercolor="gray", borderwidth=1, font=dict(color="white", size=12))

            fig.update_layout(height=850, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(r=280))
            
            # Luu ra file HTML va tu mo tren trinh duyet
            file_name = f"chart_{ticker_input}.html"
            fig.write_html(file_name)
            webbrowser.open('file://' + os.path.realpath(file_name))
            print(f"✅ Da mo do thi ma {ticker_input} tren trinh duyet.")

        except KeyboardInterrupt:
            print("\n👋 Da dung chuong trinh theo yeu cau.")
            break
        except Exception as e:
            print(f"⚠️ Co loi xay ra: {e}")

if __name__ == "__main__":
    he_thong_san_song_v9()