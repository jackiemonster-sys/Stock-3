from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# 試圖載入 FinMind，若環境未安裝則優雅降級
try:
    from FinMind.data import DataLoader
    FINMIND_AVAILABLE = True
except ImportError:
    FINMIND_AVAILABLE = False

# ==========================================
# 0. 全局設定與通用工具函式
# ==========================================
st.set_page_config(
    page_title="台股全方位選股與分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

DEFAULT_STOCKS = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2382": "廣達",
    "3231": "緯創", "2308": "台達電", "2303": "聯電", "2881": "富邦金",
    "2882": "國泰金", "2603": "長榮", "2609": "陽明", "3037": "欣興",
    "2345": "智邦", "6669": "緯穎", "3661": "世芯-KY", "3035": "智原",
    "2357": "華碩", "2356": "英業達", "2002": "中鋼", "1301": "台塑"
}

@st.cache_data(ttl=3600)
def fetch_yf_data(ticker_symbol, period="6m", interval="1d"):
    """統一 yfinance 資料抓取，自動處理 MultiIndex 與欄位格式"""
    try:
        df = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return None

# ==========================================
# 1. 導覽選單
# ==========================================
st.sidebar.title("📌 功能導覽")
page = st.sidebar.radio(
    "請選擇分析模組：",
    [
        "🚀 5日波段選股",
        "⚖️ 土洋合買 + 技術面篩選",
        "🔍 背離多重掃描",
        "⚡ 關鍵轉折 K線篩選",
        "🏛️ 證交所/櫃買全方位分析"
    ]
)

# ==========================================
# 模組 1: 5日波段選股
# ==========================================
if page == "🚀 5日波段選股":
    st.title("📈 台股 5日波段選股 App")

    sort_by = st.radio(
        "排序方式：",
        options=["模型綜合評分 (Score)", "成交金額大小"],
        horizontal=True,
    )

    @st.cache_data(ttl=1800)
    def fetch_real_stock_data(stock_dict):
        results = []
        latest_trade_date = datetime.now().strftime("%Y/%m/%d")

        for symbol, name in stock_dict.items():
            try:
                hist_df = fetch_yf_data(f"{symbol}.TW", period="3m", interval="1d")
                if hist_df is None or len(hist_df) < 10:
                    continue

                hist_df = hist_df.reset_index()
                hist_df["Date"] = hist_df["Date"].dt.strftime("%Y-%m-%d")

                latest_close = float(hist_df["Close"].iloc[-1])
                prev_close = float(hist_df["Close"].iloc[-2])
                change = latest_close - prev_close
                pct_change = round((change / prev_close) * 100, 2)
                vol = int(hist_df["Volume"].iloc[-1] / 1000)

                hist_df["MA5"] = hist_df["Close"].rolling(5).mean()
                hist_df["MA10"] = hist_df["Close"].rolling(10).mean()

                ema12 = hist_df["Close"].ewm(span=12, adjust=False).mean()
                ema26 = hist_df["Close"].ewm(span=26, adjust=False).mean()
                hist_df["DIF"] = ema12 - ema26
                hist_df["DEM"] = hist_df["DIF"].ewm(span=9, adjust=False).mean()
                hist_df["MACD_Hist"] = hist_df["DIF"] - hist_df["DEM"]

                c_ma5 = hist_df["MA5"].iloc[-1]
                c_ma10 = hist_df["MA10"].iloc[-1]

                target_tp = round(latest_close * 1.08, 2)
                stop_loss = round(c_ma10, 2)

                score = 50
                signals = []
                if latest_close > c_ma5:
                    score += 25
                    signals.append("站上5日線")
                if latest_close > c_ma10:
                    score += 25
                    signals.append("站上10日線")

                latest_trade_date = hist_df["Date"].iloc[-1]

                results.append({
                    "代號": symbol,
                    "名稱": name,
                    "綜合評分": score,
                    "收盤價": round(latest_close, 2),
                    "漲跌幅(%)": pct_change,
                    "建議停利": target_tp,
                    "建議停損": stop_loss,
                    "成交量(張)": vol,
                    "訊號": ", ".join(signals),
                    "交易日期": latest_trade_date,
                    "df": hist_df,
                })
            except Exception:
                continue

        return pd.DataFrame(results), latest_trade_date

    with st.spinner("正在讀取真實股市日 K 線數據..."):
        df_res, data_date = fetch_real_stock_data(DEFAULT_STOCKS)

    if df_res.empty:
        st.error("❌ 行情數據載入失敗，請確認網路連線或稍後再試。")
    else:
        st.markdown(f"📅 **數據更新日期：`{data_date}`**")

        if sort_by == "成交金額大小":
            df_sorted = df_res.sort_values(by=["成交量(張)", "綜合評分"], ascending=[False, False])
        else:
            df_sorted = df_res.sort_values(by=["綜合評分", "成交量(張)"], ascending=[False, False])

        display_cols = ["代號", "名稱", "綜合評分", "收盤價", "漲跌幅(%)", "建議停利", "建議停損", "成交量(張)"]

        st.dataframe(
            df_sorted[display_cols],
            column_config={
                "建議停利": st.column_config.NumberColumn("建議停利 (+8%)", format="%.2f 元"),
                "建議停損": st.column_config.NumberColumn("建議停損 (10日線)", format="%.2f 元"),
                "漲跌幅(%)": st.column_config.NumberColumn("漲跌幅(%)", format="%.2f%%"),
            },
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("📈 個股風控與 MACD 技術圖表")

        stock_options = df_sorted["代號"] + " " + df_sorted["名稱"]
        selected_option = st.selectbox("選擇股票：", options=stock_options)

        selected_code = selected_option.split(" ")[0]
        selected_row = df_sorted[df_sorted["代號"] == selected_code].iloc[0]
        hist = selected_row["df"]

        st.info(
            f"🎯 **{selected_row['名稱']} ({selected_row['代號']}) 風控提示**：\n"
            f"- 💵 **當前收盤價**：`{selected_row['收盤價']}` 元\n"
            f"- 🎯 **建議停利價**：`{selected_row['建議停利']}` 元（預期動能目標 +8%）\n"
            f"- 🛑 **建議停損價**：`{selected_row['建議停損']}` 元（跌破 10 日線支撐）"
        )

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=(f"{selected_row['名稱']} ({selected_row['代號']}) K線圖", "MACD 指標")
        )

        fig.add_trace(go.Candlestick(
            x=hist["Date"], open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name="K線", increasing_line_color="#ef5350", decreasing_line_color="#26a69a"
        ), row=1, col=1)

        colors = np.where(hist["MACD_Hist"] >= 0, "#ef5350", "#26a69a")
        fig.add_trace(go.Bar(x=hist["Date"], y=hist["MACD_Hist"], name="MACD柱狀", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["DIF"], name="DIF(快線)", line=dict(color="#2962FF", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["DEM"], name="MACD(慢線)", line=dict(color="#FF6D00", width=1.5)), row=2, col=1)

        fig.update_layout(
            height=550, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False, xaxis=dict(type="category"), xaxis2=dict(type="category")
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 模組 2: 土洋合買 + 技術面篩選
# ==========================================
elif page == "⚖️ 土洋合買 + 技術面篩選":
    st.title("📈 台股「土洋合買 + 均線/布林中軌轉上」選股")
    st.caption("自動抓取最新籌碼資料，結合布林中軌（20MA）與短均線向上訊號")

    @st.cache_data(ttl=3600)
    def get_latest_twse_data():
        current_date = datetime.now()
        for i in range(10):
            target_date = current_date - timedelta(days=i)
            date_str = target_date.strftime("%Y%m%d")
            url_fund = f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={date_str}&selectType=ALL"

            try:
                res_fund = requests.get(url_fund, headers=HTTP_HEADERS, timeout=8)
                if res_fund.status_code == 200:
                    data_fund = res_fund.json()
                    if data_fund.get("stat") == "OK" and "data" in data_fund:
                        cols_fund = [str(c).strip() for c in data_fund["fields"]]
                        df_fund = pd.DataFrame(data_fund["data"], columns=cols_fund)
                        return df_fund, target_date.strftime("%Y/%m/%d")
            except Exception:
                continue
        return pd.DataFrame(), None

    @st.cache_data(ttl=3600)
    def check_technical_signals(stock_code):
        try:
            hist = fetch_yf_data(f"{stock_code}.TW", period="50d", interval="1d")
            if hist is None or len(hist) < 30:
                return False, False, 0.0

            hist["MA5"] = hist["Close"].rolling(5).mean()
            hist["MA10"] = hist["Close"].rolling(10).mean()
            hist["MA20"] = hist["Close"].rolling(20).mean()

            latest_close = hist["Close"].iloc[-1]

            ma20_today = hist["MA20"].iloc[-1]
            ma20_prev = hist["MA20"].iloc[-2]
            ma20_prev2 = hist["MA20"].iloc[-3]

            slope_today = ma20_today - ma20_prev
            slope_prev = ma20_prev - ma20_prev2

            bb_middle_turning_up = (slope_today > 0) or (slope_today > slope_prev and slope_today >= -0.15)

            ma5_up = hist["MA5"].iloc[-1] > hist["MA5"].iloc[-2]
            ma10_up = hist["MA10"].iloc[-1] > hist["MA10"].iloc[-2]
            ma_signal = ma5_up and ma10_up and bb_middle_turning_up

            return bb_middle_turning_up, ma_signal, round(float(latest_close), 2)
        except Exception:
            return False, False, 0.0

    def process_and_filter(df_fund, filter_bollinger=False, filter_ma=False):
        if df_fund.empty:
            return pd.DataFrame()

        code_col = next((c for c in df_fund.columns if "代號" in c), df_fund.columns[0])
        name_col = next((c for c in df_fund.columns if "名稱" in c), df_fund.columns[1])
        vol_col = next((c for c in df_fund.columns if "成交" in c or "股數" in c), None)
        foreign_col = next((c for c in df_fund.columns if "外資" in c or "外陸資" in c), None)
        sitca_col = next((c for c in df_fund.columns if "投信" in c), None)

        if not (foreign_col and sitca_col):
            st.error("籌碼欄位解析失敗，請稍後再試。")
            return pd.DataFrame()

        temp_df = pd.DataFrame()
        temp_df["Code"] = df_fund[code_col].astype(str).str.strip()
        temp_df["Name"] = df_fund[name_col].astype(str).str.strip()

        for col_name, target in [(foreign_col, "Foreign_Buy"), (sitca_col, "Sitca_Buy")]:
            temp_df[target] = pd.to_numeric(
                df_fund[col_name].astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce"
            ).fillna(0)

        if vol_col:
            temp_df["Volume"] = pd.to_numeric(
                df_fund[vol_col].astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce"
            ).fillna(0)
        else:
            temp_df["Volume"] = temp_df["Foreign_Buy"].abs() + temp_df["Sitca_Buy"].abs()

        temp_df["Volume_K"] = (temp_df["Volume"] / 1000).astype(int)
        temp_df["Foreign_Buy_K"] = (temp_df["Foreign_Buy"] / 1000).astype(int)
        temp_df["Sitca_Buy_K"] = (temp_df["Sitca_Buy"] / 1000).astype(int)

        condition = (temp_df["Foreign_Buy_K"] > 0) & (temp_df["Sitca_Buy_K"] > 0) & (temp_df["Volume_K"] >= 1000)
        base_result = temp_df[condition].copy()

        if filter_bollinger or filter_ma:
            bb_signals, ma_signals, prices = [], [], []
            progress_bar = st.progress(0)
            total = len(base_result)

            for idx, (_, row) in enumerate(base_result.iterrows()):
                bb_sig, ma_sig, price = check_technical_signals(row["Code"])
                bb_signals.append(bb_sig)
                ma_signals.append(ma_sig)
                prices.append(price)
                if total > 0:
                    progress_bar.progress((idx + 1) / total)

            progress_bar.empty()
            base_result["BB_Signal"] = bb_signals
            base_result["MA_Signal"] = ma_signals
            base_result["Price"] = prices

            if filter_bollinger:
                base_result = base_result[base_result["BB_Signal"] == True]
            if filter_ma:
                base_result = base_result[base_result["MA_Signal"] == True]
        else:
            base_result["Price"] = 0.0

        result = base_result[["Code", "Name", "Price", "Volume_K", "Foreign_Buy_K", "Sitca_Buy_K"]].copy()
        result.columns = ["股票代號", "股票名稱", "收盤價", "成交量(張)", "外資買超(張)", "投信買超(張)"]
        return result

    col1, col2 = st.columns(2)
    with col1:
        sort_option = st.selectbox("📊 結果排序依據：", ["依股價/規模（高到低）", "依投信買超張數（高到低）", "依外資買超張數（高到低）"])
    with col2:
        enable_bb = st.checkbox("🔍 勾選：布林中軌（20MA）即將向上", value=False)
        enable_ma = st.checkbox("📈 勾選：5MA與10MA向上，20MA即將向上", value=False)

    if st.button("🚀 一鍵查詢符合條件股票", use_container_width=True):
        with st.spinner("正在自動尋找最新資料日並分析技術指標..."):
            raw_fund, trade_date = get_latest_twse_data()
            if not raw_fund.empty:
                result_df = process_and_filter(raw_fund, filter_bollinger=enable_bb, filter_ma=enable_ma)
                if not result_df.empty:
                    if "股價" in sort_option:
                        result_df = result_df.sort_values(by=["收盤價", "成交量(張)"], ascending=False)
                    elif "投信" in sort_option:
                        result_df = result_df.sort_values(by=["投信買超(張)", "外資買超(張)"], ascending=False)
                    elif "外資" in sort_option:
                        result_df = result_df.sort_values(by=["外資買超(張)", "投信買超(張)"], ascending=False)

                    st.success(f"📅 **數據日期：{trade_date}**｜成功篩選出 {len(result_df)} 支標的")
                    st.dataframe(result_df, hide_index=True, use_container_width=True)
                else:
                    st.info(f"📅 **數據日期：{trade_date}**｜目前條件下無符合標的。")
            else:
                st.error("❌ 無法連線至證交所或連線逾時，請稍後再試。")

# ==========================================
# 模組 3: 背離多重掃描
# ==========================================
elif page == "🔍 背離多重掃描":
    st.title("📈 台股背離多重掃描與分析")

    def get_stock_data_with_ta(ticker_symbol):
        df = fetch_yf_data(ticker_symbol, period="6m", interval="1d")
        if df is None or len(df) < 40:
            return None
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['K'] = stoch.stoch()
        df['D'] = stoch.stoch_signal()
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD_hist'] = macd.macd_diff()
        df['Vol_MA5'] = df['Volume'].rolling(5).mean()
        return df.dropna()

    def find_divergences_ending_in_two_weeks(df, total_days=120, recent_window=10, step_window=10):
        if df is None or len(df) < (recent_window + step_window):
            return {}

        sub_df = df.tail(total_days).copy()
        results = {}
        indicators = ['RSI', 'K', 'MACD_hist', 'Vol_MA5']
        div_names = {
            'RSI': ('RSI 底背離', 'RSI 頂背離'),
            'K': ('KD 底背離', 'KD 頂背離'),
            'MACD_hist': ('MACD 底背離', 'MACD 頂背離'),
            'Vol_MA5': ('價量底背離 (價跌量縮)', '價量頂背離 (價漲量縮)')
        }
        cutoff_date = df.index[-recent_window]

        for ind in indicators:
            bull_key, bear_key = div_names[ind]
            results[bull_key], results[bear_key] = [], []

            for i in range(len(sub_df) - step_window * 2):
                prev_chunk = sub_df.iloc[i : i + step_window]
                curr_chunk = sub_df.iloc[i + step_window : i + step_window * 2]

                p_low1, p_low2 = prev_chunk['Low'].min(), curr_chunk['Low'].min()
                p_high1, p_high2 = prev_chunk['High'].max(), curr_chunk['High'].max()
                i_low1, i_low2 = prev_chunk[ind].min(), curr_chunk[ind].min()
                i_high1, i_high2 = prev_chunk[ind].max(), curr_chunk[ind].max()

                if p_low2 < p_low1 and i_low2 > i_low1:
                    end_idx = curr_chunk['Low'].idxmin()
                    if end_idx >= cutoff_date:
                        start_dt = prev_chunk['Low'].idxmin().strftime('%Y-%m-%d')
                        end_dt = end_idx.strftime('%Y-%m-%d')
                        if not any(r['end_date'] == end_dt for r in results[bull_key]):
                            results[bull_key].append({
                                'start_date': start_dt, 'end_date': end_dt,
                                'start_price': float(p_low1), 'end_price': float(p_low2)
                            })

                if p_high2 > p_high1 and i_high2 < i_high1:
                    end_idx = curr_chunk['High'].idxmax()
                    if end_idx >= cutoff_date:
                        start_dt = prev_chunk['High'].idxmax().strftime('%Y-%m-%d')
                        end_dt = end_idx.strftime('%Y-%m-%d')
                        if not any(r['end_date'] == end_dt for r in results[bear_key]):
                            results[bear_key].append({
                                'start_date': start_dt, 'end_date': end_dt,
                                'start_price': float(p_high1), 'end_price': float(p_high2)
                            })
        return results

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 背離條件選擇")
    selected_conditions = []
    col_a, col_b = st.sidebar.columns(2)

    with col_a:
        st.markdown("**看多 (底背離)**")
        if st.checkbox("RSI 底背離", value=True): selected_conditions.append("RSI 底背離")
        if st.checkbox("KD 底背離"): selected_conditions.append("KD 底背離")
        if st.checkbox("MACD 底背離"): selected_conditions.append("MACD 底背離")
        if st.checkbox("價量底背離"): selected_conditions.append("價量底背離 (價跌量縮)")

    with col_b:
        st.markdown("**看空 (頂背離)**")
        if st.checkbox("RSI 頂背離"): selected_conditions.append("RSI 頂背離")
        if st.checkbox("KD 頂背離"): selected_conditions.append("KD 頂背離")
        if st.checkbox("MACD 頂背離"): selected_conditions.append("MACD 頂背離")
        if st.checkbox("價量頂背離"): selected_conditions.append("價量頂背離 (價漲量縮)")

    match_mode = st.sidebar.radio("篩選邏輯：", ["滿足「任一」勾選條件 (OR)", "必須「同時滿足」所有勾選條件 (AND)"])

    matched_stocks, stock_div_details = {}, {}

    if selected_conditions:
        with st.spinner("正在掃描近 2 週內出現背離的股票..."):
            for code, name in DEFAULT_STOCKS.items():
                symbol = f"{code}.TW"
                df = get_stock_data_with_ta(symbol)
                if df is not None:
                    divs = find_divergences_ending_in_two_weeks(df)
                    triggered_info = {cond: divs[cond] for cond in selected_conditions if len(divs.get(cond, [])) > 0}

                    if match_mode == "滿足「任一」勾選條件 (OR)" and len(triggered_info) > 0:
                        matched_stocks[symbol] = name
                        stock_div_details[symbol] = triggered_info
                    elif match_mode == "必須「同時滿足」所有勾選條件 (AND)" and len(triggered_info) == len(selected_conditions):
                        matched_stocks[symbol] = name
                        stock_div_details[symbol] = triggered_info

    if not selected_conditions:
        st.info("👈 請在側邊欄中至少勾選一種背離條件進行掃描。")
    elif not matched_stocks:
        st.warning("⚠️ 近 2 週內沒有發生符合背離條件的股票。")
    else:
        st.subheader(f"🎯 背離發生在近 2 週內的股票 (共 {len(matched_stocks)} 檔)")
        selected_stock = st.selectbox(
            "請選擇股票查看詳細分析與背離區間：",
            options=list(matched_stocks.keys()),
            format_func=lambda x: f"{x.replace('.TW','')} {matched_stocks[x]}"
        )

        if selected_stock:
            df = get_stock_data_with_ta(selected_stock)
            ticker_info = yf.Ticker(selected_stock)

            last_close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            change = last_close - prev_close
            pct_change = (change / prev_close) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("最新成交價", f"{last_close:.2f} 元", f"{change:+.2f} ({pct_change:+.2f}%)")
            c2.metric("成交量", f"{int(df['Volume'].iloc[-1]/1000):,} 張")
            c3.metric("近半年最高 / 最低", f"{df['High'].max():.1f} / {df['Low'].min():.1f}")

            st.markdown("### 📌 近 2 週發生的背離時間區間")
            details = stock_div_details[selected_stock]
            for div_type, occurrences in details.items():
                st.markdown(f"**🔹 {div_type}**")
                for occ in occurrences:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;• **背離區間**：`{occ['start_date']}` ({occ['start_price']:.1f}元) ➔ `{occ['end_date']}` ({occ['end_price']:.1f}元)")

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='blue')), row=3, col=1)

            for div_type, occurrences in details.items():
                color = "rgba(0, 255, 0, 0.2)" if "底背離" in div_type else "rgba(255, 0, 0, 0.2)"
                for occ in occurrences:
                    fig.add_vrect(x0=occ['start_date'], x1=occ['end_date'], fillcolor=color, opacity=0.5, layer="below", line_width=0, row=1, col=1)

            fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 基本面", "🏦 籌碼面", "📰 消息面", "📐 趨勢與支撐壓力", "🔮 未來一週展望"])
            with tab1:
                try:
                    info = ticker_info.info
                    st.markdown(f"- **產業類別**：{info.get('industry', 'N/A')}\n- **本益比 (PE)**：{info.get('trailingPE', 'N/A')}\n- **股價淨值比 (PB)**：{info.get('priceToBook', 'N/A')}\n- **殖利率**：{info.get('dividendYield', 0)*100:.2f}%")
                except Exception:
                    st.write("暫無詳細基本面資料。")
            with tab2:
                vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
                vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
                st.write(f"- **5日均量**：{int(vol_ma5/1000):,} 張\n- **20日均量**：{int(vol_ma20/1000):,} 張")
            with tab3:
                try:
                    news = ticker_info.news
                    if news:
                        for item in news[:3]:
                            st.markdown(f"- [{item.get('title')}]({item.get('link')})")
                    else:
                        st.write("目前無新聞。")
                except Exception:
                    st.write("無法讀取新聞。")
            with tab4:
                high_half_year = df['High'].max()
                low_half_year = df['Low'].min()
                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                st.write(f"- **近半年最高壓力位**：{high_half_year:.2f} 元\n- **月線 (20MA)**：{ma20:.2f} 元\n- **近半年最低支撐位**：{low_half_year:.2f} 元")
            with tab5:
                has_bull = any("底背離" in d for d in details.keys())
                if has_bull:
                    st.success(f"**短線看多 / 止跌反彈機會**\n近 2 週出現【{', '.join(details.keys())}】，代表近期探底過程中指標已不破低，短線具備止跌反彈力道，初步目標看至 20MA ({ma20:.2f} 元)。")
                else:
                    st.error(f"**短線看空 / 回測風險**\n近 2 週出現【{', '.join(details.keys())}】，代表股價高點雖創高但指標力道減弱，注意逢高拉回修正，支撐觀察 20MA ({ma20:.2f} 元)。")

# ==========================================
# 模組 4: 關鍵轉折 K線篩選器
# ==========================================
elif page == "⚡ 關鍵轉折 K線篩選":
    st.title("📈 台股關鍵轉折 K 線篩選器")
    st.caption("結合 K 線型態、爆量突破、基本面與籌碼面分析")

    stock_input = st.text_area("請輸入台股代碼 (以逗點分隔)", "2330, 2317, 2454, 2308, 2382, 3231, 2356")
    stock_list = [f"{code.strip()}.TW" for code in stock_input.split(",") if code.strip()]

    def check_reversal_pattern(df):
        if len(df) < 20:
            return False, {}

        prev_day = df.iloc[-2]
        yesterday = df.iloc[-1]

        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]

        is_prev_bearish = prev_day['Close'] < prev_day['Open']
        is_yest_bullish = yesterday['Close'] > yesterday['Open']
        is_engulfing = (yesterday['Open'] <= prev_day['Close']) and (yesterday['Close'] >= prev_day['Open'])

        is_volume_up = yesterday['Volume'] > (vol_ma20 * 1.2)
        is_above_ma20 = yesterday['Close'] > ma20

        is_turnaround = is_prev_bearish and is_yest_bullish and is_engulfing and is_volume_up and is_above_ma20

        info = {
            "date": yesterday.name.strftime('%Y-%m-%d'),
            "close": round(float(yesterday['Close']), 2),
            "volume": int(yesterday['Volume']),
            "ma20": round(float(ma20), 2),
            "low_support": round(float(yesterday['Low']), 2),
            "high_resistance": round(float(df['High'].tail(10).max()), 2)
        }
        return is_turnaround, info

    def get_chip_and_fundamental(stock_id):
        if not FINMIND_AVAILABLE:
            return "FinMind 套件未安裝", "FinMind 套件未安裝"

        dl = DataLoader()
        clean_id = stock_id.replace(".TW", "")

        try:
            chip_df = dl.taiwan_stock_institutional_investors(stock_id=clean_id, start_date="2024-01-01")
            recent_chip = chip_df.tail(3)['buy'].sum() - chip_df.tail(3)['sell'].sum()
            chip_status = f"近3日三大法人淨買超 {int(recent_chip)} 張"
        except Exception:
            chip_status = "籌碼面資料無回應"

        try:
            per_df = dl.taiwan_stock_per_pbr(stock_id=clean_id, start_date="2024-01-01")
            latest_per = per_df.iloc[-1]['PER']
            fundamental_status = f"本益比 (PER): {latest_per}"
        except Exception:
            fundamental_status = "基本面資料無回應"

        return chip_status, fundamental_status

    if st.button("開始掃描轉折股", type="primary"):
        results = []
        with st.spinner("正在抓取行情與籌碼資料中..."):
            for ticker in stock_list:
                df = fetch_yf_data(ticker, period="60d", interval="1d")
                if df is None:
                    continue

                is_turn, info = check_reversal_pattern(df)
                if is_turn:
                    chip, fundamental = get_chip_and_fundamental(ticker)
                    support, resistance, close = info['low_support'], info['high_resistance'], info['close']

                    trend = f"強勢突破！關鍵防守價（昨日低點）：{support} 元。" if close >= resistance else f"築底反彈，預計在 {support} ~ {resistance} 元震盪，守住 {support} 元偏多。"

                    results.append({
                        "股票代碼": ticker.replace(".TW", ""),
                        "轉折日期": info['date'],
                        "最新收盤價": info['close'],
                        "基本面": fundamental,
                        "籌碼面": chip,
                        "未來一週走勢研判": trend
                    })

        if results:
            st.success(f"掃描完成！共有 {len(results)} 檔符合條件：")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("輸入的股票清單中，沒有出現符合關鍵轉折條件的標的。")

# ==========================================
# 模組 5: 證交所/櫃買全方位分析
# ==========================================
elif page == "🏛️ 證交所/櫃買全方位分析":
    st.title("📈 台股全方位分析系統 (證交所/櫃買官方資料源)")
    st.caption("直接對接臺灣證券交易所與櫃買中心官方數據，進行技術面與籌碼面分析")

    @st.cache_data(ttl=86400)
    def get_taiwan_stock_list():
        options, stock_map = [], {}
        default_options = [
            "2330 台積電 (上市)", "2317 鴻海 (上市)", "2408 南亞科 (上市)",
            "2454 聯發科 (上市)", "2382 廣達 (上市)", "8069 元太 (上櫃)"
        ]
        default_map = {
            "2330 台積電 (上市)": {"stock_id": "2330", "stock_name": "台積電", "market_type": "上市"},
            "2317 鴻海 (上市)": {"stock_id": "2317", "stock_name": "鴻海", "market_type": "上市"},
            "2408 南亞科 (上市)": {"stock_id": "2408", "stock_name": "南亞科", "market_type": "上市"},
            "2454 聯發科 (上市)": {"stock_id": "2454", "stock_name": "聯發科", "market_type": "上市"},
            "2382 廣達 (上市)": {"stock_id": "2382", "stock_name": "廣達", "market_type": "上市"},
            "8069 元太 (上櫃)": {"stock_id": "8069", "stock_name": "元太", "market_type": "上櫃"}
        }

        try:
            res = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL", headers=HTTP_HEADERS, timeout=6)
            if res.status_code == 200:
                for item in res.json():
                    sid, sname = item.get("Code", "").strip(), item.get("Name", "").strip()
                    if sid and len(sid) == 4 and sid.isdigit():
                        label = f"{sid} {sname} (上市)"
                        options.append(label)
                        stock_map[label] = {"stock_id": sid, "stock_name": sname, "market_type": "上市"}

            res_tpex = requests.get("https://www.tpex.org.tw/openapi/v1/mopsrev_result", headers=HTTP_HEADERS, timeout=6)
            if res_tpex.status_code == 200:
                for item in res_tpex.json():
                    sid, sname = item.get("CompanyCode", "").strip(), item.get("CompanyName", "").strip()
                    if sid and len(sid) == 4 and sid.isdigit():
                        label = f"{sid} {sname} (上櫃)"
                        if label not in stock_map:
                            options.append(label)
                            stock_map[label] = {"stock_id": sid, "stock_name": sname, "market_type": "上櫃"}
        except Exception:
            pass

        return (options, stock_map) if options else (default_options, default_map)

    stock_options, stock_map = get_taiwan_stock_list()
    selected_stock_label = st.selectbox("輸入股票代號或中文名稱進行搜尋", options=stock_options, index=0)

    selected_info = stock_map[selected_stock_label]
    stock_id = selected_info["stock_id"]
    stock_name = selected_info["stock_name"]
    market_type = selected_info["market_type"]

    @st.cache_data(ttl=3600)
    def fetch_twse_tpex_kline(stock_id, market_type):
        today = datetime.today().date()
        all_data = []

        for i in range(6):
            target_date = today - timedelta(days=i * 28)
            date_str = target_date.strftime("%Y%m01")

            try:
                if market_type == "上市":
                    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={stock_id}"
                    res = requests.get(url, headers=HTTP_HEADERS, timeout=5).json()
                    if res.get("stat") == "OK" and "data" in res:
                        all_data.extend(res["data"])
                else:
                    roc_year = target_date.year - 1911
                    roc_date_str = f"{roc_year}/{target_date.strftime('%m')}"
                    url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={roc_date_str}&stkno={stock_id}"
                    res = requests.get(url, headers=HTTP_HEADERS, timeout=5).json()
                    if "aaData" in res:
                        all_data.extend(res["aaData"])
            except Exception:
                continue

        if not all_data:
            return None

        parsed_rows = []
        for row in all_data:
            try:
                parts = row[0].replace(" ", "").split("/")
                year = int(parts[0]) + 1911
                date_fmt = f"{year}-{parts[1]:>02}-{parts[2]:>02}"

                parsed_rows.append({
                    "date": pd.to_datetime(date_fmt),
                    "Volume": float(row[1].replace(",", "")),
                    "Open": float(row[3].replace(",", "")),
                    "High": float(row[4].replace(",", "")),
                    "Low": float(row[5].replace(",", "")),
                    "Close": float(row[6].replace(",", ""))
                })
            except Exception:
                continue

        if not parsed_rows:
            return None

        return pd.DataFrame(parsed_rows).drop_duplicates(subset=['date']).sort_values('date').set_index('date')

    def calculate_indicators(df):
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['DIF'] - df['MACD_Signal']

        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        rsv = ((df['Close'] - low_min) / (high_max - low_min) * 100).fillna(50)

        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        return df

    with st.spinner("正在讀取證交所/櫃買中心官方數據..."):
        df_official = fetch_twse_tpex_kline(stock_id, market_type)

    if df_official is None or len(df_official) < 10:
        st.error("⚠️ 證交所 API 暫無回應或查無此股票，請稍後重試！")
    else:
        df_official = calculate_indicators(df_official)
        latest = df_official.iloc[-1]
        prev_close = float(df_official.iloc[-2]['Close'])
        close_price = float(latest['Close'])
        pct_change = ((close_price - prev_close) / prev_close) * 100

        st.subheader(f"📊 {stock_name} ({stock_id}) 核心行情")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤價", f"{close_price:.2f} 元", f"{pct_change:+.2f}%")
        c2.metric("最高價", f"{latest['High']:.2f} 元")
        c3.metric("最低價", f"{latest['Low']:.2f} 元")
        c4.metric("成交量", f"{int(latest['Volume']/1000):,} 張")

        df_chart = df_official.tail(80)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA5'], name='MA5'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], name='MA20'), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['K'], name='K值'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['D'], name='D值'), row=2, col=1)

        colors_macd = ['red' if h >= 0 else 'green' for h in df_chart['MACD_Hist']]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['MACD_Hist'], marker_color=colors_macd, name='MACD'), row=3, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
