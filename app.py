from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

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
    "Referer": "https://www.twse.com.tw/zh/trading/foreign/t86.html",
}

DEFAULT_STOCKS = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2382": "廣達",
    "3231": "緯創", "2308": "台達電", "2303": "聯電", "2881": "富邦金",
    "2882": "國泰金", "2603": "長榮", "2609": "陽明", "3037": "欣興",
    "2345": "智邦", "6669": "緯穎", "3661": "世芯-KY", "3035": "智原",
    "2357": "華碩", "2356": "英業達", "2002": "中鋼", "1301": "台塑"
}

def fetch_twse_kline_backup(symbol):
    """備援機制：當 yfinance 失敗時，直接向證交所 API 獲取最新的 K 線數據"""
    session = requests.Session()
    session.headers.update(HTTP_HEADERS)
    today = datetime.today()
    all_data = []

    for i in range(3):  # 抓近 3 個月的資料
        target_date = today - timedelta(days=i * 28)
        date_str = target_date.strftime("%Y%m01")
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={symbol}"
        try:
            res = session.get(url, timeout=4).json()
            if res.get("stat") == "OK" and "data" in res:
                all_data.extend(res["data"])
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
                "Date": date_fmt,
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

    df = pd.DataFrame(parsed_rows).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df

@st.cache_data(ttl=1800)
def fetch_robust_stock_data(symbol):
    """強健版個股 K 線資料抓取：首選 yfinance，失敗時自動啟用證交所備援"""
    ticker_symbol = f"{symbol}.TW"
    df = None

    # 方法 A: 嘗試使用 yfinance
    try:
        yf_df = yf.download(ticker_symbol, period="6m", interval="1d", progress=False)
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            
            yf_df = yf_df.reset_index()
            # 統一欄位名稱為首字大寫
            yf_df.rename(columns={c: c.capitalize() for c in yf_df.columns}, inplace=True)
            
            if "Close" in yf_df.columns and len(yf_df.dropna(subset=["Close"])) >= 10:
                yf_df["Date"] = pd.to_datetime(yf_df["Date"]).dt.strftime("%Y-%m-%d")
                df = yf_df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
                df["Close"] = pd.to_numeric(df["Close"].values.flatten(), errors="coerce")
                df["Open"] = pd.to_numeric(df["Open"].values.flatten(), errors="coerce")
                df["High"] = pd.to_numeric(df["High"].values.flatten(), errors="coerce")
                df["Low"] = pd.to_numeric(df["Low"].values.flatten(), errors="coerce")
                df["Volume"] = pd.to_numeric(df["Volume"].values.flatten(), errors="coerce")
                df = df.dropna()
    except Exception:
        df = None

    # 方法 B: 若 yfinance 失敗或無數據，切換至證交所官方 API 備援
    if df is None or len(df) < 10:
        df = fetch_twse_kline_backup(symbol)

    return df

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
# 模組 1: 5日波段選股 (終極雙重備援版)
# ==========================================
if page == "🚀 5日波段選股":
    st.title("📈 台股 5日波段選股 App")

    sort_by = st.radio(
        "排序方式：",
        options=["模型綜合評分 (Score)", "成交金額大小"],
        horizontal=True,
    )

    @st.cache_data(ttl=1800)
    def run_5day_strategy(stock_dict):
        results = []
        latest_trade_date = ""

        for symbol, name in stock_dict.items():
            try:
                hist_df = fetch_robust_stock_data(symbol)
                if hist_df is None or len(hist_df) < 15:
                    continue

                close_s = pd.Series(hist_df["Close"].values, index=hist_df.index)
                latest_close = float(close_s.iloc[-1])
                prev_close = float(close_s.iloc[-2])
                change = latest_close - prev_close
                pct_change = round((change / prev_close) * 100, 2)
                vol = int(hist_df["Volume"].iloc[-1] / 1000)

                hist_df["MA5"] = close_s.rolling(5).mean()
                hist_df["MA10"] = close_s.rolling(10).mean()

                ema12 = close_s.ewm(span=12, adjust=False).mean()
                ema26 = close_s.ewm(span=26, adjust=False).mean()
                hist_df["DIF"] = ema12 - ema26
                hist_df["DEM"] = hist_df["DIF"].ewm(span=9, adjust=False).mean()
                hist_df["MACD_Hist"] = hist_df["DIF"] - hist_df["DEM"]

                c_ma5 = float(hist_df["MA5"].iloc[-1])
                c_ma10 = float(hist_df["MA10"].iloc[-1])

                score = 50
                signals = []
                if latest_close > c_ma5:
                    score += 25
                    signals.append("站上5日線")
                if latest_close > c_ma10:
                    score += 25
                    signals.append("站上10日線")

                latest_trade_date = str(hist_df["Date"].iloc[-1])

                results.append({
                    "代號": symbol,
                    "名稱": name,
                    "綜合評分": score,
                    "收盤價": round(latest_close, 2),
                    "漲跌幅(%)": pct_change,
                    "建議停利": round(latest_close * 1.08, 2),
                    "建議停損": round(c_ma10, 2),
                    "成交量(張)": vol,
                    "訊號": ", ".join(signals),
                    "交易日期": latest_trade_date,
                    "df": hist_df,
                })
            except Exception:
                continue

        return pd.DataFrame(results), latest_trade_date

    with st.spinner("正在載入 5日波段數據（含雙重資料源備援）..."):
        df_res, data_date = run_5day_strategy(DEFAULT_STOCKS)

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
    st.caption("自動抓取證交所最新籌碼資料，結合布林中軌（20MA）與短均線向上訊號")

    @st.cache_data(ttl=1800)
    def fetch_latest_twse_fund():
        session = requests.Session()
        session.headers.update(HTTP_HEADERS)
        
        current_date = datetime.now()
        for i in range(10):
            target_date = current_date - timedelta(days=i)
            date_str = target_date.strftime("%Y%m%d")
            url = f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={date_str}&selectType=ALL"

            try:
                res = session.get(url, timeout=6)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("stat") == "OK" and "data" in data:
                        cols = [str(c).strip() for c in data["fields"]]
                        df = pd.DataFrame(data["data"], columns=cols)
                        return df, target_date.strftime("%Y/%m/%d")
            except Exception:
                continue
        return pd.DataFrame(), None

    @st.cache_data(ttl=1800)
    def check_technical_signals(stock_code):
        try:
            hist = fetch_robust_stock_data(stock_code)
            if hist is None or len(hist) < 25:
                return False, False, 0.0

            close = pd.Series(hist["Close"].values, index=hist.index)
            hist["MA5"] = close.rolling(5).mean()
            hist["MA10"] = close.rolling(10).mean()
            hist["MA20"] = close.rolling(20).mean()

            latest_close = float(close.iloc[-1])

            ma20_today = float(hist["MA20"].iloc[-1])
            ma20_prev = float(hist["MA20"].iloc[-2])
            ma20_prev2 = float(hist["MA20"].iloc[-3])

            slope_today = ma20_today - ma20_prev
            slope_prev = ma20_prev - ma20_prev2

            bb_middle_turning_up = (slope_today > 0) or (slope_today > slope_prev and slope_today >= -0.15)

            ma5_up = float(hist["MA5"].iloc[-1]) > float(hist["MA5"].iloc[-2])
            ma10_up = float(hist["MA10"].iloc[-1]) > float(hist["MA10"].iloc[-2])
            ma_signal = ma5_up and ma10_up and bb_middle_turning_up

            return bb_middle_turning_up, ma_signal, round(latest_close, 2)
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

        condition = (temp_df["Foreign_Buy_K"] > 0) & (temp_df["Sitca_Buy_K"] > 0) & (temp_df["Volume_K"] >= 500)
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
        sort_option = st.selectbox("📊 結果排序依據：", ["依投信買超張數（高到低）", "依外資買超張數（高到低）", "依成交量（高到低）"])
    with col2:
        enable_bb = st.checkbox("🔍 勾選：布林中軌（20MA）即將向上", value=False)
        enable_ma = st.checkbox("📈 勾選：5MA與10MA向上，20MA即將向上", value=False)

    if st.button("🚀 一鍵查詢符合條件股票", use_container_width=True):
        with st.spinner("正在自動尋找證交所最近交易日數據並分析技術指標..."):
            raw_fund, trade_date = fetch_latest_twse_fund()
            if not raw_fund.empty:
                result_df = process_and_filter(raw_fund, filter_bollinger=enable_bb, filter_ma=enable_ma)
                if not result_df.empty:
                    if "投信" in sort_option:
                        result_df = result_df.sort_values(by=["投信買超(張)", "外資買超(張)"], ascending=False)
                    elif "外資" in sort_option:
                        result_df = result_df.sort_values(by=["外資買超(張)", "投信買超(張)"], ascending=False)
                    elif "成交量" in sort_option:
                        result_df = result_df.sort_values(by=["成交量(張)", "投信買超(張)"], ascending=False)

                    st.success(f"📅 **數據日期：{trade_date}**｜成功篩選出 {len(result_df)} 支標的")
                    st.dataframe(result_df, hide_index=True, use_container_width=True)
                else:
                    st.info(f"📅 **數據日期：{trade_date}**｜目前條件下無符合標的。")
            else:
                st.error("❌ 無法連線至證交所或獲取數據，請稍後再試。")

# ==========================================
# 模組 3: 背離多重掃描 (修復含 MACD 圖表)
# ==========================================
elif page == "🔍 背離多重掃描":
    st.title("📈 台股背離多重掃描與分析")

    def get_stock_data_with_ta(symbol):
        df = fetch_robust_stock_data(symbol.replace(".TW", ""))
        if df is None or len(df) < 30:
            return None
        
        df = df.set_index("Date")
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['K'] = stoch.stoch()
        df['D'] = stoch.stoch_signal()
        
        # MACD 指標完整計算
        macd = ta.trend.MACD(close=df['Close'])
        df['DIF'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
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
                        start_dt = str(prev_chunk['Low'].idxmin())
                        end_dt = str(end_idx)
                        if not any(r['end_date'] == end_dt for r in results[bull_key]):
                            results[bull_key].append({
                                'start_date': start_dt, 'end_date': end_dt,
                                'start_price': float(p_low1), 'end_price': float(p_low2)
                            })

                if p_high2 > p_high1 and i_high2 < i_high1:
                    end_idx = curr_chunk['High'].idxmax()
                    if end_idx >= cutoff_date:
                        start_dt = str(prev_chunk['High'].idxmax())
                        end_dt = str(end_idx)
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
                df = get_stock_data_with_ta(code)
                if df is not None:
                    divs = find_divergences_ending_in_two_weeks(df)
                    triggered_info = {cond: divs[cond] for cond in selected_conditions if len(divs.get(cond, [])) > 0}

                    if match_mode == "滿足「任一」勾選條件 (OR)" and len(triggered_info) > 0:
                        matched_stocks[code] = name
                        stock_div_details[code] = triggered_info
                    elif match_mode == "必須「同時滿足」所有勾選條件 (AND)" and len(triggered_info) == len(selected_conditions):
                        matched_stocks[code] = name
                        stock_div_details[code] = triggered_info

    if not selected_conditions:
        st.info("👈 請在側邊欄中至少勾選一種背離條件進行掃描。")
    elif not matched_stocks:
        st.warning("⚠️ 近 2 週內沒有發生符合背離條件的股票。")
    else:
        st.subheader(f"🎯 背離發生在近 2 週內的股票 (共 {len(matched_stocks)} 檔)")
        selected_stock = st.selectbox(
            "請選擇股票查看詳細分析與背離區間：",
            options=list(matched_stocks.keys()),
            format_func=lambda x: f"{x} {matched_stocks[x]}"
        )

        if selected_stock:
            df = get_stock_data_with_ta(selected_stock)

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

            # 設定 4 行子圖 (含獨立 MACD)
            fig = make_subplots(
                rows=4, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=(f"{matched_stocks[selected_stock]} ({selected_stock}) K線", "成交量", "RSI / KD 指標", "MACD 指標")
            )
            
            # Row 1: K線
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            
            # Row 2: 成交量
            colors = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume']/1000, name='成交量(張)', marker_color=colors), row=2, col=1)
            
            # Row 3: RSI & KD
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D', line=dict(color='purple')), row=3, col=1)

            # Row 4: MACD 柱狀圖 + 雙線
            macd_colors = np.where(df['MACD_hist'] >= 0, '#ef5350', '#26a69a')
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD柱狀', marker_color=macd_colors), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['DIF'], name='DIF(快線)', line=dict(color='#2962FF', width=1.5)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD(慢線)', line=dict(color='#FF6D00', width=1.5)), row=4, col=1)

            fig.update_layout(
                height=700, 
                template="plotly_dark",
                margin=dict(l=10, r=10, t=30, b=10), 
                xaxis_rangeslider_visible=False,
                xaxis=dict(type="category"),
                xaxis2=dict(type="category"),
                xaxis3=dict(type="category"),
                xaxis4=dict(type="category"),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 模組 4: 關鍵轉折 K線篩選器
# ==========================================
elif page == "⚡ 關鍵轉折 K線篩選":
    st.title("📈 台股關鍵轉折 K 線篩選器")
    st.caption("結合 K 線型態與爆量突破分析")

    stock_input = st.text_area("請輸入台股代碼 (以逗點分隔)", "2330, 2317, 2454, 2308, 2382, 3231, 2356")
    stock_list = [code.strip() for code in stock_input.split(",") if code.strip()]

    def check_reversal_pattern(df):
        if len(df) < 20:
            return False, {}

        prev_day = df.iloc[-2]
        yesterday = df.iloc[-1]

        ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
        vol_ma20 = float(df['Volume'].rolling(20).mean().iloc[-1])

        is_prev_bearish = float(prev_day['Close']) < float(prev_day['Open'])
        is_yest_bullish = float(yesterday['Close']) > float(yesterday['Open'])
        is_engulfing = (float(yesterday['Open']) <= float(prev_day['Close'])) and (float(yesterday['Close']) >= float(prev_day['Open']))

        is_volume_up = float(yesterday['Volume']) > (vol_ma20 * 1.2)
        is_above_ma20 = float(yesterday['Close']) > ma20

        is_turnaround = is_prev_bearish and is_yest_bullish and is_engulfing and is_volume_up and is_above_ma20

        info = {
            "date": str(yesterday['Date']),
            "close": round(float(yesterday['Close']), 2),
            "volume": int(yesterday['Volume']),
            "ma20": round(ma20, 2),
            "low_support": round(float(yesterday['Low']), 2),
            "high_resistance": round(float(df['High'].tail(10).max()), 2)
        }
        return is_turnaround, info

    if st.button("開始掃描轉折股", type="primary"):
        results = []
        with st.spinner("正在抓取行情資料中..."):
            for ticker in stock_list:
                df = fetch_robust_stock_data(ticker)
                if df is None:
                    continue

                is_turn, info = check_reversal_pattern(df)
                if is_turn:
                    support, resistance, close = info['low_support'], info['high_resistance'], info['close']
                    trend = f"強勢突破！關鍵防守價（昨日低點）：{support} 元。" if close >= resistance else f"築底反彈，預計在 {support} ~ {resistance} 元震盪，守住 {support} 元偏多。"

                    results.append({
                        "股票代碼": ticker,
                        "轉折日期": info['date'],
                        "最新收盤價": info['close'],
                        "成交量(張)": int(info['volume'] / 1000),
                        "未來一週走勢研判": trend
                    })

        if results:
            st.success(f"掃描完成！共有 {len(results)} 檔符合條件：")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("輸入的股票清單中，目前沒有出現符合關鍵轉折吞噬型態的標的。")

# ==========================================
# 模組 5: 證交所/櫃買全方位分析
# ==========================================
elif page == "🏛️ 證交所/櫃買全方位分析":
    st.title("📈 台股全方位分析系統 (技術/籌碼/基本面/動能)")

    stock_id = st.text_input("請輸入台股代碼：", value="2330")

    @st.cache_data(ttl=1800)
    def fetch_fundamental_and_chip(symbol):
        """獲取基本面指標與三大法人籌碼數據"""
        info_data = {"pe": "N/A", "yield": "N/A", "market_cap": "N/A", "foreign_buy": 0, "sitca_buy": 0}
        
        # 1. 基本面數據 (yfinance)
        try:
            ticker = yf.Ticker(f"{symbol}.TW")
            info = ticker.info
            
            pe = info.get("trailingPE") or info.get("forwardPE")
            info_data["pe"] = f"{pe:.2f}" if pe else "N/A"
            
            dy = info.get("dividendYield")
            info_data["yield"] = f"{dy * 100:.2f}%" if dy else "N/A"
            
            mc = info.get("marketCap")
            info_data["market_cap"] = f"{mc / 1e8:.2f} 億" if mc else "N/A"
        except Exception:
            pass

        # 2. 法人籌碼數據 (證交所 T86 API)
        try:
            session = requests.Session()
            session.headers.update(HTTP_HEADERS)
            for i in range(7):
                t_date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                url = f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={t_date}&selectType=ALL"
                res = session.get(url, timeout=3).json()
                if res.get("stat") == "OK" and "data" in res:
                    cols = [str(c).strip() for c in res["fields"]]
                    df_chip = pd.DataFrame(res["data"], columns=cols)
                    code_col = next(c for c in df_chip.columns if "代號" in c)
                    row = df_chip[df_chip[code_col].astype(str).str.strip() == symbol]
                    
                    if not row.empty:
                        f_col = next(c for c in df_chip.columns if "外資" in c or "外陸資" in c)
                        s_col = next(c for c in df_chip.columns if "投信" in c)
                        f_val = float(row[f_col].values[0].replace(",", "").strip())
                        s_val = float(row[s_col].values[0].replace(",", "").strip())
                        info_data["foreign_buy"] = int(f_val / 1000)
                        info_data["sitca_buy"] = int(s_val / 1000)
                        break
        except Exception:
            pass

        return info_data

    if stock_id:
        with st.spinner("正在讀取技術面、籌碼面與基本面數據..."):
            df_official = fetch_robust_stock_data(stock_id)
            extra_info = fetch_fundamental_and_chip(stock_id)

        if df_official is None or len(df_official) < 20:
            st.error("⚠️ 暫無數據或數據筆數不足，請確認股票代碼是否正確！")
        else:
            # 均線指標計算 (5MA, 10MA, 20MA)
            close = pd.Series(df_official["Close"].values, index=df_official.index)
            df_official['MA5'] = close.rolling(5).mean()
            df_official['MA10'] = close.rolling(10).mean()
            df_official['MA20'] = close.rolling(20).mean()

            latest = df_official.iloc[-1]
            prev_close = float(df_official.iloc[-2]['Close'])
            close_price = float(latest['Close'])
            pct_change = ((close_price - prev_close) / prev_close) * 100

            ma5_val = float(latest['MA5'])
            ma10_val = float(latest['MA10'])
            ma20_val = float(latest['MA20'])

            # 1. 核心行情與 5MA / 10MA / 20MA 數值
            st.subheader(f"📊 {stock_id} 核心行情與均線數值")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("最新收盤價", f"{close_price:.2f} 元", f"{pct_change:+.2f}%")
            m2.metric("5 日均線 (5MA)", f"{ma5_val:.2f} 元")
            m3.metric("10 日均線 (10MA)", f"{ma10_val:.2f} 元")
            m4.metric("20 日均線 (20MA)", f"{ma20_val:.2f} 元")
            m5.metric("當日成交量", f"{int(latest['Volume']/1000):,} 張")

            # 2. 外資/投信買賣超與基本面指標
            st.markdown("---")
            c_chip1, c_chip2, c_fund1, c_fund2, c_fund3 = st.columns(5)
            
            f_buy = extra_info["foreign_buy"]
            s_buy = extra_info["sitca_buy"]
            c_chip1.metric("外資買賣超", f"{f_buy:+} 張", delta_color="normal")
            c_chip2.metric("投信買賣超", f"{s_buy:+} 張", delta_color="normal")
            
            c_fund1.metric("本益比 (P/E)", extra_info["pe"])
            c_fund2.metric("現金殖利率", extra_info["yield"])
            c_fund3.metric("總市值", extra_info["market_cap"])

            # 3. 動能綜合判斷卡片
            st.markdown("---")
            st.subheader("⚡ 動能與趨勢強弱診斷")
            
            vol_ma5 = df_official['Volume'].tail(5).mean()
            vol_status = "放量" if latest['Volume'] > vol_ma5 else "縮量"
            
            if close_price > ma5_val > ma10_val > ma20_val:
                momentum_status = "🔥 **強勢多頭**：價格站上所有均線且多頭排列，上攻動能強勁。"
            elif close_price < ma5_val < ma10_val < ma20_val:
                momentum_status = "❄️ **空頭修正**：價格低於所有均線且空頭排列，注意下行風險。"
            elif close_price > ma20_val:
                momentum_status = "⚖️ **偏多震盪**：站上 20 日月線支撐，短線處於整理打底或反彈格局。"
            else:
                momentum_status = "⚠️ **偏空震盪**：跌破 20 日月線，短線動能放緩。"

            st.info(f"{momentum_status}（當前成交量狀態：**{vol_status}**）")

            # 4. 技術 K 線圖表
            df_chart = df_official.tail(80)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # K 線
            fig.add_trace(go.Candlestick(
                x=df_chart['Date'], open=df_chart['Open'], high=df_chart['High'], 
                low=df_chart['Low'], close=df_chart['Close'], name='K線',
                increasing_line_color="#ef5350", decreasing_line_color="#26a69a"
            ), row=1, col=1)
            
            # 5MA, 10MA, 20MA 均線
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA5'], name='5MA', line=dict(color='#2962FF', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA10'], name='10MA', line=dict(color='#FF6D00', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA20'], name='20MA', line=dict(color='#E040FB', width=1.5)), row=1, col=1)

            # 成交量
            colors_vol = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(df_chart['Close'], df_chart['Open'])]
            fig.add_trace(go.Bar(x=df_chart['Date'], y=df_chart['Volume']/1000, marker_color=colors_vol, name='成交量(張)'), row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False, 
                height=580, 
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(type="category"), 
                xaxis2=dict(type="category")
            )
            st.plotly_chart(fig, use_container_width=True)
