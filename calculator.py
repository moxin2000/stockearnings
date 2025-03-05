"""
DISCLAIMER:

This software is provided solely for educational and research purposes.
It is not intended to provide investment advice, and no investment recommendations are made herein.
The developers are not financial advisors and accept no responsibility for any financial decisions or losses resulting from the use of this software.
Always consult a professional financial advisor before making any investment decisions.
"""

import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import time


def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)

    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i + 1]]
            break

    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc ** 2

    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc ** 2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()


def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:
            return float(spline(dte))

    return term_spline


def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]


@st.cache_data
def compute_recommendation(ticker):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."

        try:
            stock = yf.Ticker(ticker)
            if len(stock.options) == 0:
                raise KeyError()
        except KeyError:
            return f"Error: No options found for stock symbol '{ticker}'."

        exp_dates = list(stock.options)
        try:
            exp_dates = filter_dates(exp_dates)
        except:
            return "Error: Not enough option data."

        options_chains = {}
        for exp_date in exp_dates:
            options_chains[exp_date] = stock.option_chain(exp_date)

        try:
            underlying_price = get_current_price(stock)
            if underlying_price is None:
                raise ValueError("No market price found.")
        except Exception:
            return "Error: Unable to retrieve underlying stock price."

        atm_iv = {}
        straddle = None
        i = 0
        for exp_date, chain in options_chains.items():
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                continue

            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']

            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']

            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value

            if i == 0:
                call_bid = calls.loc[call_idx, 'bid']
                call_ask = calls.loc[call_idx, 'ask']
                put_bid = puts.loc[put_idx, 'bid']
                put_ask = puts.loc[put_idx, 'ask']

                if call_bid is not None and call_ask is not None:
                    call_mid = (call_bid + call_ask) / 2.0
                else:
                    call_mid = None

                if put_bid is not None and put_ask is not None:
                    put_mid = (put_bid + put_ask) / 2.0
                else:
                    put_mid = None

                if call_mid is not None and put_mid is not None:
                    straddle = (call_mid + put_mid)

            i += 1

        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."

        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)

        term_spline = build_term_structure(dtes, ivs)

        ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])

        price_history = stock.history(period='3mo')
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)

        avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]

        expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None

        return {'avg_volume': avg_volume >= 1500000, 'iv30_rv30': iv30_rv30 >= 1.25,
                'ts_slope_0_45': ts_slope_0_45 <= -0.00406,
                'expected_move': expected_move}  # Check that they are in our desired range (see video)
    except Exception as e:
        return f'Error occurred processing: {e}'


def fetch_earnings_for_date(date):
    try:
        date_str = date.strftime("%Y-%m-%d")
        url = f"https://finance.yahoo.com/calendar/earnings?day={date_str}"
        earnings_df = pd.read_html(url, attrs={'class': 'W(100%)'}, flavor='lxml')[0]  # Read the first table
        return earnings_df
    except ImportError:
        st.error("Error: The 'lxml' library is required. Please install it using: pip install lxml")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching earnings data: {e}")
        return pd.DataFrame()


def analyze_earnings_data(earnings_df):
    results = []
    for index, row in earnings_df.iterrows():
        ticker = row['Symbol']
        try:
            # Add a delay to avoid rate limiting
            time.sleep(1)

            recommendation = compute_recommendation(ticker)

            if isinstance(recommendation, str):
                results.append({'Symbol': ticker, 'Recommendation': recommendation})
            else:
                avg_volume_bool = recommendation['avg_volume']
                iv30_rv30_bool = recommendation['iv30_rv30']
                ts_slope_bool = recommendation['ts_slope_0_45']
                expected_move = recommendation['expected_move']

                if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
                    title = "Recommended"
                elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (
                        iv30_rv30_bool and not avg_volume_bool)):
                    title = "Consider"
                else:
                    title = "Avoid"

                results.append({
                    'Symbol': ticker,
                    'Recommendation': title,
                    'Avg Volume': 'PASS' if avg_volume_bool else 'FAIL',
                    'IV30/RV30': 'PASS' if iv30_rv30_bool else 'FAIL',
                    'Term Structure Slope': 'PASS' if ts_slope_bool else 'FAIL',
                    'Expected Move': expected_move
                })
        except Exception as e:
            results.append({'Symbol': ticker, 'Recommendation': f'Analysis Error: {e}'})
    return pd.DataFrame(results)


# Streamlit App
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Search Stock", "Earnings per Date"])

    if selection == "Search Stock":
        st.title("Earnings Position Checker")
        stock_symbol = st.text_input("Enter Stock Symbol:", "AAPL").upper()  # Default AAPL

        if st.button("Submit"):
            with st.spinner(f"Analyzing {stock_symbol}..."):
                try:
                    recommendation = compute_recommendation(stock_symbol)

                    if isinstance(recommendation, str):
                        st.error(recommendation)  # Display error messages
                    else:
                        avg_volume_bool = recommendation['avg_volume']
                        iv30_rv30_bool = recommendation['iv30_rv30']
                        ts_slope_bool = recommendation['ts_slope_0_45']
                        expected_move = recommendation['expected_move']

                        if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
                            title = "Recommended"
                            title_color = "#006600"
                        elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (
                                iv30_rv30_bool and not avg_volume_bool)):
                            title = "Consider"
                            title_color = "#ff9900"
                        else:
                            title = "Avoid"
                            title_color = "#800000"

                        st.markdown(f"<h2 style='color: {title_color};'>{title}</h2>", unsafe_allow_html=True)
                        st.write(f"Average Volume: {'PASS' if avg_volume_bool else 'FAIL'}")
                        st.write(f"IV30/RV30: {'PASS' if iv30_rv30_bool else 'FAIL'}")
                        st.write(f"Term Structure Slope: {'PASS' if ts_slope_bool else 'FAIL'}")
                        st.write(f"Expected Move: {expected_move}")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    elif selection == "Earnings per Date":
        st.title("Earnings Analysis by Date")
        selected_date = st.date_input("Select Date", datetime.today())

        if st.button("Analyze Earnings"):
            with st.spinner(f"Fetching earnings data for {selected_date.strftime('%Y-%m-%d')}..."):
                earnings_data = fetch_earnings_for_date(selected_date)
                if not earnings_data.empty:
                    with st.spinner("Analyzing earnings data..."):
                        analysis_results = analyze_earnings_data(earnings_data)
                        st.write("Analysis Results:")
                        st.dataframe(analysis_results)
                else:
                    st.warning(
                        "No earnings data found for the selected date. This could be due to the website having no data for the given date, or due to the website format changing.")


if __name__ == "__main__":
    main()
