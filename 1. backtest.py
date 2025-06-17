# =============================================================================
# 최종 통합 백테스팅 스크립트 (advanced_backtester.py)
# 기능: 그리드 서치, 다수 티커 테스트, 시간봉/일봉 전환 지원
# 최종 수정일: 2025-06-06 (오류 최종 수정)
# =============================================================================

import pandas as pd
import numpy as np
import sqlite3
import itertools
import pandas_ta as ta
import os
from datetime import datetime


# =============================================================================
# --- 1. ✨ 통합 설정 (이 부분만 수정하여 사용하세요) ---
# =============================================================================

# --- 실행 모드 선택 ---
# 'GRID_SEARCH' : 단일 티커에 대해 여러 파라미터 조합을 테스트 (그리드 서치)
# 'MULTI_TICKER' : 여러 티커에 대해 지정된 챔피언 전략들을 테스트 (왕중왕전)
MODE = 'GRID_SEARCH'

# --- 시간 단위 선택 ---
# 'day' : 일봉 데이터로 테스트
# 'minute60' : 60분봉(시간봉) 데이터로 테스트
TARGET_INTERVAL = 'day'  # 'day' 또는 'minute60'

# --- 기본 설정 ---
INITIAL_CAPITAL = 10000000.0
FEE_RATE = 0.0005
MIN_ORDER_KRW = 5000.0

# --- 데이터베이스 경로 ---
OHLCV_DB_PATH = "upbit_ohlcv.db"
# MACRO_DB_PATH = "upbit_ohlcv_BTC.db"
# FNG_DB_PATH = "fng_index.db"

# --- 공통 테이블 이름 ---
# FNG_TABLE = "fear_and_greed"
# MACRO_TABLE = "macro_data"
# MARKET_INDEX_TABLE = "market_index_top12_ew"

# =============================================================================
# --- 2. 모드별 상세 설정 ---
# =============================================================================

# --- 2-1. 그리드 서치 모드 설정 (MODE = 'GRID_SEARCH' 일 때 사용) ---
GRID_SEARCH_CONFIG = {
    'target_ticker': 'KRW-BTC',
    'target_strategy_name': 'rsi_mean_reversion',
    'param_grid': {
        'partial_profit_target': [0.25],
        # 'partial_profit_ratio': [0.3, 0.5, 0.7]
        # 'rsi_period': [24, 48],           # RSI 계산 기간
        # 'oversold_level': [35, 40, 45],       # 과매도 기준선
        # 'overbought_level': [80, 85],       # 과매수 기준선 (청산용)
        # 'long_term_sma_period': [120, 168, 240, 336],
        # 'stop_loss_atr_multiplier': [2.0, 2.5],
        # 'trailing_stop_percent': [0.25, 0.3]
    },
    # 그리드 서치에 포함되지 않는 기본 파라미터
    'base_params': {
        'exit_sma_period': None,         # SMA 이탈 청산
        'long_term_sma_period': 120,     # 장기 추세 판단
        'stop_loss_atr_multiplier': 2.5, # ATR
        'trailing_stop_percent': 0.25,   # trailing_stop
        # rsi_mean_reversion
        'rsi_period': 24,             # RSI 계산기간
        # 'oversold_level': 45,         # 과매도 상태 판단, 30 이하에서 올라올 때 매수
        # 'overbought_level': 70,       # 과매수 상태 판단

        # 부분익절
        # 'partial_profit_target': 0.3,  # 부분익절 수익률
        'partial_profit_ratio': 0.3,   # 부분익절 비율

        # turtle_trading
        # 'entry_period': 20,            # turtle 진입 시기 판단
        # 'exit_period': None,           # turtle 청산 시기 판단

        # volatility_breakout
        # 'k': 1.5,                      # volatility_breakout 변동성

        # trend_following
        # 'breakout_window': 20,         # N일 신고가
        # 'volume_avg_window': 3,        # N일 거래량
        # 'volume_multiplier': 1.1,      # N일 거래량 대비 증가량
        # 'exit_sma_period': 5,

        # dual_momentum
        # 'abs_momentum_period': 120,   # 자산 자체의 추세 판단
        # 'rel_momentum_period': 120,   # 시장 대비 강도 판단

        # rsi_mean_reversion
        # 'rsi_period': 24,             # RSI 계산기간
        'oversold_level': 45,         # 과매도 상태 판단, 30 이하에서 올라올 때 매수
        'overbought_level': 85,       # 과매수 상태 판단

    }
}

# --- 2-2. 다수 티커 테스트 모드 설정 (MODE = 'MULTI_TICKER' 일 때 사용) ---
MULTI_TICKER_CONFIG = {
    'tickers_to_test': ["KRW-BTC"], # "KRW-ETH", "KRW-DOGE", "KRW-ADA", "KRW-AVAX",
                        #"KRW-LINK", "KRW-SOL", "KRW-SUI", "KRW-TRX", 'KRW-XLM', "KRW-XRP"],
    # "KRW-BTC", "KRW-ETH", "KRW-DOGE", "KRW-ADA", "KRW-AVAX",
    # "KRW-LINK", "KRW-SOL", "KRW-SUI", "KRW-TRX", 'KRW-XLM', "KRW-XRP" # 예시 티커
    'champions_to_run': [
        {'strategy_name': 'volatility_breakout', 'experiment_name_prefix': 'candi1',
         'k': 1.5, 'exit_sma_period': 5},

        {'strategy_name': 'turtle_trading', 'experiment_name_prefix': 'candi2',
         'entry_period': 20, 'exit_period': 10},
        #
        # {'strategy_name': 'trend_following', 'experiment_name_prefix': 'candi3',
        #  'breakout_window': 20, 'volume_avg_window': 20, 'volume_multiplier': 1.5},
        #
        # {'strategy_name': 'rsi_mean_reversion', 'experiment_name_prefix': 'candi4',
        #  'rsi_period': 24, 'oversold_level': 30, 'overbought_level': 70, 'exit_sma_period': 10},
        #
        # {'strategy_name': 'dual_momentum', 'experiment_name_prefix': 'candi5',
        #  'abs_momentum_period': 120, 'rel_momentum_period': 120},
        #
        # {'strategy_name': 'ma_crossover', 'experiment_name_prefix': 'candi6',
        #  'short_ma': 50, 'long_ma': 120}
    ]
}

# =============================================================================
# --- 3. 핵심 함수들 (수정 완료, 더 이상 수정할 필요 없음) ---
# =============================================================================

def load_and_prepare_data(ohlcv_db_path, ohlcv_table):
    try:
        # OHLCV 데이터 로드
        print(f"'{ohlcv_db_path}'에서 OHLCV 데이터 ('{ohlcv_table}') 로드 중...")
        con_ohlcv = sqlite3.connect(ohlcv_db_path)
        df_ohlcv = pd.read_sql_query(f'SELECT * FROM "{ohlcv_table}"', con_ohlcv, index_col='timestamp',
                                     parse_dates=['timestamp'])
        con_ohlcv.close()



        # 시간대 및 시간 정보 통일 (모든 DataFrame에 적용)
        dataframes_to_normalize = [df_ohlcv]
        for i, df_item in enumerate(dataframes_to_normalize):
            if df_item.empty:  # 비어있는 DataFrame은 건너뜀
                print(f"주의: {i + 1}번째 DataFrame이 비어있어 정규화를 건너뜁니다.")
                continue
            if df_item.index.tz is not None:
                df_item.index = df_item.index.tz_localize(None)
            df_item.index = df_item.index.normalize()
        print("✅ 모든 로드된 데이터의 시간대 및 시간 정보 통일 완료.")

        # --- 데이터 병합 ---

        df_merged = df_ohlcv


        print("✅ 데이터 병합 완료.")

        # --- 데이터 전처리 ---
        df_merged.ffill(inplace=True)
        # 병합 과정에서 모든 데이터가 필수적인지, 아니면 특정 데이터만 필수인지에 따라 dropna 기준 변경 가능
        # 여기서는 'close' 가격과, 만약 시장 지수를 전략에 사용한다면 'market_index_value'도 필수라고 가정
        required_columns_for_dropna = ['close']
        if 'market_index_value' in df_merged.columns:  # 시장 지수가 성공적으로 병합되었다면
            required_columns_for_dropna.append('market_index_value')
        df_merged.dropna(subset=required_columns_for_dropna, inplace=True)

        print(f"✅ 데이터 전처리 완료. (최종 {len(df_merged)}개 행)")

        if df_merged.empty:
            print("오류: 전처리 후 데이터가 남아있지 않습니다.")
            return pd.DataFrame()

        return df_merged

    except Exception as e:
        print(f"데이터 로드 또는 병합 중 오류 발생: {e}")
        return pd.DataFrame()

# --- 3. 기술적 지표 추가 함수 (수정) ---
def add_technical_indicators(df: pd.DataFrame, strategies: list):
    """실행할 전략 목록을 기반으로 필요한 모든 기술적 보조지표를 동적으로 계산합니다."""
    print("\n--- 기술적 지표 동적 계산 시작 ---")
    if df is None or df.empty: return df
    df_copy = df.copy()

    # 필요한 모든 기간을 수집 (더 이상 24를 곱하지 않음)
    sma_periods, high_low_periods, rsi_periods = set(), set(), set()
    for params in strategies:
        for key, value in params.items():
            if value and isinstance(value, (int, float)):
                if 'sma_period' in key: sma_periods.add(int(value))
                if 'entry_period' in key or 'exit_period' in key or 'breakout_window' in key: high_low_periods.add(
                    int(value))
                if 'rsi_period' in key: rsi_periods.add(int(value))

    for period in sorted(list(sma_periods)): df_copy.ta.sma(length=period, append=True)
    for period in sorted(list(high_low_periods)):
        df_copy[f'high_{period}d'] = df_copy['high'].rolling(window=period).max()
        df_copy[f'low_{period}d'] = df_copy['low'].rolling(window=period).min()
    for period in sorted(list(rsi_periods)): df_copy.ta.rsi(length=period, append=True)

    # 기타 고정 지표 (필요시 이 값들도 파라미터화 가능)
    df_copy.ta.rsi(length=14, append=True)
    df_copy.ta.bbands(length=20, std=2, append=True)
    df_copy.ta.atr(length=14, append=True, col_names=('ATRr_14',))
    df_copy['range'] = df_copy['high'].shift(1) - df_copy['low'].shift(1)
    df_copy.ta.obv(append=True)

    # (참고) 거시경제지표 이평선 추가 (필요 시)
    if 'nasdaq_close' in df_copy.columns:
        df_copy['nasdaq_sma_200'] = df_copy['nasdaq_close'].rolling(window=200).mean()

    return df_copy


def strategy_trend_following(df, params):
    buy_condition = (df['high'] > df[f"high_{params.get('breakout_window')}d"].shift(1)) & \
                    (df['volume'] > df['volume'].rolling(window=params.get('volume_avg_window')).mean().shift(
                        1) * params.get('volume_multiplier'))
    if params.get('long_term_sma_period'): buy_condition &= (
                df['close'] > df[f"SMA_{params.get('long_term_sma_period')}"])
    df['signal'] = np.where(buy_condition, 1, 0)
    return df


def strategy_volatility_breakout(df, params):
    base_buy_condition = df['high'] > (df['open'] + df['range'] * params.get('k', 0.5))
    if params.get('long_term_sma_period'): base_buy_condition &= (
                df['close'] > df[f"SMA_{params.get('long_term_sma_period')}"])
    df['signal'] = np.where(base_buy_condition, 1, 0)
    return df


def strategy_turtle_trading(df, params):
    base_buy_condition = df['high'] > df[f"high_{params.get('entry_period')}d"].shift(1)
    if params.get('long_term_sma_period'): base_buy_condition &= (
                df['close'] > df[f"SMA_{params.get('long_term_sma_period')}"])
    df['signal'] = np.where(base_buy_condition, 1, 0)
    return df


def strategy_rsi_mean_reversion(df, params):
    rsi_col = f"RSI_{params.get('rsi_period')}"
    base_buy_condition = (df[rsi_col] > params.get('oversold_level')) & (
                df[rsi_col].shift(1) <= params.get('oversold_level'))
    if params.get('long_term_sma_period'): base_buy_condition &= (
                df['close'] > df[f"SMA_{params.get('long_term_sma_period')}"])
    sell_condition = (df[rsi_col] < params.get('overbought_level')) & (
                df[rsi_col].shift(1) >= params.get('overbought_level'))
    df['signal'] = np.where(base_buy_condition, 1, np.where(sell_condition, -1, 0))
    return df

def strategy_dual_momentum(df, params):
    """
    전략 3: 듀얼 모멘텀 (절대 모멘텀 + 상대 모멘텀)
    """
    abs_momentum_period = params.get('abs_momentum_period', 120)  # 약 6개월
    rel_momentum_period = params.get('rel_momentum_period', 120)

    # 절대 모멘텀: 자산의 N일 전 가격보다 현재 가격이 높은가?
    df['abs_momentum'] = df['close'] / df['close'].shift(abs_momentum_period) - 1
    is_abs_momentum_positive = df['abs_momentum'] > 0

    # 상대 모멘텀: 자산의 수익률이 시장 지수의 수익률보다 높은가?
    df['asset_return'] = df['close'] / df['close'].shift(rel_momentum_period) - 1
    df['market_return'] = df['market_index_value'] / df['market_index_value'].shift(rel_momentum_period) - 1
    is_rel_momentum_stronger = df['asset_return'] > df['market_return']

    buy_condition = is_abs_momentum_positive & is_rel_momentum_stronger
    df['signal'] = np.where(buy_condition, 1, 0)
    return df

# --- 4. 전략 실행기 (신규) ---
def generate_signals(df, params):
    strategy_name = params.get('strategy_name')
    strategy_functions = {
        'trend_following': strategy_trend_following, 'volatility_breakout': strategy_volatility_breakout,
        'turtle_trading': strategy_turtle_trading, 'rsi_mean_reversion': strategy_rsi_mean_reversion,
    }
    if strategy_name in strategy_functions:
        return strategy_functions[strategy_name](df, params)
    else:
        raise ValueError(f"'{strategy_name}'은(는) 알 수 없는 전략입니다.")


# --- 5. 백테스팅 실행 함수 ---
def run_backtest(df_full_data, params):
    df_signals = generate_signals(df_full_data.copy(), params)
    krw_balance, asset_balance, asset_avg_buy_price = INITIAL_CAPITAL, 0.0, 0.0
    trade_log, portfolio_history = [], []
    highest_price_since_buy, partial_profit_taken = 0, False

    for timestamp, row in df_signals.iterrows():
        current_price, atr = row['close'], row.get('ATRr_14', 0)
        if pd.isna(current_price) or current_price <= 0:
            portfolio_history.append({'timestamp': timestamp, 'portfolio_value': portfolio_history[-1][
                'portfolio_value'] if portfolio_history else INITIAL_CAPITAL})
            continue

        should_sell = False
        if asset_balance > 0:
            highest_price_since_buy = max(highest_price_since_buy, current_price)

            # 부분 익절 로직
            profit_target = params.get('partial_profit_target')
            if profit_target and not partial_profit_taken and (
                    current_price / asset_avg_buy_price - 1) >= profit_target:
                asset_to_sell = asset_balance * params.get('partial_profit_ratio', 0.5)
                if asset_to_sell * current_price >= MIN_ORDER_KRW:
                    krw_balance += (asset_to_sell * current_price * (1 - FEE_RATE));
                    asset_balance -= asset_to_sell;
                    partial_profit_taken = True
                    trade_log.append({'timestamp': timestamp, 'type': 'partial_sell', 'price': current_price,
                                      'amount': asset_to_sell})
                    portfolio_history.append(
                        {'timestamp': timestamp, 'portfolio_value': krw_balance + (asset_balance * current_price)})
                    continue

            # 1. ATR 손절매 (값이 있을 때만 실행)
            stop_loss = params.get('stop_loss_atr_multiplier')
            if not should_sell and stop_loss and atr > 0 and current_price < (
                    asset_avg_buy_price - (stop_loss * atr)): should_sell = True

            # 2. 트레일링 스탑 (값이 있을 때만 실행)
            trailing_stop = params.get('trailing_stop_percent')
            if not should_sell and trailing_stop and current_price < highest_price_since_buy * (
                    1 - trailing_stop): should_sell = True

            # 3. SMA 이탈 청산 (값이 있을 때만 실행)
            exit_sma_period = params.get('exit_sma_period')
            if not should_sell and exit_sma_period and exit_sma_period > 0:
                if current_price < row.get(f"SMA_{exit_sma_period}", float('inf')):
                     should_sell = True

            # 4. 터틀 전략 고유 청산 (값이 있을 때만 실행)
            if not should_sell and params.get('strategy_name') == 'turtle_trading':
                exit_period = params.get('exit_period')
                if exit_period and current_price < row.get(f'low_{exit_period}d', float('inf')): should_sell = True

            # 전략이 직접 매도 신호를 보냈을 경우
            if not should_sell and row.get('signal') == -1: should_sell = True

        # --- 거래 실행 ---
        if should_sell and asset_balance > 0:
            krw_balance += (asset_balance * current_price * (1 - FEE_RATE))
            trade_log.append({'timestamp': timestamp, 'type': 'sell', 'price': current_price, 'amount': asset_balance})
            asset_balance = 0.0
        elif row.get('signal') == 1 and asset_balance == 0:
            buy_amount_krw = krw_balance * 0.95
            if buy_amount_krw > MIN_ORDER_KRW:
                asset_acquired = (buy_amount_krw * (1 - FEE_RATE)) / current_price
                krw_balance -= buy_amount_krw;
                asset_balance += asset_acquired;
                asset_avg_buy_price = current_price
                highest_price_since_buy, partial_profit_taken = current_price, False
                trade_log.append(
                    {'timestamp': timestamp, 'type': 'buy', 'price': current_price, 'amount': asset_acquired})

        # 포트폴리오 가치 기록
        portfolio_history.append(
            {'timestamp': timestamp, 'portfolio_value': krw_balance + (asset_balance * current_price)})
    return pd.DataFrame(trade_log), pd.DataFrame(portfolio_history)


def get_round_trip_trades(trade_log_df):
    """
    부분 익절을 포함한 거래 기록을 바탕으로 완성된 거래(Round Trip)를 재구성합니다.
    (기존 backtesting.py의 정교한 로직으로 복원)
    """
    if trade_log_df.empty: return pd.DataFrame()
    round_trips = []
    active_buy_info = None
    for _, trade in trade_log_df.iterrows():
        if trade['type'] == 'buy':
            active_buy_info = {'entry_date': trade['timestamp'], 'entry_price': trade['price'],
                               'amount_remaining': trade['amount']}
        elif (trade['type'] == 'partial_sell' or trade['type'] == 'sell') and active_buy_info:
            amount_sold = trade['amount'] if trade['type'] == 'partial_sell' else active_buy_info['amount_remaining']
            pnl = (trade['price'] - active_buy_info['entry_price']) * amount_sold
            round_trips.append({'pnl': pnl})
            if trade['type'] == 'partial_sell':
                active_buy_info['amount_remaining'] -= amount_sold
                if active_buy_info['amount_remaining'] < 1e-9: active_buy_info = None
            else:
                active_buy_info = None
    return pd.DataFrame(round_trips)


def analyze_performance_detailed(portfolio_history_df, trade_log_df, initial_capital, params, interval,
                                 risk_free_rate_daily=0.0):
    if portfolio_history_df.empty: return {}

    #기본 성과
    experiment_name_to_print = params.get('experiment_name', params.get('strategy_name')) # 없으면 strategy_name 사용
    final_value = portfolio_history_df['portfolio_value'].iloc[-1]
    total_roi_pct = (final_value / initial_capital - 1) * 100

    #MDD
    portfolio_history_df['rolling_max'] = portfolio_history_df['portfolio_value'].cummax()
    mdd_pct = (portfolio_history_df['portfolio_value'] / portfolio_history_df['rolling_max'] - 1).min() * 100

    portfolio_history_df['returns'] = portfolio_history_df['portfolio_value'].pct_change().fillna(0)
    periods_per_year = 365 if interval == 'day' else 365 * 24

    #샤프지수
    sharpe_ratio = portfolio_history_df['returns'].mean() / portfolio_history_df['returns'].std() * np.sqrt(
        periods_per_year) if portfolio_history_df['returns'].std() > 0 else 0
    annual_return = portfolio_history_df['returns'].mean() * periods_per_year

    #캘머 지수
    calmar_ratio = annual_return / (abs(mdd_pct) / 100) if mdd_pct != 0 else 0

    rt_trades_df = get_round_trip_trades(trade_log_df)
    total_trades, win_rate_pct, profit_factor = 0, 0.0, 0.0
    if not rt_trades_df.empty:
        total_trades = len(rt_trades_df)
        wins = rt_trades_df[rt_trades_df['pnl'] > 0]
        losses = rt_trades_df[rt_trades_df['pnl'] <= 0]
        win_rate_pct = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # (복원) 거래 이벤트 상세 카운트 출력
    # num_buy = len(trade_log_df[trade_log_df['type'] == 'buy'])
    # num_partial_sell = len(trade_log_df[trade_log_df['type'] == 'partial_sell'])
    # num_full_sell = len(trade_log_df[trade_log_df['type'] == 'sell'])

    # 결과 출력
    print(f"전략명: {params.get('strategy_name')}")
    print(f"실험명(설명): {experiment_name_to_print}") # 콘솔 출력에 추가
    print(f"총 수익률 (ROI): {total_roi_pct:.2f}% | 최대 낙폭 (MDD): {mdd_pct:.2f}%")
    print(f"샤프 지수: {sharpe_ratio:.2f} | 캘머 지수: {calmar_ratio:.2f}")
    print(f"총 거래 횟수: {total_trades} | 승률: {win_rate_pct:.2f}% | 수익 팩터: {profit_factor:.2f}")
    # print(f"  매수 이벤트: {num_buy} 회 | 부분 매도: {num_partial_sell} 회 | 전량 매도: {num_full_sell} 회")

    return {
        '실험명': params.get('experiment_name', ''), '전략명': params.get('strategy_name'),
        '파라미터': str(
            {k: v for k, v in params.items() if k not in ['strategy_name', 'experiment_name', 'ticker_tested']}),
        'ROI (%)': round(total_roi_pct, 2), 'MDD (%)': round(mdd_pct, 2),
        'Sharpe': round(sharpe_ratio, 2), 'Calmar': round(calmar_ratio, 2),
        'Profit Factor': round(profit_factor, 2), 'Win Rate (%)': round(win_rate_pct, 2),
        'Total Trades': total_trades,
    }


def log_results_to_csv(result_data, log_file="advanced_backtest_log.csv"):
    """백테스팅 결과와 파라미터를 CSV 파일에 기록합니다."""
    # 파일이 존재하지 않으면 헤더와 함께 새로 만들고, 존재하면 데이터만 추가
    is_file_exist = os.path.exists(log_file)

    df_result = pd.DataFrame([result_data])
    df_result.to_csv(log_file, index=False, mode='a', header=not os.path.exists(log_file), encoding='utf-8-sig')


# =============================================================================
# --- 4. 🚀 메인 실행 블록 ---
# =============================================================================
if __name__ == "__main__":

    strategies_to_run = []

    if MODE == 'GRID_SEARCH':
        config = GRID_SEARCH_CONFIG
        keys, values = config['param_grid'].keys(), config['param_grid'].values()
        for i, combo_values in enumerate(itertools.product(*values)):
            params = {**config['base_params'], **dict(zip(keys, combo_values))}
            if config['target_strategy_name'] == 'turtle_trading' and params.get('entry_period') <= params.get(
                'exit_period'): continue
            exp_name_parts = [f"{key[:4]}{val}" for key, val in dict(zip(keys, combo_values)).items()]
            params.update({
                'strategy_name': config['target_strategy_name'], 'ticker_tested': config['target_ticker'],
                'experiment_name': f"GS_{config['target_strategy_name'][:5]}_{'_'.join(exp_name_parts)}_{i}"
            })
            strategies_to_run.append(params)

    elif MODE == 'MULTI_TICKER':
        config = MULTI_TICKER_CONFIG
        for ticker in config['tickers_to_test']:
            for champ_config in config['champions_to_run']:
                params = champ_config.copy()
                params.update({
                    'experiment_name': f"{ticker}_{params.pop('experiment_name_prefix')}",
                    'ticker_tested': ticker
                })
                strategies_to_run.append(params)

    print(f"\n총 {len(strategies_to_run)}개의 실험을 진행합니다.")

    all_results = []
    data_cache = {}

    for strategy_params in strategies_to_run:
        ticker = strategy_params['ticker_tested']

        if ticker not in data_cache:
            print(f"\n\n===== {ticker} ({TARGET_INTERVAL}) 데이터 로딩 및 지표 계산 =====")
            ohlcv_table = f"{ticker.replace('-', '_')}_{TARGET_INTERVAL}"
            df_raw = load_and_prepare_data(
                ohlcv_db_path=OHLCV_DB_PATH,
                ohlcv_table=ohlcv_table,
            )
            if df_raw.empty: continue

            strategies_for_this_ticker = [s for s in strategies_to_run if s.get('ticker_tested') == ticker]
            data_cache[ticker] = add_technical_indicators(df_raw, strategies_for_this_ticker)

        df_ready = data_cache[ticker]

        trade_log_df, portfolio_history_df = run_backtest(df_ready.copy(), strategy_params)

        if trade_log_df is not None and portfolio_history_df is not None:
            summary = analyze_performance_detailed(portfolio_history_df, trade_log_df, INITIAL_CAPITAL, strategy_params,
                                                   TARGET_INTERVAL)
            if summary:
                summary['티커'] = ticker
                all_results.append(summary)
                log_results_to_csv(summary)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by='Calmar', ascending=False)
        print("\n\n" + "=" * 90 + "\n" + "<<< 최종 백테스트 결과 요약 (정렬 기준: Calmar 내림차순) >>>".center(85) + "\n" + "=" * 90)
        cols_to_display = ['티커', '실험명', 'ROI (%)', 'MDD (%)', 'Calmar', 'Sharpe', 'Profit Factor', 'Win Rate (%)',
                           'Total Trades']
        print(results_df[cols_to_display])
        print("=" * 90)