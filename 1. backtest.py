# 1. backtest.py

# 필요한 라이브러리들을 가져옵니다.
import pandas as pd  # 데이터 분석 및 조작
import numpy as np  # 숫자 계산, 배열 처리
import sqlite3  # SQLite 데이터베이스 사용
import itertools  # 파라미터 조합을 만들기 위한 라이브러리
import pandas_ta as ta  # 기술적 분석 지표를 쉽게 계산하기 위한 라이브러리
import os  # 파일 존재 여부 확인 등 운영체제 기능 사용
from datetime import datetime  # 날짜/시간 관련 기능

# =============================================================================
# --- 1. ✨ 통합 설정 (사용자가 직접 수정하는 부분) ---
# =============================================================================

# --- 실행 모드 선택 ---
# 'GRID_SEARCH' : 하나의 티커에 여러 파라미터 조합을 테스트 (최적 파라미터 찾기)
# 'MULTI_TICKER' : 여러 티커에 정해진 우수 전략들을 테스트 (전략 비교)
MODE = 'GRID_SEARCH'

# --- 시간 단위 선택 ---
TARGET_INTERVAL = 'day'  # 'day'(일봉) 또는 'minute60'(시간봉)

# --- 기본 설정 ---
INITIAL_CAPITAL = 10000000.0  # 초기 자본금 (1천만원)
FEE_RATE = 0.0005  # 거래 수수료 (0.05%)
MIN_ORDER_KRW = 5000.0  # 최소 주문 금액 (5천원)

# --- 데이터베이스 경로 ---
OHLCV_DB_PATH = "upbit_ohlcv.db"  # collect_ohlcv.py가 생성한 DB 파일

# =============================================================================
# --- 2. 모드별 상세 설정 ---
# =============================================================================

# --- 2-1. 그리드 서치 모드 설정 (MODE = 'GRID_SEARCH' 일 때 사용) ---
GRID_SEARCH_CONFIG = {
    'target_ticker': 'KRW-BTC',  # 테스트할 단일 티커
    'target_strategy_name': 'rsi_mean_reversion',  # 테스트할 단일 전략
    # 'param_grid'에 여러 값들을 리스트로 넣어주면, 이들의 모든 조합을 테스트합니다.
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
    # 'champions_to_run': 테스트할 "챔피언" 전략들의 목록
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
# --- 3. 핵심 함수들 (이 아래부터는 수정할 필요 없음) ---
# =============================================================================

def load_and_prepare_data(ohlcv_db_path, ohlcv_table):
    """지정된 DB에서 데이터를 불러오고 백테스팅에 맞게 전처리하는 함수"""
    try:
        print(f"'{ohlcv_db_path}'에서 OHLCV 데이터 ('{ohlcv_table}') 로드 중...")
        # SQLite DB에 연결하고, SQL 쿼리로 테이블 전체를 읽어 Pandas DataFrame으로 변환
        con_ohlcv = sqlite3.connect(ohlcv_db_path)
        # index_col='timestamp' : 'timestamp' 컬럼을 DataFrame의 인덱스로 사용
        # parse_dates=['timestamp'] : 'timestamp' 컬럼을 날짜/시간 타입으로 자동 변환
        df_ohlcv = pd.read_sql_query(f'SELECT * FROM "{ohlcv_table}"', con_ohlcv, index_col='timestamp',
                                     parse_dates=['timestamp'])
        con_ohlcv.close()  # DB 연결 종료

        # 시간대 정보(timezone)를 제거하여 통일시킵니다. (오류 방지)
        if df_ohlcv.index.tz is not None:
            df_ohlcv.index = df_ohlcv.index.tz_localize(None)
        # 시간 정보를 자정(00:00:00)으로 통일합니다. (일봉 데이터 처리 시 중요)
        df_ohlcv.index = df_ohlcv.index.normalize()
        print("✅ 데이터의 시간대 및 시간 정보 통일 완료.")

        df_merged = df_ohlcv  # 이 예제에서는 다른 데이터와 병합하지 않으므로 그대로 사용

        # 데이터 전처리: 비어있는 값(NaN)을 바로 이전 값으로 채웁니다 (forward fill)
        df_merged.ffill(inplace=True)
        # 'close' 가격 데이터가 없는 행은 백테스팅에 의미가 없으므로 제거합니다.
        df_merged.dropna(subset=['close'], inplace=True)

        print(f"✅ 데이터 전처리 완료. (최종 {len(df_merged)}개 행)")

        if df_merged.empty:
            print("오류: 전처리 후 데이터가 남아있지 않습니다.")
            return pd.DataFrame()

        return df_merged

    except Exception as e:
        print(f"데이터 로드 또는 병합 중 오류 발생: {e}")
        return pd.DataFrame()


def add_technical_indicators(df: pd.DataFrame, strategies: list):
    """실행할 전략 목록을 기반으로 필요한 모든 기술적 보조지표를 동적으로 계산합니다."""
    print("\n--- 기술적 지표 동적 계산 시작 ---")
    if df is None or df.empty: return df
    df_copy = df.copy()  # 원본 데이터 보존을 위해 복사본 사용

    # 앞으로 실행할 모든 전략들에서 필요한 지표의 '기간(period)' 값들을 모두 수집
    sma_periods, high_low_periods, rsi_periods = set(), set(), set()
    for params in strategies:
        for key, value in params.items():
            if value and isinstance(value, (int, float)):
                if 'sma_period' in key: sma_periods.add(int(value))
                if 'entry_period' in key or 'exit_period' in key or 'breakout_window' in key: high_low_periods.add(
                    int(value))
                if 'rsi_period' in key: rsi_periods.add(int(value))

    # 수집된 기간 값들을 이용해 필요한 지표들을 한 번에 계산 (효율적)
    # df_copy.ta.sma(...)는 pandas_ta 라이브러리의 기능으로, 자동으로 SMA를 계산하고 DataFrame에 추가해줍니다.
    for period in sorted(list(sma_periods)): df_copy.ta.sma(length=period, append=True)
    for period in sorted(list(high_low_periods)):
        df_copy[f'high_{period}d'] = df_copy['high'].rolling(window=period).max()
        df_copy[f'low_{period}d'] = df_copy['low'].rolling(window=period).min()
    for period in sorted(list(rsi_periods)): df_copy.ta.rsi(length=period, append=True)

    # 모든 전략에서 공통적으로 사용할 수 있는 기본 지표들도 계산
    df_copy.ta.atr(length=14, append=True, col_names=('ATRr_14',))  # ATR (변동성 지표)
    df_copy['range'] = df_copy['high'].shift(1) - df_copy['low'].shift(1)  # 전일 변동폭

    return df_copy


# --- 각 전략의 매수/매도 신호를 생성하는 함수들 ---
# 모든 전략 함수는 DataFrame과 파라미터 딕셔너리를 입력받아,
# 'signal' 이라는 컬럼을 추가하여 반환합니다. (1: 매수, -1: 매도, 0: 관망)

def strategy_trend_following(df, params):
    # N일 신고가 돌파 & 거래량 급증 시 매수
    buy_condition = (df['high'] > df[f"high_{params.get('breakout_window')}d"].shift(1)) & \
                    (df['volume'] > df['volume'].rolling(window=params.get('volume_avg_window')).mean().shift(
                        1) * params.get('volume_multiplier'))
    # 장기 이동평균선 위에 있을 때만 매수 (추세 필터)
    if params.get('long_term_sma_period'):
        buy_condition &= (df['close'] > df[f"SMA_{params.get('long_term_sma_period')}"])
    df['signal'] = np.where(buy_condition, 1, 0)
    return df


def strategy_volatility_breakout(df, params):
    # (시가 + 전일 변동폭 * k) 가격을 현재 고가가 돌파하면 매수
    buy_condition = df['high'] > (df['open'] + df['range'] * params.get('k', 0.5))
    if params.get('long_term_sma_period'):
        buy_condition &= (df['close'] > df[f"SMA_{params.get('long_term_sma_period')}"])
    df['signal'] = np.where(buy_condition, 1, 0)
    return df


def strategy_turtle_trading(df, params):
    # N일 신고가를 돌파하면 매수
    buy_condition = df['high'] > df[f"high_{params.get('entry_period')}d"].shift(1)
    if params.get('long_term_sma_period'):
        buy_condition &= (df['close'] > df[f"SMA_{params.get('long_term_sma_period')}"])
    df['signal'] = np.where(buy_condition, 1, 0)
    return df


def strategy_rsi_mean_reversion(df, params):
    # RSI가 과매도선을 상향 돌파하면 매수, 과매수선을 하향 돌파하면 매도
    rsi_col = f"RSI_{params.get('rsi_period')}"
    buy_condition = (df[rsi_col] > params.get('oversold_level')) & (
                df[rsi_col].shift(1) <= params.get('oversold_level'))
    if params.get('long_term_sma_period'):
        buy_condition &= (df['close'] > df[f"SMA_{params.get('long_term_sma_period')}"])
    sell_condition = (df[rsi_col] < params.get('overbought_level')) & (
                df[rsi_col].shift(1) >= params.get('overbought_level'))
    df['signal'] = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))
    return df


# --- 전략 실행기 ---
def generate_signals(df, params):
    """파라미터에 명시된 전략 이름에 맞는 함수를 호출하여 신호를 생성하는 함수"""
    strategy_name = params.get('strategy_name')
    strategy_functions = {  # 각 전략 이름과 실제 함수를 연결하는 딕셔너리
        'trend_following': strategy_trend_following,
        'volatility_breakout': strategy_volatility_breakout,
        'turtle_trading': strategy_turtle_trading,
        'rsi_mean_reversion': strategy_rsi_mean_reversion,
    }
    if strategy_name in strategy_functions:
        return strategy_functions[strategy_name](df, params)
    else:
        raise ValueError(f"'{strategy_name}'은(는) 알 수 없는 전략입니다.")


# --- 5. 백테스팅 실행 함수 ---
def run_backtest(df_full_data, params):
    """본격적인 거래 시뮬레이션을 수행하는 핵심 함수"""
    # 1. 주어진 데이터와 파라미터로 매수/매도 신호를 먼저 계산합니다.
    df_signals = generate_signals(df_full_data.copy(), params)

    # 2. 시뮬레이션을 위한 초기 상태 변수들을 설정합니다.
    krw_balance, asset_balance, asset_avg_buy_price = INITIAL_CAPITAL, 0.0, 0.0
    trade_log, portfolio_history = [], []  # 거래 내역과 포트폴리오 가치 변화를 기록할 리스트
    highest_price_since_buy, partial_profit_taken = 0, False  # 트레일링 스탑, 부분 익절용 변수

    # 3. 데이터프레임을 한 줄씩(하루씩) 순회하며 시뮬레이션을 진행합니다.
    for timestamp, row in df_signals.iterrows():
        current_price = row['close']
        if pd.isna(current_price) or current_price <= 0:  # 가격 데이터가 없으면 건너뛰기
            continue

        should_sell = False  # 매도 여부를 결정하는 플래그

        # 4. 현재 자산을 보유하고 있는 경우, 매도 조건을 확인합니다.
        if asset_balance > 0:
            highest_price_since_buy = max(highest_price_since_buy, current_price)

            # [매도 조건 1] 부분 익절: 목표 수익률 달성 시
            profit_target = params.get('partial_profit_target')
            if profit_target and not partial_profit_taken and (
                    current_price / asset_avg_buy_price - 1) >= profit_target:
                asset_to_sell = asset_balance * params.get('partial_profit_ratio', 0.5)
                if asset_to_sell * current_price >= MIN_ORDER_KRW:
                    krw_balance += (asset_to_sell * current_price * (1 - FEE_RATE))
                    asset_balance -= asset_to_sell
                    partial_profit_taken = True
                    trade_log.append({'timestamp': timestamp, 'type': 'partial_sell', 'price': current_price,
                                      'amount': asset_to_sell})
                    # 부분 익절 후에는 다른 매도 로직을 타지 않고 다음 날로 넘어갑니다.
                    portfolio_history.append(
                        {'timestamp': timestamp, 'portfolio_value': krw_balance + (asset_balance * current_price)})
                    continue

            # [매도 조건 2] ATR 손절매
            stop_loss = params.get('stop_loss_atr_multiplier')
            if not should_sell and stop_loss and row.get('ATRr_14', 0) > 0:
                if current_price < (asset_avg_buy_price - (stop_loss * row.get('ATRr_14'))):
                    should_sell = True

            # [매도 조건 3] 트레일링 스탑
            trailing_stop = params.get('trailing_stop_percent')
            if not should_sell and trailing_stop:
                if current_price < highest_price_since_buy * (1 - trailing_stop):
                    should_sell = True

            # [매도 조건 4] 이동평균선 이탈 청산
            exit_sma_period = params.get('exit_sma_period')
            if not should_sell and exit_sma_period and exit_sma_period > 0:
                if current_price < row.get(f"SMA_{exit_sma_period}", float('inf')):
                    should_sell = True

            # [매도 조건 5] 터틀 트레이딩 전용 청산 (N일 최저가 하향 이탈)
            if not should_sell and params.get('strategy_name') == 'turtle_trading':
                exit_period = params.get('exit_period')
                if exit_period and current_price < row.get(f'low_{exit_period}d', float('inf')):
                    should_sell = True

            # [매도 조건 6] 전략이 직접 매도 신호(-1)를 보냈을 경우
            if not should_sell and row.get('signal') == -1:
                should_sell = True

        # --- 5. 최종 결정된 거래를 실행합니다 ---
        if should_sell and asset_balance > 0:  # 매도 결정!
            krw_balance += (asset_balance * current_price * (1 - FEE_RATE))
            trade_log.append({'timestamp': timestamp, 'type': 'sell', 'price': current_price, 'amount': asset_balance})
            asset_balance = 0.0
        elif row.get('signal') == 1 and asset_balance == 0:  # 매수 결정!
            buy_amount_krw = krw_balance * 0.95  # 현금의 95%를 매수에 사용
            if buy_amount_krw > MIN_ORDER_KRW:
                asset_acquired = (buy_amount_krw * (1 - FEE_RATE)) / current_price
                krw_balance -= buy_amount_krw
                asset_balance += asset_acquired
                asset_avg_buy_price = current_price
                # 매수 후, 트레일링 스탑과 부분 익절을 위한 변수 초기화
                highest_price_since_buy, partial_profit_taken = current_price, False
                trade_log.append(
                    {'timestamp': timestamp, 'type': 'buy', 'price': current_price, 'amount': asset_acquired})

        # 6. 매일의 포트폴리오 가치를 계산하여 기록합니다.
        portfolio_history.append(
            {'timestamp': timestamp, 'portfolio_value': krw_balance + (asset_balance * current_price)})

    # 모든 시뮬레이션이 끝난 후, 거래 기록과 포트폴리오 변화 기록을 반환합니다.
    return pd.DataFrame(trade_log), pd.DataFrame(portfolio_history)


def get_round_trip_trades(trade_log_df):
    """거래 기록을 바탕으로 '매수 -> 매도'로 이어지는 완성된 거래(Round Trip)를 재구성하여 손익(PNL)을 계산"""
    if trade_log_df.empty: return pd.DataFrame()
    round_trips = []
    active_buy_info = None  # 현재 진행중인 매수 정보
    for _, trade in trade_log_df.iterrows():
        if trade['type'] == 'buy':  # 매수 거래를 만나면 정보 저장
            active_buy_info = {'entry_date': trade['timestamp'], 'entry_price': trade['price'],
                               'amount_remaining': trade['amount']}
        elif (trade['type'] == 'partial_sell' or trade['type'] == 'sell') and active_buy_info:
            # 매도 거래를 만나면 손익 계산
            amount_sold = trade['amount'] if trade['type'] == 'partial_sell' else active_buy_info['amount_remaining']
            pnl = (trade['price'] - active_buy_info['entry_price']) * amount_sold
            round_trips.append({'pnl': pnl})
            if trade['type'] == 'partial_sell':  # 부분 매도면 남은 수량 업데이트
                active_buy_info['amount_remaining'] -= amount_sold
            else:  # 전량 매도면 매수 정보 초기화
                active_buy_info = None
    return pd.DataFrame(round_trips)


def analyze_performance_detailed(portfolio_history_df, trade_log_df, initial_capital, params, interval):
    """백테스팅 결과를 바탕으로 다양한 성과 지표를 계산하고 출력하는 함수"""
    if portfolio_history_df.empty: return {}

    # 총수익률(ROI) 계산
    final_value = portfolio_history_df['portfolio_value'].iloc[-1]
    total_roi_pct = (final_value / initial_capital - 1) * 100

    # 최대 낙폭(MDD) 계산
    portfolio_history_df['rolling_max'] = portfolio_history_df['portfolio_value'].cummax()
    mdd_pct = (portfolio_history_df['portfolio_value'] / portfolio_history_df['rolling_max'] - 1).min() * 100

    # 일일(또는 시간당) 수익률 계산
    portfolio_history_df['returns'] = portfolio_history_df['portfolio_value'].pct_change().fillna(0)
    periods_per_year = 365 if interval == 'day' else 365 * 24

    # 샤프 지수(Sharpe Ratio): (위험 대비 수익성), 높을수록 좋음
    sharpe_ratio = portfolio_history_df['returns'].mean() / portfolio_history_df['returns'].std() * np.sqrt(
        periods_per_year) if portfolio_history_df['returns'].std() > 0 else 0
    annual_return = portfolio_history_df['returns'].mean() * periods_per_year

    # 캘머 지수(Calmar Ratio): (최대 낙폭 대비 수익성), 높을수록 좋음
    calmar_ratio = annual_return / (abs(mdd_pct) / 100) if mdd_pct != 0 else 0

    # 거래 기반 지표 계산 (승률, 수익팩터 등)
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

    # 결과 출력
    print(f"\n--- 결과 분석: {params.get('experiment_name')} ---")
    print(f"총 수익률 (ROI): {total_roi_pct:.2f}% | 최대 낙폭 (MDD): {mdd_pct:.2f}%")
    print(f"샤프 지수: {sharpe_ratio:.2f} | 캘머 지수: {calmar_ratio:.2f}")
    print(f"총 거래 횟수: {total_trades} | 승률: {win_rate_pct:.2f}% | 수익 팩터: {profit_factor:.2f}")

    # 결과를 딕셔너리 형태로 정리하여 반환 (나중에 CSV로 저장하기 위함)
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
    """백테스팅 결과를 CSV 파일에 기록하는 함수"""
    df_result = pd.DataFrame([result_data])
    # 파일이 없으면 헤더와 함께 새로 쓰고, 파일이 있으면 헤더 없이 내용만 추가(append)
    df_result.to_csv(log_file, index=False, mode='a', header=not os.path.exists(log_file), encoding='utf-8-sig')


# =============================================================================
# --- 4. 🚀 메인 실행 블록 ---
# =============================================================================
if __name__ == "__main__":

    strategies_to_run = []  # 실행할 모든 실험(전략+파라미터 조합)을 저장할 리스트

    # 설정된 MODE에 따라 실행할 실험 목록을 생성
    if MODE == 'GRID_SEARCH':
        config = GRID_SEARCH_CONFIG
        keys, values = config['param_grid'].keys(), config['param_grid'].values()
        # itertools.product를 사용하여 파라미터 그리드의 모든 조합을 생성
        for i, combo_values in enumerate(itertools.product(*values)):
            # 기본 파라미터와 그리드 서치용 파라미터를 합쳐 하나의 실험 세트를 만듦
            params = {**config['base_params'], **dict(zip(keys, combo_values))}
            exp_name = f"GS_{config['target_strategy_name'][:5]}_{i}"  # 실험 이름 생성
            params.update({
                'strategy_name': config['target_strategy_name'],
                'ticker_tested': config['target_ticker'],
                'experiment_name': exp_name
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

    all_results = []  # 모든 실험 결과를 저장할 리스트
    data_cache = {}  # 로드한 데이터를 재사용하기 위한 캐시 (메모리 저장소)

    # 생성된 실험 목록을 하나씩 실행
    for strategy_params in strategies_to_run:
        ticker = strategy_params['ticker_tested']

        # 데이터 캐싱: 동일한 티커의 데이터를 여러 번 로드하지 않도록 처리
        if ticker not in data_cache:
            print(f"\n\n===== {ticker} ({TARGET_INTERVAL}) 데이터 로딩 및 지표 계산 =====")
            ohlcv_table = f"{ticker.replace('-', '_')}_{TARGET_INTERVAL}"
            # 데이터 로드
            df_raw = load_and_prepare_data(ohlcv_db_path=OHLCV_DB_PATH, ohlcv_table=ohlcv_table)
            if df_raw.empty: continue  # 데이터가 없으면 다음 실험으로

            # 이 티커에 대해 실행될 모든 전략을 찾아서 필요한 지표를 한 번에 계산
            strategies_for_this_ticker = [s for s in strategies_to_run if s.get('ticker_tested') == ticker]
            data_cache[ticker] = add_technical_indicators(df_raw, strategies_for_this_ticker)

        df_ready = data_cache[ticker]  # 캐시에서 준비된 데이터 가져오기

        # 백테스트 실행!
        trade_log_df, portfolio_history_df = run_backtest(df_ready.copy(), strategy_params)

        # 결과 분석 및 저장
        if portfolio_history_df is not None and not portfolio_history_df.empty:
            summary = analyze_performance_detailed(portfolio_history_df, trade_log_df, INITIAL_CAPITAL, strategy_params,
                                                   TARGET_INTERVAL)
            if summary:
                summary['티커'] = ticker
                all_results.append(summary)
                log_results_to_csv(summary)  # 결과를 CSV 파일에 즉시 기록

    # 모든 실험 완료 후 최종 결과 요약 출력
    if all_results:
        results_df = pd.DataFrame(all_results)
        # 캘머 지수(Calmar Ratio)를 기준으로 내림차순 정렬
        results_df = results_df.sort_values(by='Calmar', ascending=False)
        print("\n\n" + "=" * 90)
        print("<<< 최종 백테스트 결과 요약 (정렬 기준: Calmar 내림차순) >>>".center(85))
        print("=" * 90)
        cols_to_display = ['티커', '실험명', 'ROI (%)', 'MDD (%)', 'Calmar', 'Sharpe', 'Profit Factor', 'Win Rate (%)',
                           'Total Trades']
        print(results_df[cols_to_display].to_string(index=False))  # to_string으로 모든 행이 보이게 출력
        print("=" * 90)
