import pyupbit
import pandas as pd
import sqlite3
import time
import os  # os 라이브러리는 현재 코드에서 직접 사용되진 않지만, 파일 경로 관련 작업 시 유용할 수 있습니다.


# --- 데이터 수집 함수 (이전과 동일, 변경 없음) ---
def get_all_ohlcv(ticker: str, interval: str) -> pd.DataFrame:
    """
    pyupbit을 사용하여 특정 종목의 가능한 모든 과거 OHLCV 데이터를 가져옵니다.
    Upbit API는 한 번에 최대 200개의 캔들만 반환하므로, 루프를 돌며 순차적으로 가져옵니다.
    *** 무한 루프 방지 로직이 추가된 수정 버전입니다. ***
    """
    print(f"'{ticker}'의 전체 {interval} 데이터를 수집합니다. 시간이 걸릴 수 있습니다...")
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=200)

    # --- ✨ 디버깅 로그 추가 ✨ ---
    if df is not None and not df.empty:
        print("\n" + "="*25)
        print("--- [디버깅] get_all_ohlcv: 초기 수신 데이터 확인 ---")
        print(f"초기 수신 데이터 행 수: {len(df)}")
        print("초기 수신 데이터의 마지막 5행:")
        print(df.tail(5))
        print(f"초기 수신 데이터의 가장 최신 날짜: {df.index[-1]}")
        print("--- [디버깅] 확인 끝 ---")
        print("="*25 + "\n")
    # --- 디버깅 로그 끝 ---

    if df is None or df.empty:
        print(f"'{ticker}' 데이터를 가져오는 데 실패했습니다.")
        return pd.DataFrame()

    all_data = [df]
    oldest_date = df.index[0]

    while True:
        time.sleep(0.2)  # API 요청 간격 유지
        df_more = pyupbit.get_ohlcv(ticker, interval=interval, to=oldest_date, count=200)
        if df_more is None or df_more.empty:
            print("API로부터 더 이상 데이터를 반환받지 못했습니다. 수집을 완료합니다.")
            break
        if len(df_more) > 1:
            df_more = df_more.iloc[:-1]
        if df_more.empty:
            print("더 이상 분석할 데이터가 없습니다. 수집을 완료합니다.")
            break
        new_oldest_date = df_more.index[0]
        if new_oldest_date == oldest_date:
            print("데이터의 가장 시작점에 도달하여 더 이상 진행되지 않으므로 수집을 중단합니다.")
            break
        all_data.append(df_more)
        oldest_date = new_oldest_date
        print(f"{oldest_date} 이전 데이터 수집 중...")

    df_final = pd.concat(all_data).sort_index()
    print(f"'{ticker}' 데이터 총 {len(df_final)}건 수집 및 정렬 완료.")
    return df_final


def update_ohlcv_db(ticker: str, interval: str, db_path: str):
    """
    OHLCV 데이터를 SQLite DB에 증분 업데이트합니다.
    테이블이 없으면 전체 데이터를, 있으면 최신 데이터만 추가합니다.
    """
    table_name = f"{ticker.replace('-', '_')}_{interval}"
    con = sqlite3.connect(db_path)
    try:
        last_date = None
        try:
            query = f'SELECT MAX("timestamp") FROM "{table_name}"'
            # last_date_df = pd.read_sql_query(query, con) # 이전 코드 방식
            # if not last_date_df.empty and last_date_df.iloc[0,0] is not None:
            #     last_date = pd.to_datetime(last_date_df.iloc[0,0])
            cursor = con.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result and result[0] is not None:
                last_date = pd.to_datetime(result[0])

            if last_date:
                print(f"DB에 저장된 '{table_name}' 테이블의 마지막 데이터 시점: {last_date}")
            else:
                print(f"'{table_name}' 테이블이 없거나 비어있습니다. 전체 데이터 수집을 시작합니다.")
        except sqlite3.OperationalError:  # 테이블이 없는 경우 OperationalError 발생
            print(f"'{table_name}' 테이블이 존재하지 않습니다. 전체 데이터 수집을 시작합니다.")
        except Exception as e:  # 기타 예외 처리
            print(f"마지막 데이터 시점 조회 중 예기치 않은 오류 발생 ({table_name}): {e}")
            print(f"'{table_name}' 테이블 전체 데이터 수집을 시도합니다.")

        if last_date is None:
            df_full = get_all_ohlcv(ticker, interval)
            if not df_full.empty:
                df_full.index.name = 'timestamp'
                df_full.to_sql(table_name, con, if_exists='replace', index=True)
                print(f"'{table_name}' 테이블에 전체 데이터 {len(df_full)}건을 새로 저장했습니다.")
        else:
            # pyupbit.get_ohlcv는 기본적으로 최근 200개를 가져옴
            # 증분 업데이트 시, 마지막 날짜 이후의 모든 데이터를 가져오려면 반복 로직 필요
            # 여기서는 간단히 최근 200개 중 새로운 데이터만 추가하는 방식으로 유지
            # 더 완벽한 증분 업데이트는 get_all_ohlcv와 유사한 로직을 last_date 이후부터 적용해야 함

            print(f"'{table_name}' 테이블에 대한 증분 업데이트를 시도합니다 (마지막 데이터: {last_date})...")
            df_new_potential = pyupbit.get_ohlcv(ticker, interval=interval, count=200)  # 우선 최근 200개 가져오기

            if df_new_potential is not None and not df_new_potential.empty:
                df_new_potential.index.name = 'timestamp'
                # 시간대 정보 통일 (DB 저장 데이터와 비교를 위해)
                if df_new_potential.index.tz is not None:
                    df_new_potential.index = df_new_potential.index.tz_localize(None)
                if last_date.tz is not None:  # last_date도 naive로 만들어 비교
                    last_date_naive = last_date.tz_localize(None)
                else:
                    last_date_naive = last_date

                df_to_append = df_new_potential[df_new_potential.index > last_date_naive]

                if not df_to_append.empty:
                    # 테이블이 이미 존재하므로 append
                    df_to_append.to_sql(table_name, con, if_exists='append', index=True)
                    print(f"'{table_name}' 테이블에 새로운 데이터 {len(df_to_append)}건을 추가했습니다.")
                else:
                    print(f"'{table_name}' 테이블에 대한 새로운 데이터가 없습니다.")
            else:
                print(f"'{ticker}'에 대한 최신 데이터를 가져오지 못했습니다.")

    except Exception as e:
        print(f"'{ticker}' DB 업데이트 중 오류 발생: {e}")
    finally:
        con.close()
        # print(f"'{ticker}' 데이터베이스 연결을 종료했습니다.") # 각 티커마다 너무 많은 로그 방지


# --- 메인 실행 (10개 우량주 수집용으로 수정) ---
if __name__ == "__main__":
    # ★★★★★ 수집할 우량 암호화폐 10개 티커 리스트 ★★★★★
    # 실제 우량주 리스트로 교체해주세요. (예시 리스트)
    # 이 리스트는 사용자가 직접 조사하여 결정해야 합니다.
    BLUE_CHIP_TICKERS = [
        "KRW-BTC",
        # 여기에 실제 선정하신 10개 코인의 티커를 정확히 입력하세요.
    ]
    # "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-SOL",
    #     "KRW-AVAX", "KRW-DOGE", "KRW-LINK","KRW-TRX", "KRW-HYPE",
    #     "KRW-SUI", "KRW-XLM"

    # 수집할 데이터 간격 (예: "day" 또는 "minute60")
    TARGET_INTERVALS = ["day"] #"minute60"

    # 데이터를 저장할 DB 파일 경로 (모든 코인 데이터를 이 파일 하나에 테이블로 저장)
    DB_FILE_PATH = "upbit_ohlcv.db"

    print(f"총 {len(BLUE_CHIP_TICKERS)}개 우량 암호화폐의 '{TARGET_INTERVALS}' 간격 데이터 수집을 시작합니다.")
    print(f"데이터는 '{DB_FILE_PATH}' 파일에 각 코인별 테이블로 저장됩니다.")

    # 각 간격(interval)에 대해 반복
    for interval in TARGET_INTERVALS:
        print(f"\n===== '{interval}' 간격 데이터 처리 시작 =====")

        # 각 티커(ticker)에 대해 반복
        for ticker in BLUE_CHIP_TICKERS:
            print(f"\n processing '{ticker}' for interval '{interval}'...")
            try:
                update_ohlcv_db(
                    ticker=ticker,
                    interval=interval,  # 현재 반복 중인 interval 값을 전달
                    db_path=DB_FILE_PATH
                )
            except KeyboardInterrupt:
                print("사용자에 의해 프로그램이 중단되었습니다.")
                # 프로그램 전체를 종료하기 위해 바깥쪽 루프까지 탈출
                exit()
            except Exception as e:
                print(f"!!! '{ticker}' 처리 중 심각한 오류 발생: {e} !!!")

            # 각 티커 처리 후 약간의 딜레이를 두어 API 서버에 부담을 줄임
            time.sleep(0.5)

    print("\n모든 지정된 암호화폐 데이터 업데이트 작업 완료.")