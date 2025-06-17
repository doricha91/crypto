# 2. collect_ohlcv.py

# 필요한 라이브러리들을 가져옵니다.
import pyupbit  # 업비트 API를 사용하기 위한 라이브러리
import pandas as pd  # 데이터 분석 및 조작을 위한 라이브러리 (표 형태 데이터 처리)
import sqlite3  # 파이썬에 내장된 간단한 데이터베이스(SQLite)를 사용하기 위한 라이브러리
import time  # API 요청 사이에 시간 지연을 주기 위한 라이브러리


# --- 데이터 수집 함수 ---
def get_all_ohlcv(ticker: str, interval: str) -> pd.DataFrame:
    """
    pyupbit을 사용하여 특정 종목의 가능한 모든 과거 OHLCV 데이터를 가져옵니다.
    Upbit API는 한 번에 최대 200개의 캔들만 반환하므로, 루프를 돌며 순차적으로 가져옵니다.
    """
    # 함수가 시작되었음을 알리는 메시지를 출력합니다.
    print(f"'{ticker}'의 전체 {interval} 데이터를 수집합니다. 시간이 걸릴 수 있습니다...")

    # 가장 최근 데이터부터 200개를 우선 가져옵니다.
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=200)

    # 데이터를 제대로 받아왔는지 확인하는 디버깅 코드
    if df is not None and not df.empty:
        print("\n" + "=" * 25)
        print("--- [디버깅] get_all_ohlcv: 초기 수신 데이터 확인 ---")
        print(f"초기 수신 데이터 행 수: {len(df)}")
        print("초기 수신 데이터의 마지막 5행:")
        print(df.tail(5))  # 받아온 데이터의 마지막 5줄을 보여줍니다.
        print(f"초기 수신 데이터의 가장 최신 날짜: {df.index[-1]}")
        print("--- [디버깅] 확인 끝 ---")
        print("=" * 25 + "\n")

    # 만약 데이터 수신에 실패했다면, 빈 DataFrame을 반환하고 함수를 종료합니다.
    if df is None or df.empty:
        print(f"'{ticker}' 데이터를 가져오는 데 실패했습니다.")
        return pd.DataFrame()

    # 받아온 데이터를 리스트에 저장해두고, 가장 오래된 날짜를 기록합니다.
    all_data = [df]
    oldest_date = df.index[0]

    # 무한 루프를 돌면서 과거 데이터를 계속 요청합니다.
    while True:
        # 업비트 서버에 부담을 주지 않기 위해 0.2초간 쉽니다. (API Rate Limit 방지)
        time.sleep(0.2)

        # 가장 오래된 날짜(oldest_date) 이전의 데이터 200개를 추가로 요청합니다.
        df_more = pyupbit.get_ohlcv(ticker, interval=interval, to=oldest_date, count=200)

        # 더 이상 받아올 데이터가 없으면 루프를 중단합니다.
        if df_more is None or df_more.empty:
            print("API로부터 더 이상 데이터를 반환받지 못했습니다. 수집을 완료합니다.")
            break

        # 중복 데이터를 피하기 위해, 받아온 데이터의 마지막 행(가장 오래된 데이터)은 제외합니다.
        if len(df_more) > 1:
            df_more = df_more.iloc[:-1]

        # 마지막 행을 제외하고 데이터가 없다면 루프 중단
        if df_more.empty:
            print("더 이상 분석할 데이터가 없습니다. 수집을 완료합니다.")
            break

        # 새로 받아온 데이터 묶음에서 가장 오래된 날짜를 기록합니다.
        new_oldest_date = df_more.index[0]

        # 만약 이전 루프의 가장 오래된 날짜와 현재가 같다면, 데이터의 시작점에 도달한 것이므로 루프를 중단합니다.
        if new_oldest_date == oldest_date:
            print("데이터의 가장 시작점에 도달하여 더 이상 진행되지 않으므로 수집을 중단합니다.")
            break

        # 새로 받아온 데이터를 리스트에 추가하고, oldest_date를 갱신합니다.
        all_data.append(df_more)
        oldest_date = new_oldest_date
        print(f"{oldest_date} 이전 데이터 수집 중...")

    # 리스트에 저장된 모든 데이터 조각들을 하나로 합치고, 날짜 순으로 정렬합니다.
    df_final = pd.concat(all_data).sort_index()
    print(f"'{ticker}' 데이터 총 {len(df_final)}건 수집 및 정렬 완료.")
    return df_final


def update_ohlcv_db(ticker: str, interval: str, db_path: str):
    """
    OHLCV 데이터를 SQLite DB에 증분 업데이트합니다.
    테이블이 없으면 전체 데이터를, 있으면 최신 데이터만 추가합니다.
    """
    # DB에 저장될 테이블 이름을 만듭니다. (예: 'KRW-BTC' -> 'KRW_BTC_day')
    table_name = f"{ticker.replace('-', '_')}_{interval}"
    # 지정된 경로의 DB 파일에 연결합니다. 파일이 없으면 새로 생성됩니다.
    con = sqlite3.connect(db_path)
    try:
        last_date = None  # DB에 저장된 마지막 날짜를 저장할 변수
        try:
            # SQL 쿼리를 통해 테이블에서 가장 마지막(최신) timestamp를 조회합니다.
            query = f'SELECT MAX("timestamp") FROM "{table_name}"'
            cursor = con.cursor()
            cursor.execute(query)
            result = cursor.fetchone()  # 조회 결과를 가져옵니다.

            # 결과가 있고 비어있지 않다면, 날짜/시간 형태로 변환하여 last_date에 저장합니다.
            if result and result[0] is not None:
                last_date = pd.to_datetime(result[0])

            if last_date:
                print(f"DB에 저장된 '{table_name}' 테이블의 마지막 데이터 시점: {last_date}")
            else:
                print(f"'{table_name}' 테이블이 없거나 비어있습니다. 전체 데이터 수집을 시작합니다.")

        # 테이블이 존재하지 않으면 'OperationalError'가 발생합니다.
        except sqlite3.OperationalError:
            print(f"'{table_name}' 테이블이 존재하지 않습니다. 전체 데이터 수집을 시작합니다.")
        except Exception as e:
            print(f"마지막 데이터 시점 조회 중 예기치 않은 오류 발생 ({table_name}): {e}")
            print(f"'{table_name}' 테이블 전체 데이터 수집을 시도합니다.")

        # last_date가 None이라는 것은 DB에 데이터가 없다는 의미입니다.
        if last_date is None:
            # get_all_ohlcv 함수를 호출하여 전체 데이터를 가져옵니다.
            df_full = get_all_ohlcv(ticker, interval)
            if not df_full.empty:
                # DataFrame의 인덱스(날짜)에 'timestamp'라는 이름을 붙여줍니다.
                df_full.index.name = 'timestamp'
                # DataFrame을 DB의 테이블로 저장합니다. 테이블이 이미 있으면 덮어씁니다.
                df_full.to_sql(table_name, con, if_exists='replace', index=True)
                print(f"'{table_name}' 테이블에 전체 데이터 {len(df_full)}건을 새로 저장했습니다.")
        else:  # DB에 기존 데이터가 있는 경우 (증분 업데이트)
            print(f"'{table_name}' 테이블에 대한 증분 업데이트를 시도합니다 (마지막 데이터: {last_date})...")
            # 최신 데이터 200개를 우선 가져옵니다.
            df_new_potential = pyupbit.get_ohlcv(ticker, interval=interval, count=200)

            if df_new_potential is not None and not df_new_potential.empty:
                df_new_potential.index.name = 'timestamp'

                # 시간대 정보(timezone)가 다를 수 있으므로 통일시켜 비교 오류를 방지합니다.
                if df_new_potential.index.tz is not None:
                    df_new_potential.index = df_new_potential.index.tz_localize(None)
                if last_date.tz is not None:
                    last_date_naive = last_date.tz_localize(None)
                else:
                    last_date_naive = last_date

                # 새로 가져온 데이터 중, DB에 저장된 마지막 날짜보다 최신인 데이터만 필터링합니다.
                df_to_append = df_new_potential[df_new_potential.index > last_date_naive]

                if not df_to_append.empty:
                    # 필터링된 최신 데이터를 기존 테이블에 추가합니다.
                    df_to_append.to_sql(table_name, con, if_exists='append', index=True)
                    print(f"'{table_name}' 테이블에 새로운 데이터 {len(df_to_append)}건을 추가했습니다.")
                else:
                    print(f"'{table_name}' 테이블에 대한 새로운 데이터가 없습니다.")
            else:
                print(f"'{ticker}'에 대한 최신 데이터를 가져오지 못했습니다.")

    except Exception as e:
        print(f"'{ticker}' DB 업데이트 중 오류 발생: {e}")
    finally:
        # 작업이 끝나면 (성공하든 실패하든) 반드시 DB 연결을 닫아줍니다.
        con.close()


# 이 스크립트 파일이 직접 실행될 때만 아래 코드가 동작합니다.
if __name__ == "__main__":
    # ★★★★★ 수집할 암호화폐 티커 목록을 여기에 입력하세요 ★★★★★
    BLUE_CHIP_TICKERS = [
        "KRW-BTC",
        # "KRW-ETH", "KRW-XRP" 등 원하는 티커를 추가할 수 있습니다.
    ]
    #"KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-SOL",
    #"KRW-AVAX", "KRW-DOGE", "KRW-LINK","KRW-TRX",
    #"KRW-SUI", "KRW-XLM"

    # 수집할 데이터 시간 단위를 리스트로 지정합니다. (여러 개 지정 가능)
    TARGET_INTERVALS = ["day", "minute60"]

    # 모든 데이터를 저장할 DB 파일 이름
    DB_FILE_PATH = "upbit_ohlcv.db"

    print(f"총 {len(BLUE_CHIP_TICKERS)}개 암호화폐의 '{TARGET_INTERVALS}' 간격 데이터 수집을 시작합니다.")
    print(f"데이터는 '{DB_FILE_PATH}' 파일에 각 코인별 테이블로 저장됩니다.")

    # 지정된 각 시간 단위(interval)에 대해 반복 실행
    for interval in TARGET_INTERVALS:
        print(f"\n===== '{interval}' 간격 데이터 처리 시작 =====")

        # 지정된 각 티커(ticker)에 대해 반복 실행
        for ticker in BLUE_CHIP_TICKERS:
            print(f"\n processing '{ticker}' for interval '{interval}'...")
            try:
                # 핵심 함수인 update_ohlcv_db를 호출하여 데이터 수집/업데이트를 수행합니다.
                update_ohlcv_db(
                    ticker=ticker,
                    interval=interval,
                    db_path=DB_FILE_PATH
                )
            except KeyboardInterrupt:  # 사용자가 Ctrl+C를 눌러 중단했을 때
                print("사용자에 의해 프로그램이 중단되었습니다.")
                exit()  # 프로그램 강제 종료
            except Exception as e:
                print(f"!!! '{ticker}' 처리 중 심각한 오류 발생: {e} !!!")

            # API 서버 부하를 줄이기 위해 0.5초 대기
            time.sleep(0.5)

    print("\n모든 지정된 암호화폐 데이터 업데이트 작업 완료.")

