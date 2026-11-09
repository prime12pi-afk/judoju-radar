"""데이터 수집: FinanceDataReader 기반 (KRX 로그인 불필요).

- 종목 목록/시가총액: fdr.StockListing('KRX')
- 업종: fdr.StockListing('KRX-DESC') 의 Industry
- 일봉: fdr.DataReader(code, start)
- 지수: KS11(코스피), KQ11(코스닥)
수집 결과는 하루 단위로 디스크에 캐시합니다.
"""
from __future__ import annotations

import datetime as dt
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

INDEX_CODE = {"KOSPI": "KS11", "KOSDAQ": "KQ11"}


LISTING_CACHE = CACHE_DIR / "listing.pkl"
_NAVER_M = "https://m.stock.naver.com/api/stocks"


def _listing_krx() -> pd.DataFrame:
    """KRX 기반(FinanceDataReader). 망에서 KRX가 차단되면 예외가 난다."""
    listing = fdr.StockListing("KRX")
    desc = fdr.StockListing("KRX-DESC")[["Code", "Industry"]].drop_duplicates("Code")
    listing = listing.merge(desc, on="Code", how="left")
    return listing[["Code", "Name", "Market", "Industry", "Marcap"]]


def _get_json(url: str, params: dict | None = None) -> dict:
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=_UA, timeout=10)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:  # 마지막 페이지를 넘으면 빈 목록 대신 404가 온다 → '끝'으로 처리
                return {}
            last = RuntimeError(f"HTTP {r.status_code} {url}")
        except Exception as e:
            last = e
        time.sleep(0.6 * (attempt + 1))
    raise last or RuntimeError(url)


def _listing_naver(per_market: int = 450) -> pd.DataFrame:
    """네이버 증권(비공식) 시가총액 목록 + 업종 그룹. KRX 접속이 막힌 망에서의 대체 경로."""
    # 업종 그룹 → 종목코드 매핑
    # 그룹 목록도, 그룹 안의 종목 목록도 페이지로 나뉘어 있어 끝까지 읽어야 한다.
    groups: dict[int, str] = {}
    for page in range(1, 30):
        part = _get_json(f"{_NAVER_M}/industry", {"page": page, "pageSize": 100}).get("groups", [])
        new = {g["no"]: g["name"] for g in part if g["no"] not in groups}
        if not new:
            break
        groups.update(new)

    industry: dict[str, str] = {}
    for no, name in groups.items():
        for page in range(1, 30):
            try:
                members = _get_json(f"{_NAVER_M}/industry/{no}", {"page": page, "pageSize": 100}).get("stocks", [])
            except Exception:
                break
            if not members:
                break
            for s in members:
                industry[s["itemCode"]] = name
            if len(members) < 100:
                break

    rows = []
    for market in ("KOSPI", "KOSDAQ"):
        got = 0
        for page in range(1, 40):  # KOSPI 목록에는 ETF/ETN이 섞여 있어 보통주만 걸러 가며 채운다
            stocks = _get_json(f"{_NAVER_M}/marketValue/{market}", {"page": page, "pageSize": 100}).get("stocks", [])
            if not stocks:
                break
            for s in stocks:
                if s.get("stockEndType") != "stock":
                    continue
                rows.append(dict(
                    Code=s["itemCode"], Name=s["stockName"], Market=market,
                    Industry=industry.get(s["itemCode"], "기타"),
                    Marcap=_num(s.get("marketValueRaw") or s.get("marketValue")),
                ))
                got += 1
            if got >= per_market:
                break
    return pd.DataFrame(rows)


MIN_USABLE_ROWS = 300  # 필터 적용 후 이 수 미만이면 '쓸 수 없는 목록'으로 보고 다음 출처를 시도


def _filter_listing(listing: pd.DataFrame) -> pd.DataFrame:
    """분석 대상만 남긴다: 코스피/코스닥 보통주, 스팩 제외, 시가총액이 있는 종목."""
    out = listing[listing["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
    out["Marcap"] = pd.to_numeric(out["Marcap"], errors="coerce")
    out = out[out["Code"].astype(str).str.endswith("0")]  # 우선주(끝자리 0 아님) 제외
    out = out[~out["Name"].str.contains("스팩|제[0-9]+호", regex=True, na=False)]
    out = out[out["Marcap"] > 0]
    out["Industry"] = out["Industry"].fillna("기타")
    return out


def _get_listing() -> tuple[pd.DataFrame, str]:
    """오늘자 저장본 → KRX → 네이버 → (마지막 성공한) 저장본 순으로 시도.

    '행이 있다'가 아니라 '필터 후 쓸 수 있는 행이 충분하다'를 기준으로 판단한다.
    (예: KRX가 시가총액이 빈 목록을 주면 다음 출처로 넘어간다)
    """
    if LISTING_CACHE.exists():
        stamp = dt.date.fromtimestamp(LISTING_CACHE.stat().st_mtime)
        if stamp == dt.date.today():
            cached = pd.read_pickle(LISTING_CACHE)
            if len(_filter_listing(cached)) >= MIN_USABLE_ROWS:
                return cached, cached.attrs.get("source", "cache")
    errors = []
    # 소스별 시간 제한: KRX는 막히면 응답이 없을 수 있어 짧게, 네이버는 요청이 많아 길게
    for name, fn, limit in (("KRX", _listing_krx, 25), ("NAVER", _listing_naver, 180)):
        try:
            df = _call_with_timeout(fn, limit)
            usable = len(_filter_listing(df))
            if usable >= MIN_USABLE_ROWS:
                df.attrs["source"] = name
                df.to_pickle(LISTING_CACHE)
                return df, name
            errors.append(f"{name}: 쓸 수 있는 종목 {usable}개 (원본 {len(df)}행)")
        except Exception as e:  # 차단/네트워크 오류 → 다음 경로
            errors.append(f"{name}: {type(e).__name__}")
    if LISTING_CACHE.exists():
        stale = pd.read_pickle(LISTING_CACHE)
        if len(_filter_listing(stale)) >= MIN_USABLE_ROWS:
            return stale, "stale-cache"
    raise RuntimeError("종목 목록을 가져오지 못했습니다 (" + "; ".join(errors) + ")")


def load_universe(n_kospi: int, n_kosdaq: int) -> pd.DataFrame:
    """시가총액 상위 종목 목록 (스팩/우선주 등 제외). 출처는 결과의 attrs['listing_source']."""
    listing_raw, source = _get_listing()
    listing = _filter_listing(listing_raw)

    parts = []
    for market, n in (("KOSPI", n_kospi), ("KOSDAQ", n_kosdaq)):
        if n > 0:
            parts.append(listing[listing["Market"] == market].nlargest(n, "Marcap"))
    if not parts:
        return listing.iloc[0:0]
    out = pd.concat(parts, ignore_index=True)[["Code", "Name", "Market", "Industry", "Marcap"]]
    out.attrs["listing_source"] = source
    return out


def _fetch_one(code: str, start: str) -> tuple[str, pd.DataFrame | None]:
    for _ in range(2):
        try:
            df = _call_with_timeout(lambda: fdr.DataReader(code, start), 25)
            if df is not None and len(df) > 0:
                return code, df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass
    return code, None


def load_prices(codes: list[str], start: str, progress=None) -> dict[str, pd.DataFrame]:
    """여러 종목 일봉을 병렬 수집. 하루 단위 디스크 캐시."""
    today = dt.date.today().isoformat()
    cache_file = CACHE_DIR / f"prices_{today}_{start}.pkl"
    cache: dict[str, pd.DataFrame] = {}
    if cache_file.exists():
        try:
            cache = pickle.loads(cache_file.read_bytes())
        except Exception:
            cache = {}

    todo = [c for c in codes if c not in cache]
    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_fetch_one, c, start) for c in todo]
            for f in as_completed(futs):
                code, df = f.result()
                if df is not None:
                    cache[code] = df
                done += 1
                if progress:
                    progress(done / len(todo))
        cache_file.write_bytes(pickle.dumps(cache))
    return {c: cache[c] for c in codes if c in cache}


# ---------------------------------------------------------------- 외국인·기관 수급
# 네이버 증권 모바일이 쓰는 JSON 엔드포인트(비공식). 최대 60거래일, 호출 한도/변경 가능성에 유의.
FLOW_URL = "https://m.stock.naver.com/api/stock/{code}/trend"
_UA = {"User-Agent": "Mozilla/5.0"}


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").replace("+", "").replace("%", ""))
    except ValueError:
        return float("nan")


def _fetch_flow(code: str) -> tuple[str, pd.DataFrame | None]:
    for attempt in range(3):
        try:
            r = requests.get(FLOW_URL.format(code=code), params={"pageSize": 60},
                             headers=_UA, timeout=10)
            if r.status_code == 200:
                rows = r.json()
                if rows:
                    df = pd.DataFrame(rows)
                    out = pd.DataFrame({
                        "date": pd.to_datetime(df["bizdate"]),
                        "frgn": df["foreignerPureBuyQuant"].map(_num),
                        "org": df["organPureBuyQuant"].map(_num),
                        "ind": df["individualPureBuyQuant"].map(_num),
                        "hold": df["foreignerHoldRatio"].map(_num),
                        "vol": df["accumulatedTradingVolume"].map(_num),
                    }).set_index("date").sort_index()
                    return code, out
            time.sleep(0.6 * (attempt + 1))
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    return code, None


def load_flows(codes: list[str], progress=None) -> dict[str, pd.DataFrame]:
    """종목별 외국인/기관/개인 일별 순매수(주). 하루 단위 디스크 캐시, 실패 종목은 다음에 재시도."""
    cache_file = CACHE_DIR / f"flows_{dt.date.today().isoformat()}.pkl"
    cache: dict[str, pd.DataFrame] = {}
    if cache_file.exists():
        try:
            cache = pickle.loads(cache_file.read_bytes())
        except Exception:
            cache = {}
    todo = [c for c in codes if c not in cache]
    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=6) as ex:  # 서버 부담을 줄이려 병렬도 낮게
            futs = [ex.submit(_fetch_flow, c) for c in todo]
            for f in as_completed(futs):
                code, df = f.result()
                if df is not None:
                    cache[code] = df
                done += 1
                if progress:
                    progress(done / len(todo))
        cache_file.write_bytes(pickle.dumps(cache))
    return {c: cache[c] for c in codes if c in cache}


YF_INDEX = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11"}


def _call_with_timeout(fn, seconds: float):
    """fn()을 별도 스레드에서 실행하고 seconds 안에 안 끝나면 TimeoutError (응답 없는 서버 대비)."""
    import threading

    box: dict = {}

    def runner():
        try:
            box["r"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=runner, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise TimeoutError(f"{seconds:.0f}초 안에 응답이 없습니다")
    if "e" in box:
        raise box["e"]
    return box["r"]


def _index_yahoo(market: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(YF_INDEX[market], start=start, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if len(df) < 100:
        raise ValueError("지수 데이터가 너무 적습니다")
    return df


def _index_fdr(market: str, start: str) -> pd.DataFrame:
    df = fdr.DataReader(INDEX_CODE[market], start)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_index(market: str, start: str) -> pd.DataFrame:
    """지수 일봉. 하루 단위 디스크 캐시 → Yahoo → FinanceDataReader(시간 제한) 순."""
    cache_file = CACHE_DIR / f"index_{market}_{start}_{dt.date.today().isoformat()}.pkl"
    if cache_file.exists():
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass
    errors = []
    for name, fn in (("Yahoo", _index_yahoo), ("FDR", _index_fdr)):
        try:
            df = _call_with_timeout(lambda: fn(market, start), 25)
            df.to_pickle(cache_file)
            return df
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}")
    raise RuntimeError(f"{market} 지수를 가져오지 못했습니다 ({', '.join(errors)})")
