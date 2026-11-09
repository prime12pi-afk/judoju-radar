"""주도주 판단 지표 계산.

핵심 아이디어: 하락장에서 지수보다 덜 빠지고(방어력), RS 라인이 주가보다 먼저 신고가를 내며,
반등 구간에서 지수를 앞서는 종목이 다음 상승장의 주도주 후보가 될 가능성이 높다는 경험칙.
(통계적 경향일 뿐 수익을 보장하지 않습니다.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# IBD 방식 가중 수익률: 최근 3개월에 가중치 2배
RS_PERIODS = ((63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2))


@dataclass
class MarketPhase:
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    last_date: pd.Timestamp
    peak: float
    trough: float
    last: float
    drawdown: float          # 고점 대비 저점 낙폭
    from_peak: float         # 고점 대비 현재
    rebound: float           # 저점 대비 현재
    days_since_trough: int
    ftd_status: str          # 팔로우스루 데이 판정 문구
    stage: str               # 하락 중 / 반등 시도 / 반등 확인 / 조정 없음


def detect_phase(idx: pd.DataFrame, lookback: int = 120) -> MarketPhase:
    """최근 lookback 거래일 내 고점→저점→현재로 시장 국면 판정."""
    w = idx.tail(lookback)
    c = w["Close"]
    peak_date = c.idxmax()
    after = c.loc[peak_date:]
    trough_date = after.idxmin()
    peak, trough, last = float(c[peak_date]), float(c[trough_date]), float(c.iloc[-1])
    since = idx.loc[trough_date:]
    days_since = len(since) - 1

    # 팔로우스루 데이: 저점 이후 상승 마감 첫날을 1일차로 보고, 4일차 이후
    # 지수가 +1.5% 이상 오르며 전일보다 거래량이 늘면 확인 신호. 저점 이탈 시 카운트 리셋.
    ftd_status = "해당 없음"
    stage = "조정 없음"
    dd = trough / peak - 1
    if dd <= -0.07:  # 고점 대비 -7% 이상 빠졌을 때만 조정으로 간주
        rets = since["Close"].pct_change()
        day = 0
        ftd_date = None
        low = float(since["Low"].iloc[0])
        for i in range(1, len(since)):
            if float(since["Low"].iloc[i]) < low:  # 저점 갱신 → 리셋
                low, day = float(since["Low"].iloc[i]), 0
                ftd_date = None
                continue
            if day == 0:
                if rets.iloc[i] > 0:
                    day = 1
                continue
            day += 1
            if (
                day >= 4
                and ftd_date is None
                and rets.iloc[i] >= 0.015
                and since["Volume"].iloc[i] > since["Volume"].iloc[i - 1]
            ):
                ftd_date = since.index[i]
        if ftd_date is not None:
            ftd_status = f"확인됨 ({ftd_date:%Y-%m-%d})"
            stage = "반등 확인"
        elif day >= 1:
            ftd_status = f"반등 시도 {day}일차 (미확인)"
            stage = "반등 시도"
        else:
            ftd_status = "저점 탐색 중"
            stage = "하락 중"
    return MarketPhase(
        peak_date=peak_date, trough_date=trough_date, last_date=c.index[-1],
        peak=peak, trough=trough, last=last, drawdown=dd,
        from_peak=last / peak - 1, rebound=last / trough - 1,
        days_since_trough=days_since, ftd_status=ftd_status, stage=stage,
    )


def _weighted_return(close: pd.Series) -> float:
    """가용한 기간만으로 가중 수익률 계산(상장 1년 미만 종목 대응)."""
    num = den = 0.0
    n = len(close)
    for period, wt in RS_PERIODS:
        if n > period:
            num += wt * (close.iloc[-1] / close.iloc[-1 - period] - 1)
            den += wt
    return num / den if den else np.nan


def _flow_metrics(fl: pd.DataFrame, idx_close: pd.Series) -> dict:
    """외국인·기관 수급 지표 (최근 최대 60거래일, 순매수량 ÷ 거래량으로 정규화)."""
    f = fl.join(idx_close.rename("idx"), how="left")
    if len(f) < 20:
        return {}
    smart = f["frgn"] + f["org"]
    t = f.tail(20)
    smart_t = smart.tail(20)
    out = {
        "frgn20": float(t["frgn"].sum() / t["vol"].sum()),
        "org20": float(t["org"].sum() / t["vol"].sum()),
        "smart20": float(smart_t.sum() / t["vol"].sum()),
        "smart60": float(smart.sum() / f["vol"].sum()),
        "smart_days20": float((smart_t > 0).mean()),
        "hold_chg": float(f["hold"].iloc[-1] - f["hold"].iloc[0]),  # 외국인 보유비율 변화(%p)
    }
    # 지수가 내린 날의 외국인+기관 순매수 비중: 양(+)이면 하락을 받아낸 종목
    down = f["idx"].pct_change() < 0
    if down.sum() >= 5 and f.loc[down, "vol"].sum() > 0:
        out["dip_buy"] = float(smart[down].sum() / f.loc[down, "vol"].sum())
    return out


def _stock_metrics(df: pd.DataFrame, idx: pd.DataFrame, ph: MarketPhase,
                   flow: pd.DataFrame | None = None) -> dict | None:
    d = df.join(idx["Close"].rename("idx"), how="inner").dropna(subset=["Close", "idx"])
    d = d[d["Close"] > 0]
    if len(d) < 130:
        return None
    if d.index[-1] != ph.last_date:  # 거래정지 등으로 최근 데이터 없음
        return None

    close, ic = d["Close"], d["idx"]
    out: dict = {"rs_raw": _weighted_return(close) - _weighted_return(ic)}

    # --- 하락 구간(고점~현재) 방어력 ---
    win = d.loc[ph.peak_date:]
    r, ir = win["Close"].pct_change().dropna(), win["idx"].pct_change().dropna()
    down = ir < 0
    out["down_days"] = int(down.sum())
    out["defense"] = float((r[down] > ir[down]).mean()) if down.sum() >= 5 else np.nan
    out["stock_dd"] = float(win["Close"].iloc[-1] / win["Close"].iloc[0] - 1)
    out["idx_dd"] = float(win["idx"].iloc[-1] / win["idx"].iloc[0] - 1)
    out["rel_dd"] = out["stock_dd"] - out["idx_dd"]
    out["max_dd"] = float((win["Close"] / win["Close"].cummax() - 1).min())

    # --- 저점 이후 반등 상대강도 ---
    if ph.days_since_trough >= 1 and ph.trough_date in d.index:
        sub = d.loc[ph.trough_date:]
        out["rebound"] = float(sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1)
        out["rel_rebound"] = out["rebound"] - float(sub["idx"].iloc[-1] / sub["idx"].iloc[0] - 1)
    else:
        out["rebound"] = out["rel_rebound"] = np.nan

    # --- RS 라인(종목/지수) 선행 신고가 ---
    rs_line = close / ic
    look = 120
    rs_hi = rs_line.tail(look).max()
    px_hi = close.tail(look).max()
    out["rs_line_gap"] = float(rs_line.iloc[-1] / rs_hi - 1)       # 0에 가까울수록 신고가권
    out["px_gap"] = float(close.iloc[-1] / px_hi - 1)              # 주가 고점 대비
    out["rs_lead"] = bool(out["rs_line_gap"] >= -0.01 and out["px_gap"] <= -0.05)

    # --- 거래량 수급(최근 20일 상승일/하락일 거래량 비) ---
    t = d.tail(21)
    chg = t["Close"].diff()
    up_v = t.loc[chg > 0, "Volume"].sum()
    dn_v = t.loc[chg < 0, "Volume"].sum()
    out["vol_ratio"] = float(up_v / dn_v) if dn_v > 0 else (3.0 if up_v > 0 else np.nan)

    # --- 추세 ---
    ma50 = close.rolling(50).mean().iloc[-1]
    out["above_50ma"] = bool(close.iloc[-1] > ma50)
    out["close"] = float(close.iloc[-1])
    out["ret_1m"] = float(close.iloc[-1] / close.iloc[-21] - 1)

    if flow is not None:
        out.update(_flow_metrics(flow, idx["Close"]))
    return out


def _pct(s: pd.Series) -> pd.Series:
    """0~100 백분위(높을수록 좋음)."""
    return s.rank(pct=True) * 100


def compute_scores(
    universe: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    indices: dict[str, pd.DataFrame],
    phases: dict[str, MarketPhase],
    weights: dict[str, float],
    flows: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    flows = flows or {}
    rows = []
    for _, u in universe.iterrows():
        df = prices.get(u["Code"])
        if df is None:
            continue
        m = _stock_metrics(df, indices[u["Market"]], phases[u["Market"]], flows.get(u["Code"]))
        if m:
            m.update(Code=u["Code"], Name=u["Name"], Market=u["Market"],
                     Industry=u["Industry"], Marcap=u["Marcap"])
            rows.append(m)
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)

    # 시장별 백분위가 아니라 전체 유니버스 기준 백분위 (IBD 방식)
    res["RS등급"] = _pct(res["rs_raw"]).round().clip(1, 99)
    res["방어력점수"] = (
        0.5 * _pct(res["defense"].fillna(res["defense"].median()))
        + 0.5 * _pct(res["rel_dd"])
    )
    res["반등점수"] = _pct(res["rel_rebound"].fillna(res["rel_rebound"].median()))
    res["거래량점수"] = _pct(res["vol_ratio"].fillna(res["vol_ratio"].median()))
    res["선행점수"] = np.where(res["rs_lead"], 100.0, _pct(res["rs_line_gap"]) * 0.6)

    # 외국인·기관 수급: 최근 20일 순매수, 하락일 순매수(저가 매수), 60일 순매수, 외국인 보유비율 변화
    def _pf(col: str) -> pd.Series:
        s = res[col] if col in res else pd.Series(np.nan, index=res.index)
        return _pct(s.fillna(s.median())) if s.notna().any() else pd.Series(50.0, index=res.index)

    has_flow = "smart20" in res and res["smart20"].notna().any()
    res["외인기관점수"] = (
        0.35 * _pf("smart20") + 0.35 * _pf("dip_buy") + 0.15 * _pf("smart60") + 0.15 * _pf("hold_chg")
    ) if has_flow else 0.0

    # 데이터가 없는 항목(저점 이후 반등이 아직 없음, 수급 수집 실패)은 제외하고 가중치 재분배
    w = dict(weights)
    if res["반등점수"].isna().all():
        w["rebound"] = 0.0
        res["반등점수"] = 0.0
    if not has_flow:
        w["flow"] = 0.0
    wsum = sum(w.values()) or 1.0
    res["종합점수"] = (
        w["rs"] * res["RS등급"]
        + w["defense"] * res["방어력점수"]
        + w["rebound"] * res["반등점수"]
        + w["lead"] * res["선행점수"]
        + w["volume"] * res["거래량점수"]
        + w["flow"] * res["외인기관점수"]
    ) / wsum
    res.attrs["has_flow"] = bool(has_flow)
    return res.sort_values("종합점수", ascending=False).reset_index(drop=True)


def sector_table(res: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    g = res.groupby("Industry")
    t = g.agg(
        종목수=("Code", "count"),
        RS등급_중앙값=("RS등급", "median"),
        방어력_중앙값=("방어력점수", "median"),
        반등_중앙값=("반등점수", "median"),
        외인기관_중앙값=("외인기관점수", "median"),
        종합_중앙값=("종합점수", "median"),
        RS선행_종목수=("rs_lead", "sum"),
    )
    t = t[t["종목수"] >= min_n]
    top = (
        res.sort_values("종합점수", ascending=False)
        .groupby("Industry")["Name"].apply(lambda s: ", ".join(s.head(3)))
        .rename("상위종목")
    )
    return t.join(top).sort_values("종합_중앙값", ascending=False)
