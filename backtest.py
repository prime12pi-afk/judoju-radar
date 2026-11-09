"""백테스트: 각 과거 시점에서 '그 날까지의 데이터만'으로 점수를 다시 계산하고,
상위 종목의 이후 N거래일 수익률을 유니버스 평균과 비교한다. (수급은 60일치뿐이라 제외)

주의(결과 해석 전 필독)
- 생존 편향: 유니버스가 '오늘 기준' 시총 상위 종목이라, 과거에 상장폐지/급락한 종목이 빠져 있고
  지금 큰 종목은 과거에 잘 올랐던 종목입니다. 절대 수익률은 부풀려지며, 상대 비교도 일부 왜곡됩니다.
- 표본: 큰 조정장은 10년에 몇 번뿐이고, 시점을 5~10일 간격으로 겹쳐 뽑아 독립 표본 수보다 훨씬 많아 보입니다.
- 거래비용·슬리피지·유동성은 반영하지 않았습니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import metrics

SCORE_COLS = {
    "종합점수": "종합점수(현재 가중치)",
    "RS등급": "RS 등급",
    "방어력점수": "하락장 방어력",
    "선행점수": "RS 라인 선행",
    "반등점수": "저점 이후 반등",
    "거래량점수": "거래량 우위",
}
WARMUP = 320  # 지표 계산에 필요한 최소 봉 수(252 + 여유)


def run_backtest(
    universe: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    indices: dict[str, pd.DataFrame],
    weights: dict[str, float],
    lookback: int = 120,
    step: int = 10,
    horizons: tuple[int, ...] = (20, 60),
    top_frac: float = 0.2,
    progress=None,
) -> pd.DataFrame:
    """시점×점수×기간 단위의 long 형식 결과를 돌려준다.

    열: date, stage(코스피 국면), from_peak, score, h, top_ret, univ_ret, idx_ret, n
    """
    w = dict(weights)
    w["flow"] = 0.0  # 과거 수급 데이터가 없다
    cal = indices["KOSPI"].index
    closes = pd.DataFrame({c: df["Close"] for c, df in prices.items()}).reindex(cal)
    hmax = max(horizons)
    positions = list(range(WARMUP, len(cal) - min(horizons), step))
    rows = []
    for k, i in enumerate(positions):
        t = cal[i]
        idx_t = {m: indices[m].loc[:t].tail(WARMUP) for m in indices}
        phases = {m: metrics.detect_phase(idx_t[m], lookback) for m in idx_t}
        pr_t = {c: df.loc[:t].tail(WARMUP) for c, df in prices.items()}
        res = metrics.compute_scores(universe, pr_t, idx_t, phases, w, None)
        if len(res) >= 30:
            for h in horizons:
                if i + h >= len(cal):
                    continue
                fwd = (closes.iloc[i + h] / closes.iloc[i] - 1).dropna()
                r = res[res["Code"].isin(fwd.index)]
                if len(r) < 30:
                    continue
                f = r["Code"].map(fwd)
                univ = float(f.mean())
                idx_ret = float(indices["KOSPI"]["Close"].iloc[i + h] / indices["KOSPI"]["Close"].iloc[i] - 1)
                n_top = max(5, int(len(r) * top_frac))
                for col, label in SCORE_COLS.items():
                    if col == "반등점수" and (r[col] == 0).all():
                        continue  # 반등 구간이 없던 시점(저점 갱신 중)에는 정의되지 않음
                    top = r.nlargest(n_top, col)
                    rows.append(dict(
                        date=t, stage=phases["KOSPI"].stage, from_peak=phases["KOSPI"].from_peak,
                        score=label, h=h, top_ret=float(f.loc[top.index].mean()),
                        univ_ret=univ, idx_ret=idx_ret, n=len(r),
                    ))
        if progress:
            progress((k + 1) / len(positions))
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    """국면×점수×기간별 평균 초과수익(상위 − 유니버스), 승률, 시점 수."""
    if bt.empty:
        return bt
    d = bt.assign(excess=bt["top_ret"] - bt["univ_ret"])
    g = d.groupby(["stage", "h", "score"])
    out = g.agg(
        시점수=("excess", "size"),
        평균초과수익=("excess", "mean"),
        중앙값초과수익=("excess", "median"),
        승률=("excess", lambda s: float((s > 0).mean())),
        상위평균수익=("top_ret", "mean"),
        유니버스평균=("univ_ret", "mean"),
        코스피평균=("idx_ret", "mean"),
    ).reset_index()
    return out


def episodes(bt: pd.DataFrame, h: int = 60, score: str = "종합점수(현재 가중치)", gap_days: int = 30) -> pd.DataFrame:
    """조정 국면('하락 중'/'반등 시도'/'반등 확인')을 연속 구간(에피소드)으로 묶어 구간별 성과 표시.

    시점이 겹쳐 보이는 문제를 줄이려고, 구간 안의 모든 시점 결과를 평균내 구간 1개로 센다.
    """
    d = bt[(bt["h"] == h) & (bt["score"] == score) & (bt["stage"] != "조정 없음")].sort_values("date").copy()
    if d.empty:
        return d
    d["ep"] = (d["date"].diff().dt.days.fillna(0) > gap_days).cumsum()
    g = d.groupby("ep")
    out = g.agg(
        시작=("date", "min"), 끝=("date", "max"), 시점수=("date", "size"),
        최대낙폭=("from_peak", "min"),
        상위수익=("top_ret", "mean"), 유니버스=("univ_ret", "mean"), 코스피=("idx_ret", "mean"),
    )
    out["초과수익"] = out["상위수익"] - out["유니버스"]
    return out.reset_index(drop=True)
