"""주도주 레이더 — 폭락 후 반등 시 주도주 후보를 찾는 Streamlit 앱.

실행: python -m streamlit run app.py
"""
import datetime as dt
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest
import data
import metrics

st.set_page_config(page_title="주도주 레이더", page_icon="📡", layout="wide")

UP, DOWN = "#e0443e", "#2f6fdb"  # 한국식: 상승=빨강, 하락=파랑

# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("설정")
    n_kospi = st.slider("코스피 시총 상위 종목 수", 0, 400, 150, 10)
    n_kosdaq = st.slider("코스닥 시총 상위 종목 수", 0, 400, 100, 10)
    lookback = st.slider("시장 고점 탐색 기간(거래일)", 60, 250, 120, 10,
                         help="이 기간 안의 지수 최고점을 '하락 시작점'으로 봅니다.")

    st.subheader("종합점수 가중치")
    st.caption("기본값은 백테스트에서 가장 일관됐던 RS 등급 중심입니다. 방어력·RS 선행은 낮게 둡니다.")
    w_rs = st.slider("RS 등급 (중장기 상대강도)", 0, 100, 50)
    w_def = st.slider("하락장 방어력", 0, 100, 5)
    w_lead = st.slider("RS 라인 선행 신고가", 0, 100, 5)
    w_flow = st.slider("외국인·기관 수급", 0, 100, 15,
                       help="최근 20일 순매수, 지수 하락일 순매수(저가 매수), 외국인 보유비율 변화. "
                            "수급 수집에 실패하면 자동으로 제외됩니다.")
    w_reb = st.slider("저점 이후 반등 상대강도", 0, 100, 15,
                      help="지수가 아직 저점을 갱신 중이면 자동으로 제외됩니다.")
    w_vol = st.slider("상승일 거래량 우위", 0, 100, 10)

    st.subheader("필터")
    min_rs = st.slider("최소 RS 등급", 1, 99, 1)
    only_lead = st.checkbox("RS 라인 선행 신고가 종목만")
    only_ma = st.checkbox("50일선 위 종목만")

    only_flow = st.checkbox("외국인+기관 20일 순매수 종목만")

    refresh = st.button("🔄 오늘 데이터 다시 받기")
    st.caption("주가: FinanceDataReader · 수급: 네이버 증권(비공식, 최근 60거래일) · 하루 단위 캐시")

weights = dict(rs=w_rs, defense=w_def, rebound=w_reb, lead=w_lead, volume=w_vol, flow=w_flow)
if sum(weights.values()) == 0:
    st.warning("가중치가 모두 0입니다. 하나 이상 올려 주세요.")
    st.stop()

# ---------------------------------------------------------------- 데이터 로딩
start = (dt.date.today() - dt.timedelta(days=430)).isoformat()

if refresh:
    for f in data.CACHE_DIR.glob("prices_*.pkl"):
        f.unlink()
    st.session_state.pop("loaded", None)

key = (n_kospi, n_kosdaq, dt.date.today().isoformat())
if st.session_state.get("loaded", {}).get("key") != key:
    bar = st.progress(0.0, text="종목 목록 불러오는 중…")
    stage = "종목 목록"
    try:
        uni = data.load_universe(n_kospi, n_kosdaq)
        if uni.empty:
            bar.empty()
            st.error("종목 수를 1개 이상으로 설정해 주세요.")
            st.stop()
        stage = "종목 일봉"
        prices = data.load_prices(
            uni["Code"].tolist(), start,
            progress=lambda p: bar.progress(p, text=f"일봉 수집 중… {p:.0%} (처음 한 번만 오래 걸립니다)"),
        )
        stage = "지수"
        bar.progress(1.0, text="코스피·코스닥 지수 불러오는 중…")
        indices = {m: data.load_index(m, start) for m in ("KOSPI", "KOSDAQ")}
        stage = "외국인·기관 수급"
        flows = data.load_flows(
            list(prices),
            progress=lambda p: bar.progress(p, text=f"외국인·기관 수급 수집 중… {p:.0%}"),
        )
    except Exception as e:  # 네트워크 차단·무응답 등을 무한 로딩 대신 화면에 표시 (st.stop은 BaseException이라 통과)
        bar.empty()
        st.error(f"**{stage}** 단계에서 데이터를 가져오지 못했습니다.\n\n`{type(e).__name__}: {e}`\n\n"
                 "네트워크(학교·기관망 차단 등)를 확인한 뒤 새로고침(F5)해 주세요.")
        st.stop()
    bar.empty()
    st.session_state["loaded"] = dict(key=key, uni=uni, prices=prices, indices=indices, flows=flows)

L = st.session_state["loaded"]
uni, prices, indices, flows = L["uni"], L["prices"], L["indices"], L["flows"]
phases = {m: metrics.detect_phase(indices[m], lookback) for m in indices}


@st.cache_data(show_spinner="점수 계산 중…")
def run_scores(_uni, _prices, _indices, _phases, _flows, weights_t, sig):
    r = metrics.compute_scores(_uni, _prices, _indices, _phases, dict(weights_t), _flows)
    return r, r.attrs.get("has_flow", False)


res, has_flow = run_scores(uni, prices, indices, phases, flows, tuple(sorted(weights.items())),
                           (key, lookback, len(prices), len(flows)))
if res.empty:
    st.error("계산 가능한 종목이 없습니다. 데이터 수집에 실패했을 수 있습니다.")
    st.stop()

# ---------------------------------------------------------------- 헤더
st.title("📡 주도주 레이더")
st.caption("시장을 꾸준히 이겨 온 종목(RS 등급)을 중심으로 방어력·수급·RS 라인 신호를 더해 다음 상승장의 주도주 후보를 좁힙니다. "
           "과거 통계상의 경향이며 투자 권유가 아닙니다.")
_src = uni.attrs.get("listing_source")
if _src and _src != "KRX":
    st.caption(f"ℹ️ 종목 목록 출처: {_src} (KRX 접속이 막힌 네트워크라 대체 경로를 사용 중입니다. 업종 분류가 KRX와 다를 수 있습니다.)")

cols = st.columns(2)
for col, (m, label) in zip(cols, (("KOSPI", "코스피"), ("KOSDAQ", "코스닥"))):
    p = phases[m]
    with col:
        with st.container(border=True):
            st.markdown(f"**{label}** · 기준일 {p.last_date:%Y-%m-%d}")
            a, b, c = st.columns(3)
            a.metric("현재 (고점 대비)", f"{p.last:,.0f}", f"{p.from_peak:+.1%}")
            b.metric("저점 이후", f"{p.rebound:+.1%}", f"저점 {p.trough_date:%m/%d}", delta_color="off")
            c.metric("국면", p.stage)
            st.caption(f"고점 {p.peak_date:%Y-%m-%d} ({p.peak:,.0f}) · 팔로우스루 데이: {p.ftd_status}")

if all(p.stage == "하락 중" for p in phases.values()):
    st.info("지수가 아직 저점을 갱신 중이라 **'저점 이후 반등' 항목은 계산에서 제외**했습니다. "
            "백테스트(🧪 탭)에서는 조정장 전후로 **RS 등급**이 가장 일관되게 통했고, 방어력은 오히려 반등 때 뒤처졌습니다. "
            "다만 '하락 중' 국면은 표본이 매우 적습니다. 반등이 시작되면 다시 열어 반등 상대강도가 반영된 순위를 확인하세요.")
elif any(p.stage == "반등 시도" for p in phases.values()):
    st.warning("반등 시도 중이지만 팔로우스루 데이가 확인되지 않았습니다. 후보를 '관찰'하는 단계로 보세요.")

# 전체 종목 기준 종합점수 순위(필터를 걸어도 번호가 바뀌지 않도록 필터 전에 계산)
res = res.assign(순위=res["종합점수"].rank(ascending=False, method="min").astype(int))

# ---------------------------------------------------------------- 오늘의 주도종목 TOP 3
st.subheader("🏅 오늘의 주도종목 TOP 3")
st.caption(f"기준일 {max(p.last_date for p in phases.values()):%Y-%m-%d} · 분석 {len(res)}종목 중 종합점수 1~3위 "
           "(사이드바 필터와 무관, 가중치는 반영). 매수 추천이 아니라 점수 기준 후보 순위입니다.")
_medals = ["🥇", "🥈", "🥉"]
for i, (col, (_, r)) in enumerate(zip(st.columns(3), res.head(3).iterrows())):
    with col, st.container(border=True):
        st.markdown(f"### {_medals[i]} {r['Name']}")
        st.caption(f"{r['Market']} · {r['Industry']} · {r['Code']}")
        m1, m2 = st.columns(2)
        m1.metric("종합점수", f"{r['종합점수']:.1f}")
        m2.metric("RS 등급", f"{int(r['RS등급'])}")
        badges = []
        if r["rs_lead"]:
            badges.append("RS 라인 선행 신고가")
        if r["above_50ma"]:
            badges.append("50일선 위")
        if has_flow and pd.notna(r.get("smart20")) and r["smart20"] > 0:
            badges.append("외국인·기관 순매수")
        st.markdown(" ".join(f"`{b}`" for b in badges) if badges else "_뚜렷한 추가 신호 없음_")
        st.caption(f"1개월 {r['ret_1m']:+.1%} · 하락구간 지수 대비 {r['rel_dd']:+.1%}")

if not has_flow:
    st.warning("외국인·기관 수급 데이터를 받지 못해 **수급 항목을 제외**하고 계산했습니다. "
               "네트워크 상태를 확인하고 사이드바의 '오늘 데이터 다시 받기'를 눌러 보세요.")
elif len(flows) < len(prices):
    st.caption(f"수급 데이터 {len(flows)}/{len(prices)}종목 수집 (나머지는 수급 점수를 중앙값으로 처리).")

view = res[res["RS등급"] >= min_rs]
if only_lead:
    view = view[view["rs_lead"]]
if only_ma:
    view = view[view["above_50ma"]]
if only_flow and has_flow:
    view = view[view["smart20"] > 0]

tab1, tab2, tab3, tab_bt, tab4 = st.tabs(
    ["🏆 주도주 후보", "🏭 업종 흐름", "📈 종목 차트", "🧪 백테스트", "📖 지표 설명"])

# ---------------------------------------------------------------- 탭1: 후보
with tab1:
    st.write(f"분석 {len(res)}종목 중 조건 충족 **{len(view)}종목**")
    show = pd.DataFrame({
        "순위": view["순위"],
        "종목": view["Name"], "시장": view["Market"], "업종": view["Industry"],
        "종합점수": view["종합점수"].round(1),
        "RS등급": view["RS등급"].astype(int),
        "방어력(하락일 아웃퍼폼)": view["defense"],
        "하락구간 수익률": view["stock_dd"],
        "지수 대비": view["rel_dd"],
        "RS선행 신고가": view["rs_lead"].map({True: "✅", False: ""}),
        "50일선 위": view["above_50ma"].map({True: "✅", False: ""}),
        "외국인 20일": view.get("frgn20"),
        "기관 20일": view.get("org20"),
        "하락일 순매수": view.get("dip_buy"),
        "외국인 지분 변화(%p)": view.get("hold_chg"),
        "상승/하락일 거래량비": view["vol_ratio"].round(2),
        "1개월": view["ret_1m"],
        "코드": view["Code"],
    })
    if not has_flow:
        show = show.drop(columns=["외국인 20일", "기관 20일", "하락일 순매수", "외국인 지분 변화(%p)"])
    st.dataframe(
        show, hide_index=True, width="stretch", height=620,
        column_config={
            "순위": st.column_config.NumberColumn("순위", format="%d위", width="small",
                help="전체 분석 종목 중 종합점수 순위 (필터를 걸어도 번호는 유지됩니다). 열 제목을 눌러 정렬할 수 있습니다."),
            "종합점수": st.column_config.ProgressColumn("종합점수", min_value=0, max_value=100, format="%.1f"),
            "RS등급": st.column_config.ProgressColumn("RS등급", min_value=1, max_value=99, format="%d"),
            "방어력(하락일 아웃퍼폼)": st.column_config.NumberColumn(format="percent",
                help="지수가 내린 날 중 이 종목이 지수보다 덜 빠졌거나 오른 날의 비율"),
            "하락구간 수익률": st.column_config.NumberColumn(format="percent",
                help="시장 고점일부터 현재까지 종목 수익률"),
            "지수 대비": st.column_config.NumberColumn(format="percent",
                help="하락구간 종목 수익률 − 지수 수익률"),
            "1개월": st.column_config.NumberColumn(format="percent"),
            "외국인 20일": st.column_config.NumberColumn(format="percent",
                help="최근 20거래일 외국인 순매수량 ÷ 같은 기간 거래량"),
            "기관 20일": st.column_config.NumberColumn(format="percent",
                help="최근 20거래일 기관 순매수량 ÷ 같은 기간 거래량"),
            "하락일 순매수": st.column_config.NumberColumn(format="percent",
                help="지수가 내린 날의 (외국인+기관) 순매수량 ÷ 거래량. 양수면 하락을 받아낸 종목"),
            "외국인 지분 변화(%p)": st.column_config.NumberColumn(format="%.2f",
                help="최근 60거래일 외국인 보유비율 변화"),
        },
    )
    st.download_button("CSV 내려받기", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"leaders_{dt.date.today()}.csv", mime="text/csv")

# ---------------------------------------------------------------- 탭2: 업종
with tab2:
    st.write("업종(KSIC 세분류)별 중앙값입니다. 업종 전체가 함께 강한지를 보면 개별 종목 신호의 신뢰도를 가늠할 수 있습니다.")
    min_n = st.slider("업종당 최소 종목 수", 2, 10, 3)
    sec = metrics.sector_table(res, min_n)
    if sec.empty:
        st.info("조건에 맞는 업종이 없습니다. 최소 종목 수를 낮추거나 분석 종목 수를 늘려 보세요.")
    else:
        top = sec.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["종합_중앙값"], y=top.index, orientation="h",
                               marker_color=UP, text=top["종목수"].astype(str) + "종목", textposition="outside"))
        fig.update_layout(height=520, margin=dict(l=0, r=30, t=10, b=0), xaxis_title="종합점수 중앙값")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(sec.round(1), width="stretch")

# ---------------------------------------------------------------- 탭3: 차트
with tab3:
    names = view["Name"].tolist() or res["Name"].tolist()
    pick = st.selectbox("종목 선택 (현재 필터 결과 순서)", names)
    row = res[res["Name"] == pick].iloc[0]
    df = prices[row["Code"]].join(indices[row["Market"]]["Close"].rename("idx"), how="inner")
    p = phases[row["Market"]]
    ma50_full = df["Close"].rolling(50).mean()
    rs_full = df["Close"] / df["idx"]
    df = df.tail(150)  # 수급 60일이 보일 만큼, 지수 고점일이 포함되는 구간
    ma50 = ma50_full.loc[df.index]
    rs_line = rs_full.loc[df.index]
    rs_line = rs_line / rs_line.iloc[0] * 100

    fl = flows.get(row["Code"])
    show_flow = fl is not None and len(fl) > 0
    n_rows = 3 if show_flow else 2
    heights = [0.5, 0.25, 0.25] if show_flow else [0.65, 0.35]
    titles = [f"{pick} 주가", f"RS 라인 (종목 ÷ {'코스피' if row['Market']=='KOSPI' else '코스닥'})"]
    if show_flow:
        titles.append("외국인·기관 일별 순매수량(주)")
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.04,
                        subplot_titles=titles)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 increasing_line_color=UP, decreasing_line_color=DOWN, name="주가"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma50, line=dict(color="#888", width=1.2), name="50일선"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=rs_line, line=dict(color="#8a4fff", width=2), name="RS 라인"), row=2, col=1)
    fig.add_hline(y=rs_line.tail(120).max(), line_dash="dot", line_color="#8a4fff", row=2, col=1)
    if show_flow:
        fig.add_trace(go.Bar(x=fl.index, y=fl["frgn"], name="외국인", marker_color="#e08a1e"), row=3, col=1)
        fig.add_trace(go.Bar(x=fl.index, y=fl["org"], name="기관", marker_color="#1f9d8a"), row=3, col=1)
        fig.update_layout(barmode="group")
    for r in range(1, n_rows + 1):
        fig.add_vline(x=p.peak_date, line_dash="dot", line_color="#999", row=r, col=1)
        fig.add_vline(x=p.trough_date, line_dash="dot", line_color="#2f6fdb", row=r, col=1)
    fig.update_layout(height=780 if show_flow else 640, xaxis_rangeslider_visible=False,
                      showlegend=show_flow, legend=dict(orientation="h", y=-0.04),
                      margin=dict(l=0, r=0, t=40, b=0))
    fig.update_xaxes(range=[df.index[0].to_pydatetime(),
                            (df.index[-1] + pd.Timedelta(days=3)).to_pydatetime()])
    st.plotly_chart(fig, width="stretch")
    st.caption("회색 점선=지수 고점일, 파란 점선=지수 저점일. 보라색 점선은 최근 120일 RS 라인 최고치입니다. "
               "수급은 최근 60거래일만 제공됩니다.")

    a, b, c, d = st.columns(4)
    a.metric("RS 등급", f"{int(row['RS등급'])}")
    b.metric("방어력", f"{row['defense']:.0%}" if pd.notna(row["defense"]) else "-")
    c.metric("하락구간 지수 대비", f"{row['rel_dd']:+.1%}")
    d.metric("RS 선행 신고가", "예" if row["rs_lead"] else "아니오")
    if show_flow and pd.notna(row.get("smart20")):
        e, f_, g, h = st.columns(4)
        e.metric("외국인 20일 순매수/거래량", f"{row['frgn20']:+.1%}")
        f_.metric("기관 20일 순매수/거래량", f"{row['org20']:+.1%}")
        g.metric("하락일 순매수", f"{row['dip_buy']:+.1%}" if pd.notna(row.get("dip_buy")) else "-")
        h.metric("외국인 지분 변화", f"{row['hold_chg']:+.2f}%p")

# ---------------------------------------------------------------- 탭: 백테스트
with tab_bt:
    st.markdown(
        "과거 각 시점에서 **그 날까지의 데이터만**으로 점수를 다시 계산하고, 점수 상위 종목의 **이후 20·60거래일 수익률**을 "
        "같은 시점의 유니버스 평균과 비교합니다. 코스피 국면(하락 중/반등 시도/반등 확인)별로 나눠 "
        "'지금 같은 때 이 점수가 통했는가'를 봅니다. 수급은 과거 데이터가 없어 **제외**됩니다."
    )
    st.warning(
        "**결과를 믿기 전에:** ① 종목군이 *오늘 기준* 시총 상위라 과거에 사라진·부진했던 종목이 빠진 **생존 편향**이 있어 "
        "수익률이 부풀려집니다. ② 큰 조정장은 10년에 몇 번뿐이라 독립 표본이 적고, 시점을 겹쳐 뽑아 표본이 많아 보입니다. "
        "③ 거래비용·유동성은 반영하지 않았습니다. 방향성 참고용이지 성과 보증이 아닙니다."
    )
    if len(uni) < 40:
        st.info("백테스트는 최소 40종목이 필요합니다. 사이드바에서 종목 수를 늘려 주세요.")
    else:
        c1, c2, c3 = st.columns(3)
        n_bt = c1.slider("백테스트 종목 수 (시총 상위)", 30, len(uni), min(100, len(uni)), 10,
                         help="많을수록 오래 걸립니다. 80종목 기준 약 1~2분.")
        step = c2.select_slider("시점 간격 (거래일)", options=[5, 10, 20], value=10)
        top_pct = c3.select_slider("상위 비율 (%)", options=[10, 20, 30], value=20)

        sub = uni.nlargest(n_bt, "Marcap")
        sig = f"{dt.date.today()}|{sorted(sub['Code'])}|{step}|{top_pct}|{lookback}|{sorted(weights.items())}"
        bt_file = data.CACHE_DIR / f"bt_{hashlib.md5(sig.encode()).hexdigest()[:16]}.pkl"
        cached = bt_file.exists()
        if st.button("▶ 백테스트 실행" + (" (저장된 결과 있음)" if cached else "")):
            if cached:
                bt = pd.read_pickle(bt_file)
            else:
                bbar = st.progress(0.0, text="과거 주가(2014년~) 불러오는 중…")
                long_start = "2014-06-01"
                px = data.load_prices(
                    sub["Code"].tolist(), long_start,
                    progress=lambda p: bbar.progress(p, text=f"과거 주가 불러오는 중… {p:.0%}"))
                idx_long = {m: data.load_index(m, long_start) for m in ("KOSPI", "KOSDAQ")}
                bt = backtest.run_backtest(
                    sub, px, idx_long, weights, lookback=lookback, step=step, horizons=(20, 60),
                    top_frac=top_pct / 100,
                    progress=lambda p: bbar.progress(p, text=f"시점별 점수 재계산 중… {p:.0%}"))
                bbar.empty()
                if not bt.empty:
                    bt.to_pickle(bt_file)
            st.session_state["bt"] = (bt, n_bt, step, top_pct)

        if "bt" in st.session_state:
            bt, b_n, b_step, b_top = st.session_state["bt"]
            if bt.empty:
                st.error("계산 가능한 시점이 없습니다. 종목 수를 늘리거나 데이터 수집을 확인해 주세요.")
            else:
                h = st.radio("이후 수익률 기간 (거래일)", [20, 60], index=1, horizontal=True)
                summ = backtest.summarize(bt)
                sh = summ[summ["h"] == h]
                stages = [s for s in ("하락 중", "반등 시도", "반등 확인", "조정 없음") if s in set(sh["stage"])]
                cur = phases["KOSPI"].stage
                st.info(f"현재 코스피 국면은 **{cur}** 입니다. 아래에서 같은 이름의 국면 열을 보세요. "
                        f"(종목 {b_n}개 · 시점 간격 {b_step}일 · 상위 {b_top}% 기준, 시점 {bt['date'].nunique()}개)")

                fig = go.Figure()
                palette = {"하락 중": "#2f6fdb", "반등 시도": "#e0a21e", "반등 확인": "#e0443e", "조정 없음": "#9aa0a6"}
                order = list(backtest.SCORE_COLS.values())
                for stg in stages:
                    d = sh[sh["stage"] == stg].set_index("score").reindex(order)
                    fig.add_trace(go.Bar(name=stg, x=order, y=d["평균초과수익"] * 100, marker_color=palette[stg]))
                fig.add_hline(y=0, line_color="#666")
                fig.update_layout(barmode="group", height=420, yaxis_title="평균 초과수익 (%p, 상위 − 유니버스)",
                                  margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, width="stretch")

                def _cell(r):
                    return f"{r['평균초과수익']*100:+.1f}%p · 승률 {r['승률']:.0%} · n={int(r['시점수'])}"
                piv = (sh.assign(v=sh.apply(_cell, axis=1))
                         .pivot(index="score", columns="stage", values="v")
                         .reindex(index=order, columns=stages).fillna("-"))
                st.caption("셀 = 평균 초과수익 · 승률(상위가 유니버스를 이긴 시점 비율) · 시점 수")
                st.dataframe(piv, width="stretch")

                st.subheader("조정 구간별 성과 (종합점수, 구간당 1표본)")
                ep = backtest.episodes(bt, h)
                if ep.empty:
                    st.write("조정 구간이 없습니다.")
                else:
                    epv = pd.DataFrame({
                        "시작": ep["시작"].dt.strftime("%Y-%m-%d"), "끝": ep["끝"].dt.strftime("%Y-%m-%d"),
                        "시점수": ep["시점수"], "최대낙폭": ep["최대낙폭"],
                        "상위 종목": ep["상위수익"], "유니버스": ep["유니버스"], "코스피": ep["코스피"],
                        "초과수익": ep["초과수익"],
                    })
                    st.dataframe(epv, hide_index=True, width="stretch", column_config={
                        c: st.column_config.NumberColumn(format="percent")
                        for c in ("최대낙폭", "상위 종목", "유니버스", "코스피", "초과수익")})
                    wins = int((ep["초과수익"] > 0).sum())
                    st.caption(f"독립 조정 구간 {len(ep)}개 중 {wins}개에서 상위 종목이 유니버스를 이겼습니다. "
                               "구간 수가 적어 통계적 확신을 주기엔 부족합니다.")

# ---------------------------------------------------------------- 탭4: 설명
with tab4:
    st.markdown("""
### 이 앱이 보는 것
| 지표 | 계산 | 왜 보나 |
|---|---|---|
| **RS 등급** | 최근 3·6·9·12개월 수익률을 4:2:2:2로 가중해 지수 대비 초과분을 구하고 분석 종목 중 1~99 백분위로 환산 | 중장기로 시장을 이기던 종목인지 |
| **방어력** | 지수 고점일 이후 지수가 내린 날 중 종목이 지수보다 덜 빠졌거나 오른 날의 비율 | 하락장 버티는 힘, 큰손의 매집 가능성 |
| **RS 라인 선행** | (종목÷지수)가 120일 신고가의 1% 이내인데 주가는 고점 대비 5% 이상 아래 | 주가보다 상대강도가 먼저 신고가를 내는 초기 주도주 패턴 |
| **저점 이후 반등** | 지수 저점일 이후 종목 수익률 − 지수 수익률 | 반등 첫 구간에서 앞서 나가는지 (저점 확정 전에는 제외) |
| **외국인·기관 수급** | ① 최근 20일 (외국인+기관) 순매수량 ÷ 거래량 35% ② 지수 하락일의 순매수 비중 35% ③ 60일 순매수 비중 15% ④ 외국인 보유비율 변화 15% 를 각각 백분위로 환산해 합산 | 폭락을 받아낸 주체가 있는지. 특히 ②는 지수가 빠질 때 사 모은 종목을 가려냅니다 |
| **거래량 우위** | 최근 20일 상승일 거래량 ÷ 하락일 거래량 | 오를 때 돈이 몰리는지 |

### 시장 국면 판정
- 최근 N거래일 내 지수 고점 → 그 뒤 최저점 → 현재로 구분하며, 고점 대비 -7% 이상 빠졌을 때만 '조정'으로 봅니다.
- **팔로우스루 데이**: 저점 이후 첫 상승 마감을 1일차로 세고, 4일차 이후 지수가 +1.5% 이상 오르며 전일보다 거래량이 늘면 반등 확인 신호로 표시합니다. 저점을 다시 깨면 카운트가 초기화됩니다.

### 한계
- 데이터는 FinanceDataReader 일봉이며 수정주가 반영 여부와 지연은 보장되지 않습니다.
- 시총 상위 N개만 분석합니다. 소형주 주도주는 N을 키워야 잡힙니다.
- **수급 데이터는 네이버 증권의 비공식 엔드포인트**라 예고 없이 바뀌거나 막힐 수 있고, 최근 60거래일만 제공됩니다. 수집에 실패하면 수급 항목은 자동 제외됩니다.
- 순매수량은 주 단위라 대형주/소형주를 비교하려고 **거래량 대비 비율**로 정규화했습니다. 기관은 '기관 합계'이며 연기금·투신 등 세부 구분은 없습니다.
- **백테스트 탭**은 가격 기반 지표만 검증합니다(수급은 과거 데이터 없음). 생존 편향·적은 독립 표본·거래비용 미반영 때문에 방향성 참고용입니다.
- 실적·테마 뉴스는 아직 반영하지 않았습니다.
- 모든 지표는 과거 통계상의 경향이며 미래 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.
""")
