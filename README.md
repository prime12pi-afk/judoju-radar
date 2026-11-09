# 주도주 레이더

코스피·코스닥 종목 중 **다음 상승장의 주도주 후보**를 점수로 순위 매기는 Streamlit 앱입니다.
RS 등급, 하락장 방어력, RS 라인 선행 신고가, 저점 이후 반등, 거래량, 외국인·기관 수급을 합산하고,
과거 조정장 기준의 백테스트 탭으로 점수 방식을 검증해 볼 수 있습니다.

> 과거 통계상의 경향을 보여 주는 도구이며 투자 권유가 아닙니다. 투자 판단과 책임은 본인에게 있습니다.

## 실행 (로컬)
```
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 배포 설정
- Repository: 이 저장소 / Branch: `main` / Main file path: `app.py`
- Advanced settings → **Python version: 3.12**
- 첫 접속은 종목 일봉·수급을 모두 받느라 1~3분 걸립니다. 서버가 재시작되면 캐시가 사라져 다시 받습니다.

## 데이터 출처 (모두 비공식·무료 경로)
- 종목 목록/일봉: KRX(FinanceDataReader), 접속이 막히면 네이버 증권
- 지수: Yahoo Finance(yfinance), 보조로 FinanceDataReader
- 외국인·기관 수급: 네이버 증권 (최근 60거래일)
- 외부 사이트 사정으로 예고 없이 막히거나 바뀔 수 있습니다.
