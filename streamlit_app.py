"""
로컬에서 파이프라인 문서·체크리스트를 보며 진행할 때 사용하는 Streamlit 대시보드.
실행: python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent


def read_text(rel: str) -> str | None:
    p = ROOT / rel
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8")


def show_html_report():
    html_path = ROOT / "pipeline_overview.html"
    if not html_path.is_file():
        st.warning("pipeline_overview.html 이 없습니다.")
        return
    raw = html_path.read_text(encoding="utf-8")
    st.caption("아래는 보고서 전체 HTML을 임베드한 것입니다. 스크롤하여 §1~§19를 확인하세요.")
    st.components.v1.html(raw, height=960, scrolling=True)


def page_experiment_dashboards():
    st.header("실험 대시보드 (DL / Graph HTML)")
    st.markdown(
        """
GitHub **`blob` 링크**는 HTML을 웹페이지로 그리지 않습니다.  
**다른 폴더**에서 `python3 -m http.server`만 켜도 루트 HTML이 없어 **빈 목록·404**가 납니다.

**이렇게 하세요 (저장소 루트에서):**
        """
    )
    st.code("python3 serve_dashboards.py", language="bash")
    st.info(
        "터미널에 주소가 출력되고, 브라우저가 `index.html` 목차를 자동으로 엽니다. "
        "같은 Wi‑Fi의 다른 사람에게는 출력되는 **LAN IP** 주소를 공유하세요."
    )
    st.subheader("서버를 이미 켰다면 (기본 포트 8765)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("목차 (index)", "http://127.0.0.1:8765/index.html")
    with c2:
        st.link_button("DL 실험", "http://127.0.0.1:8765/dl_experiment_dashboard_20260331.html")
    with c3:
        st.link_button("Graph 실험", "http://127.0.0.1:8765/graph_experiment_dashboard_20260401.html")
    st.link_button("SageMaker 통합 (20260403)", "http://127.0.0.1:8765/sagemaker_experiment_dashboard_20260403.html")
    st.caption(
        "링크가 안 열리면 `serve_dashboards.py`가 **실행 중인지**, 포트가 **8765**인지 확인하세요. "
        "팀 전체 공개 URL은 GitHub Pages(워크플로 `deploy-dashboards.yml`) 설정 후 사용합니다."
    )


TASK_LABELS = [
    "[Day1] 본인 전용 prefix results/features_nextflow_team4/ + README (공유 results/ 와 구분)",
    "Nextflow 피처 파이프라인 초안 (`results/` 입력, `ml_ready/` 비입력) + S3 경로 합의",
    "AWS Batch CPU/GPU 큐·Job 정의·IAM",
    "파일럿 ML/DL/Graph → 전체 확장·모델 랭킹",
    "SageMaker git pull · 최신 코드 반영",
    "top_variance_genes 실험 (500/1K/2K/5K)",
    "E2E 12 모델 + RRF + QC3",
    "METABRIC 외부 코호트 일반화 리포트",
    "ADMET 4 Gate 필터·후보 테이블",
    "Bedrock RAG·임상 문서·모순 탐지",
    "벤치마크 보고서 + Claude/MCP 병행",
]


def page_home():
    st.title("AI Drug Discovery — 진행 대시보드")
    st.subheader("현재 실행 워크플로우 (팀4 기준)")
    st.markdown(
        """
1. 공유 전처리 입력 확인: `results/<소스>/...` (읽기 전용)
2. 팀4 스냅샷 생성: `results/features_nextflow_team4/source_snapshot/...`
3. 브리지 전처리: `prepare_fe_inputs.py` -> `input/<run_id>/{sample_features,drug_features,labels}.parquet`
4. Nextflow + AWS Batch FE 실행: `main.nf -profile awsbatch`
5. FE 산출 확인: `results/features_nextflow_team4/<run_id>/features.parquet`, `labels.parquet`, `feature_manifest.json`
6. 후속 단계: SageMaker 학습/튜닝 -> METABRIC 검증 -> ADMET 필터 -> Bedrock 리포트
        """
    )
    st.caption(
        "핵심: 공유 `results/<소스>/` 는 읽기만 사용하고, 팀4 작업물은 `features_nextflow_team4/` 아래에서만 생성/관리"
    )
    st.success(
        "**본인 전용 FE·ML 입력 prefix** — 업로드·산출: "
        "`s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/` "
        "(공유 `results/<소스>/` 와 혼동 금지)"
    )
    st.markdown(
        "팀4 Pre-Project: **S3 `results/`(raw 기반 전처리) → Nextflow·Batch 피처**(`ml_ready/`는 FE 입력으로 사용하지 않음) "
        "→ 이 폴더 FE 데이터셋으로 ML 테스트·학습 → QC3 → METABRIC → ADMET(후처리) → Bedrock 순으로 산출물을 맞춥니다."
    )
    st.info(
        "사이드바 **「전체 요약·현황」** 에 완료/예정 작업과 문서 목록이 정리되어 있습니다. "
        "**아키텍처:** FE = Nextflow+Batch / 학습·튜닝 = 주로 SageMaker (동일 Docker를 Batch GPU에 올리는 것은 선택)."
    )
    st.subheader("결측 임계값 비교 (팀4 FE v2)")
    st.markdown(
        """
| 실험 run | missing threshold | rows | feature cols | 비고 |
|----------|-------------------|------|--------------|------|
| `20260330_batch_miss70_v2` | 0.7 | 14,497 | 17,922 | 느슨한 결측 기준 |
| `20260330_batch_miss30_v2` | 0.3 | 14,497 | 17,920 | 엄격한 결측 기준 |

차이 컬럼: `drug__smiles`, `drug__canonical_smiles_raw` (30%에서 제거)

S3 위치: `.../results/features_nextflow_team4/fe_re_batch_runs/<run_id>/`
        """
    )
    st.caption("모델 비교는 A(엄격=결측치 30%), B(느슨=결측치 70%), C(SMILES 수치화 추가, 70% 기반) 프레임으로 진행")
    st.subheader("A/B/C 입력셋 경로 (고정)")
    st.markdown(
        """
공통 베이스: `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/abc_inputs/20260330_abc_v1/`

| 순서 | 실험군 | features | labels |
|------|--------|----------|--------|
| 1 | A (결측치 30%, 엄격) | `.../A/features.parquet` | `.../A/labels.parquet` |
| 2 | B (결측치 70%, 느슨) | `.../B/features_b.parquet` | `.../B/labels.parquet` |
| 3 | C (결측치 70%, 느슨+SMILES) | `.../C/features.parquet` | `.../C/labels.parquet` |

인덱스: `.../abc_index.json`
        """
    )
    st.subheader("문서 바로가기 (요약)")
    st.markdown(
        """
| 문서 | 내용 |
|------|------|
| `pipeline_overview.html` | §1–§19 파이프라인 보고서 (브라우저 탭 **파이프라인 보고서**) |
| `README.md` | AWS·S3·워크플로·Streamlit·Batch 의견 |
| `model_selection_strategy.md` | ML/DL/Graph 후보·우선순위 (PPTX) |
| `admet_postprocessing_strategy.md` | ADMET 예측 도구 + 후처리 필터 + 컷오프 참고 |
| `pipeline_sections_detail.md` | HTML §1–2 상세 + 계획 보강 |
| `results/features_nextflow_team4/README.txt` | S3 동일 키 — **본인 전용** FE prefix 안내 (공유 `results/` 와 구분) |
        """
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("보고서", "§1–§19", help="pipeline_overview.html")
    with c2:
        st.metric("팀 버킷", "drug-discovery-joe-raw-data-team4")
    with c3:
        st.metric("실행 코드", "내일~", help="Nextflow·Batch·학습 스크립트")


def page_status():
    st.title("전체 요약·현황")
    st.caption("이 대시보드·저장소에 반영된 범위 기준 (로컬 `.aws/credentials` 는 Git 제외)")

    done = [
        "팀 AWS: 프로젝트 `.aws/` + `use-team-aws.ps1` (개인 프로필과 분리)",
        "S3: **본인 전용** `results/features_nextflow_team4/` — FE·ML용 신규 업로드는 이 prefix만 사용 (공유 `results/<소스>/` 비업로드)",
        "아키텍처: FE Nextflow+Batch / 학습·튜닝 SageMaker (Batch GPU 선택)",
        "`pipeline_overview.html`: Nextflow·Batch·METABRIC·ADMET·Bedrock·§18~19·용어 ref",
        "`README.md` / `pipeline_sections_detail.md` / 모델·ADMET 전략 MD",
        "Streamlit 대시보드 (용어·모델·ADMET·플로우·체크리스트·HTML·AWS)",
        "`requirements.txt` · `run_dashboard.ps1`",
        "브리지 전처리 스크립트 추가: `nextflow/scripts/prepare_fe_inputs.py`",
        "팀4 전용 입력 스냅샷 생성: `results/features_nextflow_team4/source_snapshot/{gdsc,depmap,chembl}/...`",
        "브리지 표준 입력 생성: `results/features_nextflow_team4/input/20260330_bridge_v1/{sample_features,drug_features,labels}.parquet`",
        "결측 임계값 비교 run 완료: `20260330_batch_miss70_v2`, `20260330_batch_miss30_v2`",
        "A/B/C 실험 프로토콜 문서화: `model_selection_strategy.md`",
    ]
    nxt = [
        "A/B/C 벤치마크 시작: A=miss30, B=miss70(문자열 SMILES 제외), C=miss70+SMILES 수치화",
        "LightGBM 파일럿 + 동일 split/seed/fold로 성능 비교 (RMSE/MAE/Spearman)",
        "파일럿 학습 (예: LightGBM) 및 모델 랭킹 확장",
        "`25_admet_results.parquet` 컬럼 ↔ 컷오프 매핑 및 Pandas/Athena 필터 스크립트",
        "METABRIC 외부 검증 · Bedrock RAG",
        "필요 시 새 브리지 run_id(`input/YYYYMMDD_bridge_vN`) 롤링",
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("완료 (문서·설계·Day1)")
        for x in done:
            st.markdown(f"- {x}")
    with c2:
        st.subheader("다음 (실행·내일~)")
        for x in nxt:
            st.markdown(f"- {x}")

    st.divider()
    st.subheader("실행 명령")
    st.code(
        "cd <프로젝트루트>\n"
        "pip install -r requirements.txt\n"
        "python -m streamlit run streamlit_app.py",
        language="text",
    )
    st.subheader("GitHub 푸시 (로컬 커밋 후)")
    st.code(
        "git remote add origin https://github.com/<USER>/<REPO>.git\n"
        "git branch -M main\n"
        "git push -u origin main",
        language="text",
    )


def page_glossary():
    st.header("핵심 용어")
    st.subheader("ADMET: 피처 vs 최종 게이트")
    st.markdown(
        """
- **피처 단계 ADMET**: 예측값을 **컬럼**으로 넣어 모델이 학습·랭킹에 사용.
- **최종 임상 진입 ADMET (4 Gate)**: 숏리스트에 대해 **통과/탈락**을 정하는 정책 단계.
- **두 번 계산**이 필수는 아님. 같은 점수를 앞에서는 입력으로, 뒤에서는 컷으로 **재사용**할 수 있음.
        """
    )
    st.subheader("QC3 (Stage2) vs METABRIC (Stage3)")
    st.markdown(
        """
- **Stage2 ≈ QC3**: TCGA·GDSC 등 **같은 체계** 안에서 CV, 누수, LODO/LOCO 등 **절차 타당성**.
- **Stage3 ≈ METABRIC**: **독립 코호트**에서 **외부 일반화** — 질문이 다름.
- QC3만으로 METABRIC 성능이 **보장되지 않음**.
        """
    )


def page_checklist():
    st.header("Pre-Project 체크리스트")
    for i, label in enumerate(TASK_LABELS):
        st.checkbox(label, key=f"pipeline_task_{i}")
    n = sum(bool(st.session_state.get(f"pipeline_task_{i}", False)) for i in range(len(TASK_LABELS)))
    total = len(TASK_LABELS)
    st.progress(n / total if total else 0.0)
    st.caption(f"{n}/{total} 완료")
    if st.button("체크리스트 초기화"):
        for i in range(len(TASK_LABELS)):
            st.session_state[f"pipeline_task_{i}"] = False
        st.rerun()


def page_readme():
    st.header("README.md")
    text = read_text("README.md")
    if text is None:
        st.error("README.md 없음")
        return
    st.markdown(text)


def page_detail_md():
    st.header("pipeline_sections_detail.md")
    text = read_text("pipeline_sections_detail.md")
    if text is None:
        st.warning("파일이 없습니다.")
        return
    st.markdown(text)


def page_models():
    st.header("model_selection_strategy.md")
    st.caption("PPTX `drug_repurposing_model_selection` 반영 · 마지막 슬라이드(최종 1종 요약)는 벤치 다모델 계획과 별도")
    text = read_text("model_selection_strategy.md")
    if text is None:
        st.warning("파일이 없습니다.")
        return
    st.markdown(text)


def page_admet():
    st.header("admet_postprocessing_strategy.md")
    st.caption("순위 리스트 후처리 — 모델 구조 변경 아님")
    text = read_text("admet_postprocessing_strategy.md")
    if text is None:
        st.warning("파일이 없습니다.")
        return
    st.markdown(text)


def page_aws():
    st.header("AWS · S3 메모")
    st.markdown(
        """
| 항목 | 값 |
|------|-----|
| 버킷 ARN | `arn:aws:s3:::drug-discovery-joe-raw-data-team4` |
| Raw | `s3://.../` 에서 **`results/` 바깥** 상위 접두사 (원본) |
| 공유 전처리 (**읽기만**, 4인 공용) | `s3://.../results/<소스>/…` — **본인 산출 업로드 금지** |
| 통합 테이블 | `ml_ready/` — **FE 입력으로 사용하지 않음** |
| **본인 전용 FE·ML 데이터** | `s3://.../results/features_nextflow_team4/` — **신규 FE·실험 산출 전부 여기** |
| 팀4 입력 스냅샷 | `s3://.../results/features_nextflow_team4/source_snapshot/...` |
| 팀4 브리지 입력 | `s3://.../results/features_nextflow_team4/input/<run_id>/` |

**목록 확인**

```powershell
. .\\use-team-aws.ps1
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/"
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/" --recursive | findstr /i ".parquet"
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/" --recursive
```

**자격 증명은 대시보드에 넣지 마세요.** `.aws/credentials` 로컬만 사용.
        """
    )


def page_flow():
    st.header("§2 플로우 요약 (체크용)")
    st.caption(
        "팀4 Nextflow·Batch 피처: 입력은 **`results/<소스>/`**(raw 기반 전처리); **`ml_ready/`는 사용하지 않음.**"
    )
    steps = [
        "11 소스 → QC1",
        "Snakemake 전처리 → QC2",
        "integrate.py → ML-ready",
        "Nextflow + Batch 피처 (팀4: results/ → features_nextflow_team4/)",
        "파일럿 → 전체 확장 · 모델 랭킹",
        "12 모델 3-Track",
        "RRF 앙상블",
        "QC3 (Stage2)",
        "METABRIC (Stage3)",
        "ADMET 4 Gate",
        "Bedrock + RAG",
        "Shortlist + QC4",
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(f"{i}. {s}")


PAGES = {
    "홈": page_home,
    "실험 대시보드 (DL/Graph HTML)": page_experiment_dashboards,
    "전체 요약·현황": page_status,
    "핵심 용어 (ADMET / QC3·METABRIC)": page_glossary,
    "모델 선별 (PPTX 반영)": page_models,
    "ADMET 후처리 전략": page_admet,
    "§2 플로우 요약": page_flow,
    "Pre-Project 체크리스트": page_checklist,
    "파이프라인 보고서 (HTML)": show_html_report,
    "README.md": page_readme,
    "pipeline_sections_detail.md": page_detail_md,
    "AWS · S3 메모": page_aws,
}


def main():
    st.set_page_config(
        page_title="Drug Discovery Pipeline",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    choice = st.sidebar.selectbox("섹션", list(PAGES.keys()), index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption("실행: `python -m streamlit run streamlit_app.py`")
    PAGES[choice]()


if __name__ == "__main__":
    main()
