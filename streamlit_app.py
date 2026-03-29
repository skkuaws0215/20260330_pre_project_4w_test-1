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


TASK_LABELS = [
    "[Day1] S3 prefix 합의 results/features_nextflow_team4/ + README 스냅샷",
    "Nextflow 피처 파이프라인 초안 + S3 경로 합의",
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
    st.success(
        "**팀4 S3 합의** — Nextflow 피처 산출: "
        "`s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/`"
    )
    st.markdown(
        "팀4 Pre-Project: **S3 → Nextflow·Batch 피처 → 파일럿/확장 학습 → "
        "QC3 → METABRIC → ADMET(후처리) → Bedrock** 순으로 산출물을 맞춥니다."
    )
    st.info(
        "사이드바 **「전체 요약·현황」** 에 완료/예정 작업과 문서 목록이 정리되어 있습니다. "
        "실행 코드(Nextflow·Batch·학습)는 **내일 이후** 착수 예정."
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
| `nextflow/s3_features_nextflow_team4_README.txt` | S3 prefix 안내 업로드용 초안 |
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
        "S3: `results/` 구조·parquet 스냅샷 README 기록, prefix `results/features_nextflow_team4/` 합의",
        "`pipeline_overview.html`: Nextflow·Batch·METABRIC·ADMET·Bedrock·§18~19·용어 ref",
        "`README.md` / `pipeline_sections_detail.md` / 모델·ADMET 전략 MD",
        "Streamlit 대시보드 (용어·모델·ADMET·플로우·체크리스트·HTML·AWS)",
        "`requirements.txt` · `run_dashboard.ps1` · S3 README 템플릿",
    ]
    nxt = [
        "Nextflow `main.nf` / `nextflow.config` (S3 in → `features_nextflow_team4/` out)",
        "AWS Batch 컴퓨트 환경·ECR·Nextflow `awsbatch` 연결",
        "파일럿 학습 (예: LightGBM) 및 모델 랭킹 확장",
        "`25_admet_results.parquet` 컬럼 ↔ 컷오프 매핑 및 Pandas/Athena 필터 스크립트",
        "METABRIC 외부 검증 · Bedrock RAG",
        "선택: S3에 `README.txt` 업로드 (`nextflow/README.md` 참고)",
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
| Raw | `s3://.../` 에서 `results/` **바깥** 접두사 |
| 전처리 완료 | `s3://.../results/<소스>/` + `final_qc.csv` (스냅샷은 README) |
| **Nextflow FE 산출 (팀4)** | `s3://.../results/features_nextflow_team4/` |

**목록 확인**

```powershell
. .\\use-team-aws.ps1
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/"
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/" --recursive | findstr /i ".parquet"
```

**자격 증명은 대시보드에 넣지 마세요.** `.aws/credentials` 로컬만 사용.
        """
    )


def page_flow():
    st.header("§2 플로우 요약 (체크용)")
    steps = [
        "11 소스 → QC1",
        "Snakemake 전처리 → QC2",
        "integrate.py → ML-ready",
        "Nextflow + Batch 피처",
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
