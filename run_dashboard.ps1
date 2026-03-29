# Streamlit 대시보드 (PATH에 streamlit 없을 때도 동작)
Set-Location $PSScriptRoot
python -m pip install -r requirements.txt -q
python -m streamlit run streamlit_app.py
