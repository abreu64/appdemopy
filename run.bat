@echo off
echo Iniciando Dashboard Streamlit...
echo.
echo Abra seu navegador em: http://localhost:8501
echo.
python -m streamlit run app.py --server.port=8501 --server.address=localhost
