@echo off
echo PDF Reader アプリケーションを起動しています...
echo.
echo 注意: 初回実行前に env.example を .env にコピーして、
echo OpenAI API キーを設定してください。
echo.

uv run streamlit run pdf_uploader.py
