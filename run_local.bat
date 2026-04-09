@echo off
REM ── Fill in your keys below ──
set PUBLIC_SECRET_KEY=YOUR_PUBLIC_SECRET_KEY_HERE
set FRED_API_KEY=YOUR_FRED_API_KEY_HERE

REM ── Desktop GUI version ──
REM python main.py

REM ── Streamlit web version (open http://localhost:8501) ──
streamlit run streamlit_app.py
