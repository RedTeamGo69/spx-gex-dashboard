@echo off
REM ── Fill in your keys below ──
set TRADIER_TOKEN=6GChMbZONQUUr6vK5A08sDJbPGls
set FRED_API_KEY=80c20243b2fbb9590f64c7183ae38f53

REM ── Desktop GUI version ──
REM python main.py

REM ── Streamlit web version (open http://localhost:8501) ──
streamlit run streamlit_app.py
