# H&E Model Recommendation Tool

A Streamlit app for exploring model performance across H&E image types and dataset sizes.

## Requirements

Python 3.10+

## Setup

```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

The app will open automatically, or navigate to `http://localhost:8501` in your browser.

## Project Structure

- `app.py` — Streamlit application entry point
- `precomputed.py` — Recommendation logic and data loading
- `ShinyAppData-*/` — Model results, curves, and confusion matrices
- `iqsim_assets/` — Image quality simulator assets
- `requirements-streamlit.txt` — Python dependencies