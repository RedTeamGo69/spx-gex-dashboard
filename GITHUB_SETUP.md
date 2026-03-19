# SPX GEX Dashboard — GitHub Setup Guide

## Your folder should look like this:

```
GEXQ/
├── streamlit_app.py          ← Streamlit web app
├── requirements.txt          ← Python dependencies
├── main.py                   ← Original desktop launcher (keep it)
├── .gitignore                ← Tells git to skip secrets/junk
├── phase1/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── confidence.py
│   ├── dashboard.py
│   ├── data_client.py
│   ├── expected_move.py
│   ├── gex_engine.py
│   ├── liquidity.py
│   ├── market_clock.py
│   ├── model_inputs.py
│   ├── parity.py
│   ├── quote_filters.py
│   ├── rates.py
│   ├── run_metadata.py
│   ├── scenarios.py
│   ├── staleness.py
│   └── wall_credibility.py
```

## Step-by-step (Windows):

### 1. Install Git (if you don't have it)
Download from: https://git-scm.com/download/win
Install with all defaults.

### 2. Create a new repo on GitHub
- Go to https://github.com/new
- Name it something like "spx-gex-dashboard"
- Pick **Private** (your API keys won't be in the code, but still)
- Do NOT check "Add a README" or .gitignore (we'll make our own)
- Click "Create repository"
- You'll see a page with setup instructions — keep it open

### 3. Open Command Prompt in your project folder
- Open File Explorer
- Navigate to c:\Users\luisq\OneDrive\Desktop\GEXQ
- Click in the address bar, type: cmd
- Press Enter (this opens Command Prompt in that folder)

### 4. Run these commands (one at a time):

```
git init
git add .
git commit -m "Initial commit - SPX GEX Dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/spx-gex-dashboard.git
git push -u origin main
```

Replace YOUR_USERNAME with your GitHub username and the repo name
with whatever you named it in step 2.

GitHub will ask for your credentials — use your GitHub username and
a Personal Access Token (not your password):
  → https://github.com/settings/tokens → Generate new token (classic)
  → Check "repo" scope → Generate → Copy the token and paste it as your password

### 5. Connect to Streamlit Cloud
- Go to https://share.streamlit.io
- Sign in with GitHub
- Click "New app"
- Select your repo, branch "main", file "streamlit_app.py"
- Click "Advanced settings" and paste:

```toml
TRADIER_TOKEN = "your_actual_tradier_token"
FRED_API_KEY = "your_actual_fred_key"
```

- Click Deploy
- Wait 2-3 minutes — you'll get a URL that works on your phone!
