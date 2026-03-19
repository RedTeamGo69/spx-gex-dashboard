import os

from phase1.app import run_app

# Keys are read from environment variables only.
# Set them before running:
#   set TRADIER_TOKEN=your_token_here
#   set FRED_API_KEY=your_key_here
TRADIER_TOKEN = os.environ.get("TRADIER_TOKEN", "")
TRADIER_BASE_URL = "https://api.tradier.com/v1"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
DEBUG = False


def main():
    run_app(
        tradier_token=TRADIER_TOKEN,
        fred_api_key=FRED_API_KEY,
        tradier_base_url=TRADIER_BASE_URL,
        debug=DEBUG,
        tool_version="v5",
    )


if __name__ == "__main__":
    main()
