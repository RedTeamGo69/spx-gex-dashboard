import os

from phase1.app import run_app

# Keys are read from environment variables only.
# Set them before running:
#   set PUBLIC_SECRET_KEY=your_key_here
#   set FRED_API_KEY=your_key_here
PUBLIC_SECRET_KEY = os.environ.get("PUBLIC_SECRET_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
DEBUG = False


def main():
    run_app(
        public_secret_key=PUBLIC_SECRET_KEY,
        fred_api_key=FRED_API_KEY,
        debug=DEBUG,
        tool_version="v5",
    )


if __name__ == "__main__":
    main()
