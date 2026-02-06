from schwab.auth import easy_client
import os

try:
    import toml
except ImportError:
    toml = None

TOKEN_PATH = 'token.json'


def _load_schwab_secrets():
    """Load Schwab credentials from .streamlit/secrets.toml or Streamlit st.secrets."""
    # Prefer Streamlit secrets when running inside Streamlit (e.g. production)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            s = st.secrets.get("schwab", {})
            return {
                "api_key": s.get("app_key", ""),
                "app_secret": s.get("app_secret", ""),
                "callback_url": s.get("redirect_uri", "https://127.0.0.1"),
            }
    except Exception:
        pass

    # Standalone: load from .streamlit/secrets.toml
    if toml is None:
        raise RuntimeError("Install 'toml' to load secrets when not running in Streamlit.")
    possible_paths = [
        ".streamlit/secrets.toml",
        os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    for path in possible_paths:
        if os.path.isfile(path):
            secrets = toml.load(path)
            s = secrets.get("schwab", {})
            return {
                "api_key": s.get("app_key", ""),
                "app_secret": s.get("app_secret", ""),
                "callback_url": s.get("redirect_uri", "https://127.0.0.1"),
            }
    raise FileNotFoundError(
        "No .streamlit/secrets.toml found. Add a [schwab] section with app_key, app_secret, redirect_uri."
    )


def get_schwab_client():
    """
    Handles the 7-day refresh logic automatically.
    If token.json exists, it logs in silently.
    If not, it opens a browser for a manual login.
    Credentials are read from .streamlit/secrets.toml [schwab] or Streamlit secrets.
    """
    try:
        creds = _load_schwab_secrets()
        client = easy_client(
            api_key=creds["api_key"],
            app_secret=creds["app_secret"],
            callback_url=creds["callback_url"],
            token_path=TOKEN_PATH,
        )
        return client
    except Exception as e:
        print(f"Error authenticating: {e}")
        return None

if __name__ == "__main__":
    print("Attempting to connect to Schwab...")
    schwab = get_schwab_client()
    
    if schwab:
        # Test the connection by getting your account info
        resp = schwab.get_accounts()
        if resp.status_code == 200:
            print("Successfully connected! Token saved to token.json.")
            print("Your accounts:", resp.json())
        else:
            print(f"Connection failed. Status code: {resp.status_code}")