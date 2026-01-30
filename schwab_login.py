from schwab.auth import easy_client
import os

# Your Schwab Developer Credentials
API_KEY = 'YOUR_APP_KEY'
APP_SECRET = 'YOUR_APP_SECRET'
CALLBACK_URL = 'https://127.0.0.1' # Must match exactly what's in the portal
TOKEN_PATH = 'token.json'

def get_schwab_client():
    """
    Handles the 7-day refresh logic automatically.
    If token.json exists, it logs in silently.
    If not, it opens a browser for a manual login.
    """
    try:
        # easy_client is a 'smart' function that checks for the token file first
        client = easy_client(
            api_key=API_KEY,
            app_secret=APP_SECRET,
            callback_url=CALLBACK_URL,
            token_path=TOKEN_PATH
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