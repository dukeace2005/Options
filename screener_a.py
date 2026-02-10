import pandas as pd
import time
from schwab.client import Client
from schwab_login import get_schwab_client

def get_sp500_tickers():
    """Scrapes the current S&P 500 list from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    df = table[0]
    # Standardize symbols (replaces '.' with '-' for Schwab compatibility)
    return [s.replace('.', '-') for s in df['Symbol'].tolist()]

def screen_schwab_csp(min_price=10, max_price=300):
    """
    Main screener function. 
    Usage: screen_schwab_csp(10, 300)
    """
    client = get_schwab_client()
    if not client:
        print("Authentication failed. Check your configuration.")
        return pd.DataFrame()
        
    all_tickers = get_sp500_tickers()
    
    print(f"Total S&P 500 Tickers: {len(all_tickers)}")
    
    # 1. BATCH PRICE FILTERING (Efficiency Layer)
    # We chunk because Schwab limits the number of symbols per quote request
    candidates = []
    chunk_size = 50 
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i + chunk_size]
        resp = client.get_quotes(chunk)
        
        if resp.status_code == 200:
            quotes = resp.json()
            for symbol, data in quotes.items():
                price = data.get('quote', {}).get('lastPrice', 0)
                if min_price <= price <= max_price:
                    candidates.append(symbol)
        
    print(f"Tickers within budget (${min_price}-${max_price}): {len(candidates)}")

    # 2. DETAILED OPTIONS SCAN
    results = []
    for symbol in candidates:
        # Rate limiting: 120 calls per minute
        time.sleep(0.5) 
        
        # Pull option chain for the nearest monthly expiration
        resp = client.get_option_chain(
            symbol, 
            contract_type=Client.Options.ContractType.PUT,
            strike_count=10, 
            days_to_expiration=45 # Approx. March 20, 2026
        )
        
        if resp.status_code != 200: continue
        
        data = resp.json()
        put_map = data.get('putExpDateMap', {})
        
        for date, strikes in put_map.items():
            for strike_price, contract_list in strikes.items():
                contract = contract_list[0]
                
                # Schwab API provides 'probabilityITM'
                # Prob OTM = 100 - Prob ITM
                prob_otm = 100 - contract.get('probabilityITM', 100)
                
                # LAYER 1: Prob OTM Filter
                if prob_otm >= 75 and prob_otm <= 85:
                    results.append({
                        'Ticker': symbol,
                        'Price': data['underlying']['last'],
                        'Strike': strike_price,
                        'Prob OTM': f"{prob_otm:.1f}%",
                        'Premium': contract['bid'],
                        'Exp Date': date.split(':')[0],
                        'OI': contract['openInterest']
                    })
                    break # Only take the best-fit strike per ticker

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example: Running for S&P 500 stocks between $50 and $200
    df_results = screen_schwab_csp(50, 200)
    
    if not df_results.empty:
        print("\n--- Top High-Probability CSP Candidates ---")
        print(df_results.sort_values(by='Premium', ascending=False).head(20))
    else:
        print("No matches found for the given criteria.")