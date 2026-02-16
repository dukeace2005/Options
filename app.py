import streamlit as st
import pandas as pd
import time
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import requests
import toml
import json
from urllib.request import urlopen
import certifi
import os
import math
from schwab_login import get_schwab_client
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel
from deps import FinancialDeps
import tools 

class DividendAnalysis(BaseModel):
    ticker: str
    is_sustainable: bool
    explanation: str

def _load_openai_api_key():
    """Resolve OpenAI API key from env, Streamlit secrets, or local secrets.toml."""
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")

    # Streamlit runtime secrets
    try:
        if hasattr(st, "secrets") and st.secrets:
            openai_section = st.secrets.get("openai", {})
            key = openai_section.get("api_key", "")
            if not key:
                key = st.secrets.get("OPENAI_API_KEY", "")
            if key:
                return key
    except Exception:
        pass

    # Local .streamlit/secrets.toml fallback
    possible_paths = [
        ".streamlit/secrets.toml",
        os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    for path in possible_paths:
        if os.path.isfile(path):
            try:
                secrets = toml.load(path)
                openai_section = secrets.get("openai", {})
                key = openai_section.get("api_key", "")
                if not key:
                    key = secrets.get("OPENAI_API_KEY", "")
                if key:
                    return key
            except Exception:
                continue

    return ""

openai_api_key = _load_openai_api_key()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 1. Initialize Agent with the dependency type
openai_provider = OpenAIProvider(api_key=openai_api_key or None)
agent = Agent(
    OpenAIChatModel(OPENAI_MODEL, provider=openai_provider),
    deps_type=FinancialDeps,
    output_type=DividendAnalysis,
)

# 2. Register multiple tools from the other file
agent.tool(tools.fetch_dividend_history)
agent.tool(tools.fetch_payout_ratio)

# ========== 1. INITIAL SETUP & CONFIG ==========
st.set_page_config(
    page_title="AlphaWheel v2", 
    layout="wide",
    page_icon="💎",  # Premium / alpha
    initial_sidebar_state="expanded"
)

# ========== 2. CONFIGURATION LOADER ==========
def load_configuration():
    """Load configuration from config.toml and secrets.toml with proper paths"""
    try:
        # Try multiple possible locations for config.toml
        possible_config_paths = [
            'config.toml',
            './config.toml',
            os.path.join(os.path.dirname(__file__), 'config.toml'),
            os.path.join(os.getcwd(), 'config.toml')
        ]
        
        config_data = None
        config_path = None
        
        for path in possible_config_paths:
            if os.path.exists(path):
                config_path = path
                config_data = toml.load(path)
                break
        
        if config_data is None:
            # Create default config if not found
            config_data = {'screener': {'source': 'yfinance'}}
        
        # Try to load secrets
        try:
            secrets = toml.load('.streamlit/secrets.toml')
        except:
            # Try alternative secrets location
            try:
                secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
                secrets = toml.load(secrets_path)
            except:
                secrets = {}
        
        SCREENER_CONFIG = config_data.get('screener', {})
        DATA_SOURCE = SCREENER_CONFIG.get('source', 'yfinance')
        FMP_API_KEY = secrets.get('fmp', {}).get('api_key', '')


        my_context = FinancialDeps(
            fmp_api_key = FMP_API_KEY,
        )

        # The Agent will automatically call 'fetch_dividend_history' AND 'fetch_payout_ratio'
        # because the prompt requires both pieces of data.
       # result = agent.run_sync(
       #     "Check AAPL's dividend history and payout ratio. Is it sustainable for my risk level?",
       #     deps = my_context
       # )

       # print(f"Analysis for {result.data.ticker}:")
       # print(f"Sustainable: {result.data.is_sustainable}")
       # print(f"Reasoning: {result.data.explanation}")
        
        return DATA_SOURCE, FMP_API_KEY
        
    except Exception as e:
        st.warning(f"Configuration loading issue: {e}. Using default settings.")
        return 

# Load configuration ONCE at startup
DATA_SOURCE, FMP_API_KEY = load_configuration()

# ========== 3. STATE MANAGEMENT ==========
if 'authenticated' not in st.session_state: 
    st.session_state.authenticated = False
if 'last_scan' not in st.session_state: 
    st.session_state.last_scan = pd.DataFrame()
if 'watchlist' not in st.session_state: 
    st.session_state.watchlist = []
if 'portfolio' not in st.session_state: 
    st.session_state.portfolio = {}
if 'scan_history' not in st.session_state: 
    st.session_state.scan_history = []

# ========== 4. CUSTOM CSS ==========
st.markdown("""
    <style>
    .blue-num { 
        color: #003366; 
        font-weight: bold; 
        font-size: 1.1em; 
    }
    .target-box { 
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #6c5ce7;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 5px;
    }
    .positive { color: #00b894; font-weight: bold; }
    .negative { color: #d63031; font-weight: bold; }
    .stProgress > div > div > div > div {
        background-color: #6c5ce7;
    }
    .small-font {
        font-size: 0.8em;
        color: #666;
    }
    .strike-cell {
        background-color: #e8f4fd;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ========== 5. HELPER FUNCTIONS ==========
def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution"""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def calculate_prob_otm(current_price, strike_price, iv, dte, r=0.05, option_type="put"):
    """Calculate Probability OTM using Black-Scholes d2."""
    if current_price <= 0 or strike_price <= 0 or iv <= 0 or dte <= 0:
        return 0.0
    
    t = dte / 365.0
    numerator = math.log(current_price / strike_price) + (r - 0.5 * iv**2) * t
    denominator = iv * math.sqrt(t)
    d2 = numerator / denominator
    
    if option_type == "call":
        # Probability that S_T < K
        return norm_cdf(-d2)
    # put: Probability that S_T > K
    return norm_cdf(d2)

def calculate_rsi(data, window=14):
    """Calculate RSI from price series using Wilder's smoothing"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_stock_data(symbols):
    """Fetch real stock data and RSI from Yahoo Finance"""
    try:
        symbols_clean = [s.replace('.', '-') for s in symbols if s]
        if not symbols_clean:
            return {}
        
        tickers = yf.Tickers(" ".join(symbols_clean))
        # Fetch 3 months to ensure enough data for RSI
        data = tickers.history(period="3mo", interval="1d")
        
        if not data.empty and 'Close' in data.columns:
            # Calculate RSI
            rsi_data = calculate_rsi(data['Close'])
            
            # Calculate Volatility (Annualized std dev of log returns)
            log_returns = np.log(data['Close'] / data['Close'].shift(1))
            volatility = log_returns.std() * (252 ** 0.5)
            
            prices = data['Close'].iloc[-1]
            volumes = data['Volume'].iloc[-1] if 'Volume' in data.columns else None
            current_rsi = rsi_data.iloc[-1]
            
            result_dict = {}
            
            # Handle single vs multiple tickers
            if isinstance(prices, pd.Series):
                for yahoo_symbol, price in prices.items():
                    if isinstance(yahoo_symbol, str):
                        original_symbol = yahoo_symbol.replace('-', '.')
                        rsi_val = current_rsi[yahoo_symbol] if yahoo_symbol in current_rsi else 50.0
                        iv_val = volatility[yahoo_symbol] if isinstance(volatility, pd.Series) and yahoo_symbol in volatility else 0.3
                        vol_val = volumes[yahoo_symbol] if isinstance(volumes, pd.Series) and yahoo_symbol in volumes else None
                        result_dict[original_symbol] = {
                            'price': float(price),
                            'rsi': float(rsi_val),
                            'iv': float(iv_val),
                            'volume': int(vol_val) if vol_val is not None and not pd.isna(vol_val) else None
                        }
            else:
                # Single ticker case
                if symbols_clean:
                    sym = symbols_clean[0].replace('-', '.')
                    iv_val = float(volatility) if not isinstance(volatility, pd.Series) else float(volatility.iloc[0])
                    vol_val = None
                    if isinstance(volumes, pd.Series) and not volumes.empty:
                        vol_val = volumes.iloc[0]
                    elif volumes is not None:
                        vol_val = volumes
                    result_dict[sym] = {
                        'price': float(prices),
                        'rsi': float(current_rsi),
                        'iv': iv_val,
                        'volume': int(vol_val) if vol_val is not None and not pd.isna(vol_val) else None
                    }
                    
            return result_dict
    except Exception as e:
        st.error(f"Data fetch error: {str(e)[:100]}")
    return {}

@st.cache_data(ttl=3600)
def fetch_option_snapshot(symbol, target_dte=30):
    """Fetch option data (put) from Yahoo Finance near target DTE and strike."""
    try:
        yf_symbol = symbol.replace('.', '-')
        ticker = yf.Ticker(yf_symbol)
        expirations = ticker.options
        if not expirations:
            return None
        
        today = datetime.now().date()
        exp_dates = []
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                exp_dates.append((exp, dte))
            except Exception:
                continue
        
        if not exp_dates:
            return None
        
        exp_dates.sort(key=lambda x: (x[1] < 0, abs(x[1] - target_dte)))
        chosen_exp, chosen_dte = exp_dates[0]
        
        chain = ticker.option_chain(chosen_exp)
        puts = chain.puts
        if puts is None or puts.empty:
            return None
        
        return {
            'expiration': chosen_exp,
            'dte': max(chosen_dte, 0),
            'puts': puts
        }
    except Exception:
        return None

def calculate_premium(current_price, strike_price, iv=0.3, dte=30, r=0.05):
    """Calculate estimated option premium using simplified Black-Scholes"""
    if current_price <= 0 or strike_price <= 0:
        return 0
    
    # Simplified calculation
    moneyness = strike_price / current_price
    time_years = dte / 365
    
    # Base premium calculation
    if moneyness < 0.9:  # Deep ITM
        intrinsic = max(strike_price - current_price, 0)
        premium = intrinsic + (current_price * 0.02)
    elif moneyness > 1.1:  # Deep OTM
        premium = current_price * 0.01
    else:  # Near the money
        distance_pct = abs(current_price - strike_price) / current_price
        premium = current_price * iv * (time_years ** 0.5) * (1 + distance_pct)
    
    return round(max(premium, 0.05), 2)

def assess_option_liquidity(option_row):
    """Return liquidity label and score based on volume, OI, and bid-ask spread."""
    try:
        volume = option_row.get('volume', np.nan)
        oi = option_row.get('openInterest', np.nan)
        bid = option_row.get('bid', np.nan)
        ask = option_row.get('ask', np.nan)
        
        volume = float(volume) if pd.notna(volume) else 0.0
        oi = float(oi) if pd.notna(oi) else 0.0
        bid = float(bid) if pd.notna(bid) else np.nan
        ask = float(ask) if pd.notna(ask) else np.nan
        
        spread_pct = np.nan
        if pd.notna(bid) and pd.notna(ask) and ask > 0 and bid >= 0:
            mid = (bid + ask) / 2.0
            if mid > 0:
                spread_pct = (ask - bid) / mid
        
        score = 0
        if volume >= 100:
            score += 40
        elif volume >= 20:
            score += 20
        
        if oi >= 500:
            score += 40
        elif oi >= 100:
            score += 20
        
        if pd.notna(spread_pct):
            if spread_pct <= 0.10:
                score += 20
            elif spread_pct <= 0.20:
                score += 10
        
        if score >= 80:
            label = "High"
        elif score >= 40:
            label = "Med"
        else:
            label = "Low"
        
        return label, int(score)
    except Exception:
        return "Low", 0

def get_company_name(symbol):
    """Get company name for symbol"""
    company_names = {
        'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet', 
        'AMZN': 'Amazon.com', 'TSLA': 'Tesla Inc.', 'NVDA': 'NVIDIA', 
        'META': 'Meta Platforms', 'JPM': 'JPMorgan Chase', 'JNJ': 'Johnson & Johnson', 
        'V': 'Visa Inc.', 'WMT': 'Walmart', 'XOM': 'Exxon Mobil', 
        'BAC': 'Bank of America', 'MA': 'Mastercard',
        'DIS': 'Disney', 'NFLX': 'Netflix', 'PYPL': 'PayPal', 'INTC': 'Intel',
        'CSCO': 'Cisco', 'CMCSA': 'Comcast', 'PEP': 'PepsiCo', 'ADBE': 'Adobe',
        'CRM': 'Salesforce', 'ABT': 'Abbott Labs', 'TMO': 'Thermo Fisher',
        'ACN': 'Accenture', 'NKE': 'Nike', 'PM': 'Philip Morris',
        'LIN': 'Linde', 'T': 'AT&T', 'HD': 'Home Depot', 'MRK': 'Merck',
        'ABBV': 'AbbVie', 'ORCL': 'Oracle', 'AVGO': 'Broadcom', 'COST': 'Costco',
        'MDT': 'Medtronic', 'DHR': 'Danaher', 'UNH': 'UnitedHealth', 'LLY': 'Eli Lilly',
        'BMY': 'Bristol Myers', 'PFE': 'Pfizer', 'CVX': 'Chevron', 'KO': 'Coca-Cola',
        'PGR': 'Progressive', 'MCD': "McDonald's", 'NEE': 'NextEra Energy', 'TXN': 'Texas Instruments',
        'HON': 'Honeywell', 'UPS': 'UPS'
    }
    return company_names.get(symbol, symbol)

def get_sp500_symbols(return_dict=False):
    """Get S&P 500 symbols with fallback"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            table = tables[0]
            if return_dict:
                return {str(s).strip(): str(n).strip() for s, n in zip(table['Symbol'], table['Security'])}
            return [str(s).strip() for s in table['Symbol'].tolist()]
        else:
            if return_dict:
                fallback = get_sp500_fallback_list()
                return {s: get_company_name(s) for s in fallback}
            return get_sp500_fallback_list()
            
    except Exception as e:
        if return_dict:
            fallback = get_sp500_fallback_list()
            return {s: get_company_name(s) for s in fallback}
        return get_sp500_fallback_list()

def get_sp500_fallback_list():
    """Fallback S&P 500 list"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'V',
        'WMT', 'XOM', 'BAC', 'MA', 'DIS', 'NFLX', 'PYPL', 'INTC', 'CSCO', 'CMCSA',
        'PEP', 'ADBE', 'CRM', 'ABT', 'TMO', 'ACN', 'NKE', 'PM', 'LIN', 'T',
        'HD', 'MRK', 'ABBV', 'ORCL', 'AVGO', 'COST', 'MDT', 'DHR', 'UNH', 'LLY',
        'BMY', 'PFE', 'CVX', 'KO', 'PGR', 'MCD', 'NEE', 'TXN', 'HON', 'UPS',
        'AMD', 'QCOM', 'TGT', 'LOW', 'CAT', 'DE', 'BA', 'RTX', 'GS', 'BLK',
        'AMGN', 'GILD', 'REGN', 'ISRG', 'VRTX', 'ILMN', 'BKNG', 'MAR', 'SBUX'
    ]

# ========== 6. DATA SOURCE SCREENING FUNCTIONS ==========
def get_fmp_jsonparsed_data(url):
    """Fetch and parse JSON data from FMP API"""
    try:
        response = urlopen(url, cafile=certifi.where())
        data = response.read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        st.error(f"FMP API Error: {e}")
        return []

def screen_with_fmp(price_range, api_key, **kwargs):
    """Screen stocks using Financial Modeling Prep API"""
    base_url = "https://financialmodelingprep.com/stable/company-screener"
    
    # Build parameters
    params = {
        'apikey': api_key,
        'priceMoreThan': price_range[0],
        'priceLessThan': price_range[1],
        'limit': 500,
        'isActivelyTrading': 'true',
        'isEtf': 'false',
        'isFund': 'false'
    }
    
    # Add any additional parameters
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value
    
    # Build URL
    param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    url = f"{base_url}?{param_string}"
    
    try:
        stocks_data = get_fmp_jsonparsed_data(url)
        
        if not stocks_data:
            return pd.DataFrame()
        
        # Transform to consistent format
        screened_stocks = []
        for stock in stocks_data:
            screened_stocks.append({
                'symbol': stock.get('symbol', ''),
                'companyName': stock.get('companyName', ''),
                'price': stock.get('price', 0),
                'marketCap': stock.get('marketCap', 0),
                'sector': stock.get('sector', ''),
                'industry': stock.get('industry', ''),
                'beta': stock.get('beta', 0),
                'volume': stock.get('volume', 0),
                'exchange': stock.get('exchangeShortName', ''),
                'country': stock.get('country', ''),
                'isActivelyTrading': stock.get('isActivelyTrading', False)
            })
        
        df = pd.DataFrame(screened_stocks)
        
        # Convert marketCap to readable format
        if 'marketCap' in df.columns:
            df['marketCap'] = df['marketCap'].apply(lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.0f}M")
        
        return df
        
    except Exception as e:
        st.error(f"Error in FMP screening: {e}")
        return pd.DataFrame()

def screen_with_yfinance(price_range, **kwargs):
    """Screen stocks using yfinance"""
    sp500_symbols = get_sp500_symbols()
    
    # Fetch prices in batches
    all_prices = {}
    batch_size = 100
    
    for i in range(0, len(sp500_symbols), batch_size):
        batch = sp500_symbols[i:i+batch_size]
        price_data = fetch_stock_data(batch)
        if price_data:
            all_prices.update(price_data)
        
        # Small delay
        if i % 200 == 0:
            time.sleep(0.1)
    
    # Filter by price range
    screened_stocks = []
    for symbol, data in all_prices.items():
        price = data['price']
        rsi = data['rsi']
        if price_range[0] <= price <= price_range[1]:
            screened_stocks.append({
                'symbol': symbol,
                'companyName': get_company_name(symbol),
                'price': price,
                'rsi': rsi,
                'marketCap': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
                'beta': 0,
                'volume': 0,
                'exchange': 'N/A',
                'country': 'US',
                'isActivelyTrading': True
            })
    
    return pd.DataFrame(screened_stocks)

def get_screened_stocks(price_range, **additional_filters):
    """Main function to get screened stocks based on config.toml"""
    if DATA_SOURCE == 'yfinance':
        return screen_with_yfinance(price_range, **additional_filters)
    
    elif DATA_SOURCE == 'fmp':
        if not FMP_API_KEY:
            st.error("❌ FMP API key not found. Falling back to yfinance.")
            return screen_with_yfinance(price_range, **additional_filters)
        
        st.info(f"📊 Using Financial Modeling Prep data source")
        
        # Add any FMP-specific filters
        fmp_filters = {}
        
        if 'min_volume' in additional_filters:
            fmp_filters['volumeMoreThan'] = additional_filters['min_volume']
        
        if 'sectors' in additional_filters and additional_filters['sectors']:
            fmp_filters['sector'] = ','.join(additional_filters['sectors'])
        
        return screen_with_fmp(price_range, FMP_API_KEY, **fmp_filters)
    
    else:
        st.error(f"❌ Unknown data source: {DATA_SOURCE}")
        return pd.DataFrame()

# ========== 7. CSP CANDIDATE FUNCTIONS ==========
def process_stock_candidate(symbol, current_price, company_name, price_range, discount_value, 
                           discount_mode, prob_ownership, min_premium, dte, candidates_list, rsi=None):
    """Process a single stock candidate"""
    if price_range[0] <= current_price <= price_range[1]:
        # Calculate INDIVIDUAL strike price
        if discount_mode == "Fixed Amount ($)":
            strike_price = max(round(current_price - discount_value, 2), 0.01)
            discount_amount = discount_value
            discount_pct = (discount_value / current_price) * 100 if current_price > 0 else 0
        else:
            discount_decimal = discount_value / 100
            strike_price = round(current_price * (1 - discount_decimal), 2)
            discount_amount = round(current_price - strike_price, 2)
            discount_pct = discount_value
        
        if strike_price <= 0 or strike_price > current_price:
            return
        
        # Calculate premium
        est_premium = calculate_premium(current_price, strike_price, dte=dte)
        
        if est_premium < min_premium:
            return
        
        # Calculate metrics
        premium_yield = (est_premium / strike_price) * 100 if strike_price > 0 else 0
        annualized_return = premium_yield * (365 / dte)
        cushion_pct = ((current_price - strike_price) / current_price) * 100 if current_price > 0 else 0
        probability_adjusted_return = annualized_return * (prob_ownership)
        
        # Add to candidates list
        candidates_list.append({
            'Symbol': symbol,
            'Company': company_name,
            'Current': round(current_price, 2),
            'RSI': round(rsi, 1) if rsi is not None else 'N/A',
            'Strike': round(strike_price, 2),
            'Discount $': round(discount_amount, 2),
            'Discount %': round(discount_pct, 2),
            'Premium': round(est_premium, 2),
            'Premium %': round(premium_yield, 2),
            'Annual Return %': round(annualized_return, 2),
            'Adj. Return %': round(probability_adjusted_return, 2),
            'Prob.': round(prob_ownership * 100, 1),
            'Cushion %': round(cushion_pct, 1),
            'DTE': dte
        })

def create_csp_candidates(price_range, discount_value, discount_mode, prob_ownership, min_premium=0.5, dte=30):
    """Generate CSP candidates using configured data source"""
    
    candidates = []
    
    # Get screened stocks
    screened_stocks = get_screened_stocks(price_range)
    
    if screened_stocks.empty:
        st.warning(f"No stocks found in price range ${price_range[0]} - ${price_range[1]}")
        return pd.DataFrame()
    
    st.success(f"Found {len(screened_stocks)} stocks in price range")
    
    # Process each stock
    for _, row in screened_stocks.iterrows():
        symbol = row['symbol']
        current_price = row['price']
        company_name = row['companyName']
        rsi = row.get('rsi', 50)
        
        process_stock_candidate(symbol, current_price, company_name, price_range, discount_value, 
                              discount_mode, prob_ownership, min_premium, dte, candidates, rsi)
    
    if candidates:
        df = pd.DataFrame(candidates)
        return df.sort_values('Annual Return %', ascending=False)
    
    return pd.DataFrame()

# ========== 8. SIDEBAR ==========
with st.sidebar:
    st.title("💎 AlphaWheel v2.0")
    
    menu_options = {
        "Portfolio": "💰",
        "Stock Hunter (CSP)": "🔍",
        "Watchlist": "👀", 
        "Market Dashboard": "📊"
    }

    menu_selection = st.selectbox(
        "Dashboard",  # Changed from "Action Plan"
        list(menu_options.keys()),
        format_func=lambda x: f"{menu_options[x]} {x}"
    )    
    st.divider()
    
    # Brokerage Connection
    st.subheader("🌐 Brokerage Connection")
    if not st.session_state.authenticated:
        broker = st.selectbox("Select Broker", ["None", "Charles Schwab", "E*Trade", "Fidelity", "Merrill"])
        if st.button("🔗 Connect", use_container_width=True, type="primary"):
            with st.spinner("Connecting..."):
                if broker == "None":
                    st.warning("Please select a brokerage to connect.")
                elif broker == "Charles Schwab":
                    client = get_schwab_client()
                    if client:
                        st.session_state.authenticated = True
                        st.session_state.broker = broker
                        st.rerun()
                    else:
                        st.error("Authentication failed. Did you cancel the login or deny access?")
                else:
                    time.sleep(0.5)
                    st.session_state.authenticated = True
                    st.session_state.broker = broker
                    st.rerun()
    else:
        st.success(f"✅ {st.session_state.get('broker', 'None')}")
        if st.button("Disconnect", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    st.divider()
    
    # CSP Strategy Settings (only when CSP plan is selected)
    if menu_selection == "Stock Hunter (CSP)":
        run_scan = st.button(
            "🚀 Run CSP Scan", 
            type="primary", 
            use_container_width=True,
            help="Find CSP opportunities with individual strike prices"
        )
        
        with st.expander("Scan Criteria", expanded=True):
            price_range = st.slider("Target Price Range ($)", 10, 500, (50, 200), step=5)
            
            discount_mode = st.radio(
                "Discount Method:", 
                ["Fixed Amount ($)", "Percentage (%)"], 
                horizontal=True
            )
            
            if discount_mode == "Fixed Amount ($)":
                discount_value = st.slider(
                    "Entry Discount ($)", 
                    1.0, 50.0, 10.0, 0.5
                )
                st.caption(f"Example: $100 stock → Strike at ${100 - discount_value:.2f} (${discount_value} discount)")
            else:
                discount_value = st.slider(
                    "Entry Discount (%)", 
                    1.0, 30.0, 10.0, 0.5
                )
                st.caption(f"Example: $100 stock → Strike at ${100 * (1 - discount_value/100):.2f} ({discount_value}% discount)")
            
            prob_ownership = st.slider("Target Probability", 0.10, 0.50, 0.30, 0.01)
            dte = st.selectbox("Days to Expiry", [7, 14, 30, 45, 60], index=2)
            min_premium = st.number_input("Min. Premium ($)", 0.10, 20.0, 0.50, 0.10)
        
        # Advanced filters (mainly for FMP)
        with st.expander("Advanced Filters", expanded=False):
            min_volume = st.number_input("Minimum Volume", min_value=0, value=100000, step=10000)
            
            sectors = st.multiselect(
                "Filter by Sector",
                ["Technology", "Healthcare", "Financial Services", "Consumer Cyclical", 
                 "Industrials", "Energy", "Utilities", "Real Estate", "Basic Materials", 
                 "Communication Services", "Consumer Defensive"],
                default=[]
            )
            
            st.caption("Note: Advanced filters work best with FMP data source")
    else:
        # Defaults when CSP is not selected (main content won't use these)
        run_scan = False
        price_range = (50, 200)
        discount_mode = "Fixed Amount ($)"
        discount_value = 10.0
        prob_ownership = 0.30
        dte = 30
        min_premium = 0.50
        min_volume = 100000
        sectors = []

# ========== 9. MAIN CONTENT ==========
if menu_selection == "Stock Hunter (CSP)":
    st.title("🔍 Cash Secured Puts Scanner")
    
    # Strategy Summary
    st.markdown(f"""
    <div class="target-box">
        <h3>📊 Current Strategy</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-left: 10px;">
            <div>
                <strong>🎯 Entry Parameters:</strong><br>
                <div style="margin-left: 15px;">
                    • <b>Price Range:</b> ${price_range[0]:.0f} - ${price_range[1]:.0f}<br>
                    • <b>Discount Method:</b> {discount_mode}<br>
                    • <b>Entry Discount:</b> {f'${discount_value:.1f}' if discount_mode == 'Fixed Amount ($)' else f'{discount_value:.1f}%'}<br>
                    • <b>Probability Target:</b> {prob_ownership*100:.0f}% (Delta: {prob_ownership:.2f})<br>
                </div>
            </div>
            <div>
                <strong>⚙️ Trade Settings:</strong><br>
                <div style="margin-left: 15px;">
                    • <b>Days to Expiry:</b> {dte} days<br>
                    • <b>Minimum Premium:</b> ${min_premium:.2f}<br>
                    • <b>Data Source:</b> {DATA_SOURCE.upper()}<br>
                    • <b>Calculation:</b> Individual strikes per stock<br>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Run Scan
    if run_scan:
        with st.spinner(f"Scanning stocks..."):
            # Pass advanced filters
            additional_filters = {
                'min_volume': min_volume,
                'sectors': sectors
            }
            
            # Get candidates using configured data source
            candidates_df = create_csp_candidates(
                price_range, discount_value, discount_mode, 
                prob_ownership, min_premium, dte
            )
            
            if not candidates_df.empty:
                st.session_state.last_scan = candidates_df
                st.toast(f"✅ Found {len(candidates_df)} opportunities!")
            else:
                st.warning("No stocks match your criteria. Try widening price range or reducing minimum premium.")
                st.session_state.last_scan = pd.DataFrame()
    
    # Display Results
    if not st.session_state.last_scan.empty:
        st.subheader(f"📈 Results: {len(st.session_state.last_scan)} CSP Opportunities")
        
        # Metrics Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_strike = st.session_state.last_scan['Strike'].mean()
            st.metric("Avg. Strike Price", f"${avg_strike:,.2f}")
        with col2:
            avg_premium = st.session_state.last_scan['Premium'].mean()
            st.metric("Avg. Premium", f"${avg_premium:,.2f}")
        with col3:
            avg_return = st.session_state.last_scan['Annual Return %'].mean()
            st.metric("Avg. Annual Return", f"{avg_return:.1f}%")
        with col4:
            avg_cushion = st.session_state.last_scan['Cushion %'].mean()
            st.metric("Avg. Safety Cushion", f"{avg_cushion:.1f}%")
        
        # Display Table with INDIVIDUAL STRIKES
        df_display = st.session_state.last_scan.copy()
        
        # Format display
        currency_cols = ['Current', 'Strike', 'Discount $', 'Premium']
        for col in currency_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"${float(x):,.2f}" if isinstance(x, (int, float, np.number)) else str(x))
        
        # Format strike column
        if 'Strike' in df_display.columns:
            df_display['Strike'] = df_display['Strike'].apply(
                lambda x: f"<span class='strike-cell'>${float(str(x).replace('$', '').replace(',', '')):,.2f}</span>" 
                if str(x).replace('$', '').replace(',', '').replace('.', '').isdigit() 
                else x
            )
        
        # Format percentage columns
        pct_cols = ['Discount %', 'Premium %', 'Annual Return %', 'Adj. Return %', 'Prob.', 'Cushion %']
        for col in pct_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{float(x):.1f}%" if isinstance(x, (int, float, np.number)) else str(x)
                )
        
        # Display as HTML
        st.markdown("""
        <style>
        .strike-cell {
            background-color: #e8f4fd;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 3px;
            display: inline-block;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th {
            background-color: #f0f2f6;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #eee;
            vertical-align: middle;
        }
        tr:hover {
            background-color: #f9f9f9;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Select columns to display
        display_columns = ['Symbol', 'Company', 'Current', 'RSI', 'Strike', 'Discount $', 'Discount %', 
                          'Premium', 'Annual Return %', 'Prob.', 'Cushion %']
        
        available_columns = [col for col in display_columns if col in df_display.columns]
        
        # Convert to HTML
        html_table = df_display[available_columns].to_html(escape=False, index=False)
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Export Options
        col1, col2 = st.columns(2)
        with col1:
            csv = st.session_state.last_scan.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"csp_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("⭐ Add All to Watchlist"):
                symbols = st.session_state.last_scan['Symbol'].tolist()
                added = 0
                for symbol in symbols:
                    if symbol not in st.session_state.watchlist:
                        st.session_state.watchlist.append(symbol)
                        added += 1
                st.success(f"Added {added} symbols to watchlist")
                st.rerun()
        
        # Visualization
        if len(st.session_state.last_scan) > 1:
            st.subheader("📊 Opportunity Distribution")
            chart_data = st.session_state.last_scan[['Symbol', 'Annual Return %']].set_index('Symbol')
            st.bar_chart(chart_data, height=300)
        
    else:
        # Fixed the f-string issue here
        sectors_text = f"Enabled ({len(sectors)} sectors)" if sectors else "Disabled"
        st.info(f"""
        ### 👈 Configure and run your CSP scan
        
        **How it works:**
        1. Screens stocks using **{DATA_SOURCE.upper()}** data source
        2. Each stock gets its **own individual strike price** based on your discount method
        3. Calculates premium and returns for each stock
        4. Filters by minimum premium requirement
        
        **Current Configuration:**
        - **Data Source:** {DATA_SOURCE.upper()}
        - **Price Range:** ${price_range[0]} - ${price_range[1]}
        - **Advanced Filters:** {sectors_text}
        """)

elif menu_selection == "Watchlist":
    st.title("👀 Watchlist Manager")
    st.caption(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    watchlist_columns = [
        "Symbol", "Price", "Strike", "Exp (Days)", "Vol", "Prob OTM", "Premium",
        "Return", "Cash Req", "Liquidity", "Liq Score", "Contracts", "Total Cash", "Total Premium"
    ]
    
    # Add Symbols + Free Cash
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.subheader("Your Watchlist")
    with col2:
        # Autocomplete using S&P 500 list
        sp500_dict = get_sp500_symbols(return_dict=True)
        extra_tickers = {"F": "Ford Motor"}
        
        def add_symbol_from_selectbox():
            symbol = st.session_state.get("new_symbol", "").upper().strip()
            if symbol:
                if symbol not in st.session_state.watchlist:
                    st.session_state.watchlist.append(symbol)
                st.session_state.new_symbol = ""
        
        if sp500_dict:
            sp500_dict.update({k: v for k, v in extra_tickers.items() if k not in sp500_dict})
            options = [""] + sorted(sp500_dict.keys(), key=lambda s: (len(s), s))
            new_symbol = st.selectbox(
                "Add Symbol",
                options=options,
                format_func=lambda x: f"{x} - {sp500_dict[x]}" if x else "",
                key="new_symbol",
                on_change=add_symbol_from_selectbox
            )
        else:
            new_symbol = st.selectbox("Add Symbol", options=[""], key="new_symbol")
    with col3:
        max_cash = int(st.session_state.get('free_cash', 50000))
        cash_options = list(range(5000, max(max_cash, 5000) + 1, 5000))
        if cash_options[-1] != max_cash:
            cash_options.append(max_cash)
        
        default_cash = st.session_state.get('free_cash_selected', cash_options[0])
        if default_cash not in cash_options:
            default_cash = cash_options[0]
        
        selected_cash = st.selectbox(
            "Free Cash",
            options=cash_options,
            index=cash_options.index(default_cash),
            help="Used to check if 1 CSP contract (100 shares) is affordable"
        )
        st.session_state.free_cash_selected = selected_cash
    
    # Display Watchlist
    if st.session_state.watchlist:
        selected_cash = st.session_state.get('free_cash_selected', 5000)
        
        price_data = fetch_stock_data(st.session_state.watchlist)
        
        watchlist_items = []
        for symbol in st.session_state.watchlist:
            if symbol in price_data:
                data = price_data[symbol]
                price = data['price']
                iv = data.get('iv', 0.30)
                stock_volume = data.get('volume')
                
                # Option data from Yahoo Finance (closest to 30 DTE, 5% OTM put)
                dte = 30
                strike_price = round(price * 0.95, 1) # fallback
                premium = calculate_premium(price, strike_price, iv, dte)
                option_volume = None
                liquidity_label = "Low"
                liquidity_score = 0
                
                option_snapshot = fetch_option_snapshot(symbol, target_dte=30)
                if option_snapshot:
                    puts = option_snapshot['puts']
                    try:
                        dte = int(option_snapshot['dte'])
                        puts = puts.copy()
                        if 'impliedVolatility' in puts.columns:
                            puts = puts[puts['impliedVolatility'].notna()]
                            puts = puts[puts['impliedVolatility'] > 0]
                        
                        if puts.empty:
                            raise ValueError("No valid IV in option chain")
                        
                        puts['prob_otm'] = puts.apply(
                            lambda r: calculate_prob_otm(
                                price,
                                float(r['strike']),
                                float(r['impliedVolatility']),
                                dte,
                                option_type="put"
                            ),
                            axis=1
                        )
                        
                        # Pick strike with prob_otm >= 0.80 closest to 0.80
                        eligible = puts[puts['prob_otm'] >= 0.80]
                        if not eligible.empty:
                            idx = (eligible['prob_otm'] - 0.80).abs().idxmin()
                            row = eligible.loc[idx]
                        else:
                            # Fallback: highest probability OTM
                            row = puts.loc[puts['prob_otm'].idxmax()]
                        
                        strike_price = float(row['strike'])
                        liquidity_label, liquidity_score = assess_option_liquidity(row)
                        
                        iv_val = row.get('impliedVolatility')
                        if pd.notna(iv_val) and iv_val > 0:
                            iv = float(iv_val)
                        
                        bid = row.get('bid', np.nan)
                        ask = row.get('ask', np.nan)
                        last_price = row.get('lastPrice', np.nan)
                        if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                            premium = float((bid + ask) / 2.0)
                        elif pd.notna(last_price) and last_price > 0:
                            premium = float(last_price)
                        
                        opt_vol = row.get('volume')
                        if pd.notna(opt_vol):
                            option_volume = int(opt_vol)
                    except Exception:
                        pass
                
                # Calculate Metrics
                prob_otm = calculate_prob_otm(price, strike_price, iv, dte)
                volume = option_volume if option_volume is not None else stock_volume
                
                watchlist_items.append({
                    'Symbol': symbol,
                    'Price': price,
                    'Strike': strike_price,
                    'Exp (Days)': dte,
                    'Vol': volume if volume is not None else 0,
                    'Prob OTM': prob_otm * 100,
                    'Premium': premium,
                    'Return': (premium / strike_price) * (365/dte) * 100,
                    'Cash Req': strike_price * 100,
                    'Liquidity': liquidity_label,
                    'Liq Score': liquidity_score
                })
        
        if watchlist_items:
            df_watch = pd.DataFrame(watchlist_items)
            
            df_watch["Contracts"] = (selected_cash / df_watch["Cash Req"]).floordiv(1).astype(int)
            df_watch["Total Cash"] = df_watch["Cash Req"] * df_watch["Contracts"]
            df_watch["Total Premium"] = df_watch["Premium"] * df_watch["Contracts"] * 100
            
            def cash_style(row):
                if row["Contracts"] < 1:
                    return ["background-color: #efefef; color: #888;"] * len(row)
                return [""] * len(row)
            
            styled = (
                df_watch.style
                .apply(cash_style, axis=1)
                .format({
                    "Price": "${:,.2f}",
                    "Strike": "${:,.2f}",
                    "Exp (Days)": "{:.0f}",
                    "Vol": "{:,.0f}",
                    "Prob OTM": "{:.1f}%",
                    "Premium": "${:,.2f}",
                    "Return": "{:.1f}%",
                    "Cash Req": "${:,.0f}",
                    "Liq Score": "{:,.0f}",
                    "Contracts": "{:,.0f}",
                    "Total Cash": "${:,.0f}",
                    "Total Premium": "${:,.2f}"
                })
            )
            
            st.dataframe(
                styled,
                hide_index=True,
                width='stretch'
            )
            
            # Remove functionality
            to_remove = st.multiselect("Select symbols to remove:", st.session_state.watchlist)
            if st.button("Remove Selected"):
                for s in to_remove:
                    if s in st.session_state.watchlist:
                        st.session_state.watchlist.remove(s)
                st.rerun()
        else:
            st.info("No price data available")
            st.dataframe(pd.DataFrame(columns=watchlist_columns), hide_index=True, width='stretch')
    else:
        st.info("Watchlist is empty. Add symbols above.")
        st.dataframe(pd.DataFrame(columns=watchlist_columns), hide_index=True, width='stretch')

elif menu_selection == "Portfolio":
    st.title("💰 Portfolio Manager")
    
    # Check mode
    if not st.session_state.authenticated:
        st.warning("⚠️ Please connect to your brokerage to view positions.")
        portfolio_source = {}
    else:
        # Real Portfolio (from session state or API)
        if not st.session_state.portfolio:
            st.info("No positions found in connected account.")
        portfolio_source = st.session_state.portfolio

    if portfolio_source:
        # Fetch real-time data for the symbols in portfolio
        symbols = list(portfolio_source.keys())
        with st.spinner("Fetching market data..."):
            market_data = fetch_stock_data(symbols)
        
        portfolio_rows = []
        total_equity = 0.0
        total_cost_basis = 0.0
        
        for symbol, position in portfolio_source.items():
            shares = position['shares']
            avg_cost = position['cost']
            
            # Get current price (fallback to cost if data fetch fails)
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                rsi = market_data[symbol]['rsi']
            else:
                current_price = avg_cost
                rsi = 50.0
            
            market_value = shares * current_price
            cost_basis = shares * avg_cost
            pnl_open = market_value - cost_basis
            pnl_pct = (pnl_open / cost_basis) * 100 if cost_basis != 0 else 0
            
            total_equity += market_value
            total_cost_basis += cost_basis
            
            portfolio_rows.append({
                'Symbol': symbol,
                'Qty': shares,
                'Trade Price': avg_cost,
                'Mark': current_price,
                'Mkt Value': market_value,
                'P/L Open': pnl_open,
                'P/L %': pnl_pct,
                'RSI': rsi
            })
            
        # Summary Metrics
        total_pnl = total_equity - total_cost_basis
        total_pnl_pct = (total_pnl / total_cost_basis) * 100 if total_cost_basis != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Net Liq Value", f"${total_equity:,.2f}")
        col2.metric("P/L Open", f"${total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%")
        col3.metric("Buying Power", "N/A")
        col4.metric("Day P/L", "N/A")
        
        st.divider()
        
        # Display Table
        if portfolio_rows:
            df_view = pd.DataFrame(portfolio_rows)
            
            # Formatting
            st.dataframe(
                df_view,
                column_config={
                    "Symbol": "Symbol",
                    "Qty": st.column_config.NumberColumn("Qty", format="%d"),
                    "Trade Price": st.column_config.NumberColumn("Trade Price", format="$%.2f"),
                    "Mark": st.column_config.NumberColumn("Mark", format="$%.2f"),
                    "Mkt Value": st.column_config.NumberColumn("Mkt Value", format="$%.2f"),
                    "P/L Open": st.column_config.NumberColumn("P/L Open", format="$%.2f"),
                    "P/L %": st.column_config.NumberColumn("P/L %", format="%.2f%%"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                },
                hide_index=True,
                use_container_width=True
            )

elif menu_selection == "Market Dashboard":
    st.title("📊 Market Dashboard")
    
    # Market Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("S&P 500", "4,850.43", "+0.75%")
    with col2:
        st.metric("NASDAQ", "15,450.32", "+1.2%")
    with col3:
        st.metric("VIX", "15.20", "-2.5%")
    
    # Top Movers
    st.subheader("Today's Movers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gainers**")
        st.markdown("• TSLA: +5.2% ($215.50)")
        st.markdown("• NVDA: +3.8% ($625.75)")
        st.markdown("• AMD: +2.9% ($175.30)")
    with col2:
        st.markdown("**Losers**")
        st.markdown("• INTC: -2.5% ($45.20)")
        st.markdown("• DIS: -1.8% ($92.75)")
        st.markdown("• BA: -1.2% ($210.40)")

# ========== 10. FOOTER ==========
st.divider()
current_year = datetime.now().year
st.caption(f"© {current_year} AlphaWheel Financial Technologies. All rights reserved. | Data Source: {DATA_SOURCE.upper()} | This tool is for educational purposes only.")


