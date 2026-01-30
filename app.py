import streamlit as st
import pandas as pd
import time

# 1. INITIAL SETUP
st.set_page_config(page_title="AlphaWheel v2", layout="wide")

# State Management
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'last_scan' not in st.session_state: st.session_state.last_scan = pd.DataFrame()

st.markdown("""
    <style>
    .blue-num { color: #003366; font-weight: bold; font-size: 1.1em; }
    .target-box { 
        background-color: #f0f2f6; padding: 20px; 
        border-radius: 10px; border-left: 5px solid #6c5ce7;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR (All Controls Restored) ---
with st.sidebar:
    st.title("🛡️ AlphaWheel")
    menu_selection = st.selectbox("Action Plan", ["Stock Hunter (CSP)", "Watchlist Manager", "Portfolio Manager"])
    
    st.divider()
    
    # Brokerage Connection (Stateful)
    st.subheader("🌐 Brokerage Connection")
    if not st.session_state.authenticated:
        if st.button("🔗 Link Brokerage Account", use_container_width=True):
            st.session_state.authenticated = True
            st.rerun()
    else:
        st.success("✅ Account Linked")
        if st.button("Unlink Account", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    st.divider()
    
    # Strategy Parameters
    st.subheader("🎯 Strategy Settings")
    
    # Price Range Control
    price_range = st.slider("Stock Price Range ($)", 0, 500, (20, 100), step=5)
    min_p, max_p = price_range
    
    # Probability/Delta Control
    prob_ownership = st.slider("Probability of Ownership (Delta)", 0.10, 0.50, 0.30)
    
    # Discount Controls
    discount_mode = st.radio("Discount Type:", ["Dollar ($)", "Percentage (%)"], horizontal=True)
    if discount_mode == "Dollar ($)":
        min_disc = st.slider("Min. Price Discount ($)", 1.0, 100.0, 10.0)
        calc_strike = max_p - min_disc
    else:
        min_disc_pct = st.slider("Min. Price Discount (%)", 1, 30, 10)
        calc_strike = max_p * (1 - (min_disc_pct / 100))
    
    st.info(f"**Target Strike:** ${calc_strike:,.2f}")
    
    # Dummy Scan Button
    run_scan = st.button("🚀 Run Market Scan", type="primary", use_container_width=True)

# --- 3. MAIN SCREEN ---
if menu_selection == "Stock Hunter (CSP)":
    st.title("🔍 Stock Hunter")
    
    # Strategy Summary Card
    st.markdown(f"""
    <div class="target-box">
        <h3>Current Strategy Summary</h3>
        You are hunting for assets priced between <b>${min_p}</b> and <b>${max_p}</b>. 
        Your goal is to secure entry at <span class="blue-num">${calc_strike:,.2f}</span> 
        with a <b>{int(prob_ownership*100)}%</b> probability of assignment.
    </div>
    """, unsafe_allow_html=True)

    

    # Simulation Logic (Dummy Scan)
    if run_scan:
        with st.spinner("Simulating Market Scan..."):
            time.sleep(1) # Visual feedback
            # Create dummy data that respects your sliders
            dummy_data = {
                'Symbol': ['AAPL', 'MSFT', 'TSLA', 'AMD', 'NVDA'],
                'Company': ['Apple Inc.', 'Microsoft Corp.', 'Tesla Inc.', 'AMD', 'Nvidia'],
                'Current Price': [max_p - 5, min_p + 10, (max_p + min_p)/2, max_p - 2, min_p + 5],
                'Sector': ['Technology', 'Technology', 'Consumer Discretionary', 'Technology', 'Technology']
            }
            st.session_state.last_scan = pd.DataFrame(dummy_data)
            st.toast("Scan Complete (Simulated Mode)")

    # Display Results
    if not st.session_state.last_scan.empty:
        st.subheader("Market Results")
        
        # Searchable/Sortable Table
        st.dataframe(
            st.session_state.last_scan, 
            use_container_width=True, 
            hide_index=True,
            column_config={"Current Price": st.column_config.NumberColumn(format="$%.2f")}
        )
    else:
        st.info("Adjust your parameters in the sidebar and click 'Run Market Scan' to populate results.")

elif menu_selection == "Watchlist Manager":
    st.title("👀 Watchlist Manager")
    st.caption("Reporting Date: January 30, 2026")
    st.info("Watchlist tracking will be integrated once brokerage data is live.")