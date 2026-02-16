from pydantic_ai import RunContext
import httpx
from pydantic_ai import RunContext
from deps import FinancialDeps
import httpx

async def fetch_dividend_history(ctx: RunContext[FinancialDeps], ticker: str) -> str:
    api_key = ctx.deps.fmp_api_key
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"apikey": api_key})
        resp.raise_for_status()
        return resp.text

async def fetch_payout_ratio(ctx: RunContext[FinancialDeps], ticker: str) -> str:
    """Fetches the dividend payout ratio and net income metrics."""
    key = ctx.deps.fmp_api_key
    url = f"https://financialmodelingprep.com/api/v3/key-metrics-ticker/{ticker}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"apikey": key})
        return resp.text