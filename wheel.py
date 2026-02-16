from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Tuple, Dict, Any
from datetime import datetime, date
from enum import Enum
import math

class SignalType(str, Enum):
    CLOSE = "CLOSE"
    ROLL = "ROLL"
    KEEP = "KEEP"
    WATCH = "WATCH"

class ShortPutPosition(BaseModel):
    """Pydantic model for a short put position with all required data"""
    symbol: str
    expiry: date
    strike: float
    quantity: int = Field(default=-1, description="Negative for short positions")
    entry_price: float = Field(gt=0, description="Premium received at entry per share")
    
    # Current market data (fetched from FMP)
    stock_price: float = Field(gt=0)
    option_ask: float = Field(gt=0)
    days_to_expiry: int = Field(ge=0)
    delta: float = Field(ge=-1, le=0)
    theta: float = Field(le=0)  # Negative for short puts
    gamma: float = Field(ge=0)
    vega: float
    iv_rank: float = Field(ge=0, le=100)
    implied_volatility: float = Field(ge=0)
    dividend_date: Optional[date] = None
    
    # Computed fields
    intrinsic_value: float = 0
    extrinsic_value: float = 0
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('extrinsic_value', always=True)
    def calc_extrinsic(cls, v, values):
        if 'option_ask' in values and 'intrinsic_value' in values:
            return max(0, values['option_ask'] - values['intrinsic_value'])
        return v
    
    @validator('intrinsic_value', always=True)
    def calc_intrinsic(cls, v, values):
        if 'strike' in values and 'stock_price' in values:
            return max(0, values['strike'] - values['stock_price'])
        return v
    
    def profit_pct(self) -> float:
        """Profit percentage based on premium received"""
        return (self.entry_price - self.option_ask) / self.entry_price
    
    def profit_amount(self) -> float:
        """Actual P/L in dollars (per share)"""
        return (self.entry_price - self.option_ask) * abs(self.quantity) * 100
    
    def roc(self) -> float:
        """Return on Capital (based on strike * 100 as capital requirement)"""
        capital_at_risk = self.strike * 100 * abs(self.quantity)
        return (self.profit_amount() / capital_at_risk) if capital_at_risk > 0 else 0
    
    def total_credit_received(self) -> float:
        """Total credit received at entry"""
        return self.entry_price * 100 * abs(self.quantity)
    
    def current_liability(self) -> float:
        """Current cost to close position"""
        return self.option_ask * 100 * abs(self.quantity)

class SignalResult(BaseModel):
    position_symbol: str
    signal: SignalType
    reason: str
    current_pl_amount: float = Field(description="Current P/L in dollars")
    current_pl_percent: float = Field(description="Current P/L as % of premium")
    projected_pl_after_action: Optional[float] = Field(None, description="Projected P/L if action taken")
    recommended_action: Optional[str] = None
    limit_price: Optional[float] = None
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"

class ShortPutAnalyzer:
    """Main analyzer class for short put positions"""
    
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = thresholds or {
            'profit_target_pct': 0.50,  # 50% of premium
            'roc_target': 0.02,  # 2% return on capital
            'loss_limit_pct': 2.0,  # 200% loss (relative to credit)
            'dte_roll_min': 14,
            'dte_roll_max': 60,
            'delta_limit': 0.30,
            'iv_rank_min_roll': 30,
            'extrinsic_min': 0.10,
        }
    
    def analyze(self, position: ShortPutPosition) -> SignalResult:
        """Main analysis function returning signal with P/L"""
        
        # Calculate current P/L
        current_pl_amount = position.profit_amount()
        current_pl_percent = position.profit_pct() * 100
        
        # 1. CHECK CLOSE SIGNALS
        close_signal = self._check_close_signals(position)
        if close_signal:
            return SignalResult(
                position_symbol=position.symbol,
                signal=SignalType.CLOSE,
                reason=close_signal,
                current_pl_amount=current_pl_amount,
                current_pl_percent=current_pl_percent,
                projected_pl_after_action=current_pl_amount,  # Realized P/L
                confidence="HIGH"
            )
        
        # 2. CHECK ROLL SIGNALS
        roll_signal, roll_details = self._check_roll_signals(position)
        if roll_signal:
            projected_pl = self._calculate_roll_pl(position, roll_details)
            return SignalResult(
                position_symbol=position.symbol,
                signal=SignalType.ROLL,
                reason=roll_signal,
                current_pl_amount=current_pl_amount,
                current_pl_percent=current_pl_percent,
                projected_pl_after_action=projected_pl,
                recommended_action=roll_details.get('action', ''),
                limit_price=roll_details.get('limit_price'),
                confidence=roll_details.get('confidence', 'MEDIUM')
            )
        
        # 3. KEEP SIGNAL
        return SignalResult(
            position_symbol=position.symbol,
            signal=SignalType.KEEP,
            reason="Position healthy, no exit triggers hit",
            current_pl_amount=current_pl_amount,
            current_pl_percent=current_pl_percent,
            confidence="HIGH" if position.days_to_expiry > 21 else "MEDIUM"
        )
    
    def _check_close_signals(self, position: ShortPutPosition) -> Optional[str]:
        """Check if position should be closed"""
        
        # Profit target based on premium
        if position.profit_pct() >= self.thresholds['profit_target_pct']:
            return f"Profit target {self.thresholds['profit_target_pct']*100:.0f}% reached"
        
        # ROC-based profit target
        if position.roc() >= self.thresholds['roc_target']:
            if position.days_to_expiry <= 30:
                return f"ROC target {self.thresholds['roc_target']*100:.1f}% achieved"
        
        # Gamma risk management
        if position.days_to_expiry <= self.thresholds['dte_roll_min']:
            if position.profit_pct() > 0.25:
                return f"Gamma risk increasing, {position.days_to_expiry} DTE remaining"
        
        # Deep ITM protection
        if position.delta < -0.70:  # Deep ITM
            return "Deep ITM position, high assignment risk"
        
        return None
    
    def _check_roll_signals(self, position: ShortPutPosition) -> Tuple[Optional[str], Optional[Dict]]:
        """Check if position should be rolled"""
        
        details = {}
        
        # Delta too high (directional risk)
        if abs(position.delta) > self.thresholds['delta_limit']:
            if position.stock_price <= position.strike * 0.95:  # 5% ITM
                details['action'] = "Close (avoid assignment)"
                details['confidence'] = "HIGH"
                return "Position ITM, delta elevated", details
            elif position.iv_rank >= self.thresholds['iv_rank_min_roll']:
                details['action'] = f"Roll to {self._next_expiry(position)} for credit"
                details['limit_price'] = self._estimate_roll_credit(position, lower_strike=True)
                details['confidence'] = "HIGH"
                return "Delta > 0.30, roll for credit possible", details
            else:
                details['action'] = "Watch or close"
                details['confidence'] = "LOW"
                return "Delta elevated but low IV", details
        
        # Extrinsic value depleted
        if position.extrinsic_value < self.thresholds['extrinsic_min']:
            if position.days_to_expiry < 10:
                if abs(position.delta) < 0.20:  # OTM
                    details['action'] = f"Roll to {self._next_expiry(position)} for fresh theta"
                    details['limit_price'] = self._estimate_roll_credit(position, lower_strike=False)
                    details['confidence'] = "MEDIUM"
                    return "Extrinsic value depleted", details
                else:
                    details['action'] = "Close (gamma risk)"
                    details['confidence'] = "HIGH"
                    return "Extrinsic gone, gamma risk high", details
        
        # Too much time remaining (theta decay slowing)
        if position.days_to_expiry > self.thresholds['dte_roll_max']:
            details['action'] = f"Roll to {self._next_month_expiry(position)} for optimal theta"
            details['limit_price'] = self._estimate_roll_credit(position, lower_strike=False)
            details['confidence'] = "MEDIUM"
            return f"DTE > {self.thresholds['dte_roll_max']}, roll for better decay", details
        
        return None, None
    
    def _calculate_roll_pl(self, position: ShortPutPosition, roll_details: Dict) -> float:
        """Calculate projected P/L after rolling"""
        # Current loss/gain + new premium collected
        current_pl = position.profit_amount()
        estimated_new_credit = roll_details.get('limit_price', 0) or 0
        
        # Net of closing cost and new credit
        if estimated_new_credit > 0:
            # You're paying to close, receiving credit for new
            net_credit = estimated_new_credit - position.option_ask
            return current_pl + (net_credit * 100 * abs(position.quantity))
        else:
            return current_pl  # Can't estimate without limit price
    
    def _next_expiry(self, position: ShortPutPosition) -> str:
        """Get next weekly expiry (simplified)"""
        # In real implementation, fetch from options chain
        return f"{position.expiry.month}/{position.expiry.day+7}/26"
    
    def _next_month_expiry(self, position: ShortPutPosition) -> str:
        """Get next monthly expiry"""
        return f"{position.expiry.month+1}/{position.expiry.day}/26"
    
    def _estimate_roll_credit(self, position: ShortPutPosition, lower_strike: bool = False) -> float:
        """Estimate credit for rolling (simplified)"""
        # Based on typical time premium and IV
        time_value = position.theta * 7 * abs(position.quantity)  # 7 days theta
        if lower_strike:
            strike_diff = position.strike * 0.05  # 5% lower strike
            return max(0, time_value - strike_diff)
        return max(0, time_value)

# Simplified version of your original code (now using Pydantic)
class SimpleShortPutPosition(BaseModel):
    """Simplified version matching your original code structure"""
    symbol: str
    strike: float
    stock_price: float
    current_ask: float
    entry_price: float
    dte: int
    delta: float
    theta: float
    iv_rank: float
    dividend_date: Optional[date] = None
    
    @property
    def profit_pct(self) -> float:
        return (self.entry_price - self.current_ask) / self.entry_price
    
    @property
    def intrinsic_value(self) -> float:
        return max(0, self.strike - self.stock_price)
    
    @property
    def extrinsic_value(self) -> float:
        return max(0, self.current_ask - self.intrinsic_value)

def simple_analyze_short_put(position: SimpleShortPutPosition) -> SignalResult:
    """Simplified analyzer based on your original code"""
    
    profit_pct = position.profit_pct
    pl_amount = (position.entry_price - position.current_ask) * 100  # Per contract
    
    # 1. CLOSE SIGNALS
    if profit_pct >= 0.50:
        return SignalResult(
            position_symbol=position.symbol,
            signal=SignalType.CLOSE,
            reason="Standard 50% profit target reached.",
            current_pl_amount=pl_amount,
            current_pl_percent=profit_pct * 100,
            projected_pl_after_action=pl_amount,
            confidence="HIGH"
        )
    
    if position.dte <= 21 and profit_pct > 0.25:
        return SignalResult(
            position_symbol=position.symbol,
            signal=SignalType.CLOSE,
            reason="Gamma risk increasing; capture remaining gains.",
            current_pl_amount=pl_amount,
            current_pl_percent=profit_pct * 100,
            projected_pl_after_action=pl_amount,
            confidence="HIGH"
        )
    
    # 2. ROLL SIGNALS
    if position.delta > 0.45 or position.stock_price <= position.strike:
        if position.iv_rank > 30:
            # Estimate roll credit (simplified)
            est_credit = max(0, position.theta * -7)  # 7 days theta
            projected_pl = pl_amount + (est_credit * 100)
            
            return SignalResult(
                position_symbol=position.symbol,
                signal=SignalType.ROLL,
                reason="ITM/At-the-money. Roll out for net credit.",
                current_pl_amount=pl_amount,
                current_pl_percent=profit_pct * 100,
                projected_pl_after_action=projected_pl,
                recommended_action=f"Roll to next expiry, target credit ${est_credit:.2f}",
                confidence="MEDIUM"
            )
        else:
            return SignalResult(
                position_symbol=position.symbol,
                signal=SignalType.WATCH,
                reason="Low IV makes rolling difficult; wait or prepare for assignment.",
                current_pl_amount=pl_amount,
                current_pl_percent=profit_pct * 100,
                confidence="LOW"
            )
    
    if position.extrinsic_value < (0.10 * position.entry_price) and position.dte < 7:
        est_credit = max(0, position.theta * -14)  # 14 days theta for next cycle
        projected_pl = pl_amount + (est_credit * 100)
        
        return SignalResult(
            position_symbol=position.symbol,
            signal=SignalType.ROLL,
            reason="Extrinsic value depleted. Roll to next cycle to restart Theta.",
            current_pl_amount=pl_amount,
            current_pl_percent=profit_pct * 100,
            projected_pl_after_action=projected_pl,
            recommended_action=f"Roll to next monthly, target credit ${est_credit:.2f}",
            confidence="MEDIUM"
        )
    
    # 3. KEEP SIGNAL
    return SignalResult(
        position_symbol=position.symbol,
        signal=SignalType.KEEP,
        reason="Position healthy. Theta decay working in your favor." if position.theta < 0 else "No exit triggers hit.",
        current_pl_amount=pl_amount,
        current_pl_percent=profit_pct * 100,
        confidence="HIGH"
    )

# Example usage with your NVDA position
if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    # NVDA position from your CSV
    nvda_position = ShortPutPosition(
        symbol="NVDA",
        expiry=date(2026, 3, 13),
        strike=175.0,
        quantity=-1,
        entry_price=4.9948,
        stock_price=184.10,
        option_ask=5.95,
        days_to_expiry=28,
        delta=-0.3253,
        theta=-0.16676,
        gamma=0.01,  # Approximated
        vega=0.15,   # Approximated
        iv_rank=44.74,
        implied_volatility=50.65,
        dividend_date=date(2026, 12, 26)
    )
    
    analyzer = ShortPutAnalyzer()
    result = analyzer.analyze(nvda_position)
    
    print(f"NVDA Analysis:")
    print(f"Signal: {result.signal}")
    print(f"Reason: {result.reason}")
    print(f"Current P/L: ${result.current_pl_amount:.2f} ({result.current_pl_percent:.1f}%)")
    if result.projected_pl_after_action:
        print(f"Projected P/L after action: ${result.projected_pl_after_action:.2f}")
    if result.recommended_action:
        print(f"Recommended: {result.recommended_action}")
    if result.limit_price:
        print(f"Limit price: ${result.limit_price:.2f}")
    print(f"Confidence: {result.confidence}")