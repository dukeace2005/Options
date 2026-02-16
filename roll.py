from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
import math

class OptionType(str, Enum):
    PUT = "PUT"
    CALL = "CALL"

class RollDirection(str, Enum):
    OFFENSIVE = "OFFENSIVE"  # Rolling to improve position (stock moving in your favor)
    DEFENSIVE = "DEFENSIVE"  # Rolling to reduce risk (stock moving against you)
    INCOME = "INCOME"        # Rolling to collect more premium (neutral)

class CurrentPosition(BaseModel):
    """Current position details"""
    symbol: str
    option_type: OptionType
    strike: float
    expiry: date
    entry_price: float  # Premium received
    quantity: int = Field(default=-1, description="Negative for short positions")
    
    # Current market data
    stock_price: float
    days_to_expiry: int
    delta: float
    theta: float
    iv_rank: float
    current_ask: float  # Cost to buy back
    
    @property
    def profit_pct(self) -> float:
        return (self.entry_price - self.current_ask) / self.entry_price
    
    @property
    def profit_amount(self) -> float:
        return (self.entry_price - self.current_ask) * 100 * abs(self.quantity)
    
    @property
    def is_itm(self) -> bool:
        if self.option_type == OptionType.PUT:
            return self.strike > self.stock_price
        else:  # CALL
            return self.strike < self.stock_price

class RollCandidate(BaseModel):
    """Potential roll target option"""
    expiry: date
    strike: float
    bid: float  # Premium you would receive for selling
    ask: float  # Premium you would pay to buy
    delta: float
    theta: float
    iv: float
    dte: int
    
    # Calculated fields
    credit_received: float = Field(0, description="Net credit from roll (per share)")
    new_cost_basis: Optional[float] = None
    projected_pl: Optional[float] = None
    yield_boost: Optional[float] = None
    annualized_yield: Optional[float] = None
    probability_otm: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True

class RollAnalysisResult(BaseModel):
    """Complete roll analysis result"""
    symbol: str
    current_position: CurrentPosition
    roll_direction: RollDirection
    reason: str
    candidates: List[Dict[str, Any]] = Field(default_factory=list, description="Top 3 roll candidates")
    
    # Summary metrics
    current_pl: float
    best_credit: Optional[float] = None
    best_annualized: Optional[float] = None
    recommendation: Optional[str] = None

class RollAnalyzer:
    """Tool to analyze and find best roll options for puts and covered calls"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def analyze_roll_options(
        self, 
        position: CurrentPosition,
        available_options: List[Dict[str, Any]],  # List of options from FMP
        max_candidates: int = 3
    ) -> RollAnalysisResult:
        """
        Analyze available options to find best roll candidates
        
        Args:
            position: Current position details
            available_options: List of option contracts with all Greeks
            max_candidates: Number of top candidates to return
        """
        
        # Determine roll direction based on position status
        roll_direction, reason = self._determine_roll_direction(position)
        
        # Filter relevant options (same type, future expiries)
        candidates = []
        
        for opt in available_options:
            # Skip if not same option type
            if opt.get('option_type') != position.option_type.value:
                continue
            
            # Skip if expiry is in the past or same as current
            opt_expiry = datetime.strptime(opt['expiry'], '%Y-%m-%d').date()
            if opt_expiry <= position.expiry:
                continue
            
            # Calculate roll metrics
            candidate = self._evaluate_roll_candidate(position, opt)
            if candidate and candidate.credit_received > 0:  # Only consider credit rolls
                candidates.append(candidate)
        
        # Sort by different criteria and get top candidates
        candidates.sort(key=lambda x: x.credit_received, reverse=True)
        top_by_credit = candidates[:max_candidates]
        
        candidates.sort(key=lambda x: x.annualized_yield or 0, reverse=True)
        top_by_yield = candidates[:max_candidates]
        
        candidates.sort(key=lambda x: abs(x.delta), reverse=True)  # Higher delta = more premium
        top_by_delta = candidates[:max_candidates]
        
        # Combine and deduplicate top candidates
        top_candidates = {}
        for cand in top_by_credit + top_by_yield + top_by_delta:
            key = f"{cand.expiry}_{cand.strike}"
            if key not in top_candidates:
                top_candidates[key] = cand
        
        # Format results
        formatted_candidates = []
        for cand in list(top_candidates.values())[:max_candidates]:
            formatted_candidates.append(self._format_candidate(cand, position))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            position, formatted_candidates, roll_direction
        )
        
        return RollAnalysisResult(
            symbol=position.symbol,
            current_position=position,
            roll_direction=roll_direction,
            reason=reason,
            candidates=formatted_candidates,
            current_pl=position.profit_amount,
            best_credit=max([c['credit_received'] for c in formatted_candidates]) if formatted_candidates else None,
            best_annualized=max([c['annualized_yield'] for c in formatted_candidates if c['annualized_yield']]) if formatted_candidates else None,
            recommendation=recommendation
        )
    
    def _determine_roll_direction(self, position: CurrentPosition) -> tuple:
        """Determine if roll should be offensive, defensive, or income-focused"""
        
        if position.option_type == OptionType.PUT:
            if position.stock_price > position.strike * 1.05:  # Stock up 5%+
                return RollDirection.OFFENSIVE, "Stock moving higher, can roll up for more premium"
            elif position.stock_price < position.strike * 0.95:  # Stock down 5%+
                return RollDirection.DEFENSIVE, "Position challenged, roll down/out to reduce risk"
            else:
                return RollDirection.INCOME, "Position neutral, roll for time premium"
        
        else:  # Covered Call
            if position.stock_price < position.strike * 0.95:  # Stock down
                return RollDirection.DEFENSIVE, "Stock down, roll down to generate more income"
            elif position.stock_price > position.strike * 1.05:  # Stock up, call ITM
                return RollDirection.OFFENSIVE, "Call ITM, roll up/out to capture upside"
            else:
                return RollDirection.INCOME, "Position neutral, roll for time premium"
    
    def _evaluate_roll_candidate(
        self, 
        position: CurrentPosition, 
        option_data: Dict[str, Any]
    ) -> Optional[RollCandidate]:
        """Evaluate a single roll candidate and calculate metrics"""
        
        try:
            expiry = datetime.strptime(option_data['expiry'], '%Y-%m-%d').date()
            dte = (expiry - datetime.now().date()).days
            
            # Calculate net credit for rolling
            # You buy back current (ask) and sell new (bid)
            cost_to_close = position.current_ask
            premium_to_open = option_data.get('bid', 0)
            
            net_credit = premium_to_open - cost_to_close
            
            # Calculate new cost basis
            total_credits = position.entry_price + net_credit
            new_cost_basis = position.strike - total_credits if position.option_type == OptionType.PUT else None
            
            # Calculate projected P/L after roll
            # Current P/L + new net credit
            projected_pl = position.profit_amount + (net_credit * 100 * abs(position.quantity))
            
            # Calculate yield metrics
            capital_at_risk = position.strike * 100 * abs(position.quantity)
            yield_boost = (net_credit * 100 * abs(position.quantity)) / capital_at_risk if capital_at_risk > 0 else 0
            annualized_yield = yield_boost * (365 / dte) if dte > 0 else 0
            
            # Probability OTM (simplified using delta)
            prob_otm = 1 - abs(option_data.get('delta', 0))
            
            return RollCandidate(
                expiry=expiry,
                strike=option_data['strike'],
                bid=premium_to_open,
                ask=option_data.get('ask', 0),
                delta=option_data.get('delta', 0),
                theta=option_data.get('theta', 0),
                iv=option_data.get('iv', 0),
                dte=dte,
                credit_received=net_credit,
                new_cost_basis=new_cost_basis,
                projected_pl=projected_pl,
                yield_boost=yield_boost * 100,  # Convert to percentage
                annualized_yield=annualized_yield * 100,  # Convert to percentage
                probability_otm=prob_otm * 100
            )
            
        except Exception as e:
            print(f"Error evaluating candidate: {e}")
            return None
    
    def _format_candidate(self, candidate: RollCandidate, position: CurrentPosition) -> Dict[str, Any]:
        """Format candidate for output similar to NVDA example"""
        
        option_type_str = "PUT" if position.option_type == OptionType.PUT else "CALL"
        expiry_str = candidate.expiry.strftime("%b-%d")
        
        # Determine if strike adjustment is up or down
        strike_diff = candidate.strike - position.strike
        strike_direction = "up" if strike_diff > 0 else "down" if strike_diff < 0 else "same"
        
        # Calculate new breakeven for puts
        if position.option_type == OptionType.PUT:
            new_breakeven = candidate.strike - candidate.bid
            current_breakeven = position.strike - position.entry_price
        else:  # Covered call
            new_breakeven = position.stock_price - candidate.bid  # Simplified
            current_breakeven = position.stock_price - position.entry_price
        
        return {
            "option": f"{expiry_str} ${candidate.strike} {option_type_str}",
            "net_credit": f"${candidate.credit_received:.2f}",
            "credit_per_contract": f"${candidate.credit_received * 100:.2f}",
            "dte": candidate.dte,
            "delta": f"{candidate.delta:.3f}",
            "theta": f"${candidate.theta:.3f}",
            "iv": f"{candidate.iv:.1f}%" if candidate.iv else "N/A",
            "yield_boost": f"{candidate.yield_boost:.2f}%" if candidate.yield_boost else "N/A",
            "annualized": f"{candidate.annualized_yield:.1f}%" if candidate.annualized_yield else "N/A",
            "prob_otm": f"{candidate.probability_otm:.0f}%" if candidate.probability_otm else "N/A",
            "new_cost_basis": f"${candidate.new_cost_basis:.2f}" if candidate.new_cost_basis else "N/A",
            "projected_pl": f"${candidate.projected_pl:.2f}",
            "strike_adjustment": strike_direction,
            "current_breakeven": f"${current_breakeven:.2f}",
            "new_breakeven": f"${new_breakeven:.2f}",
            "explanation": self._generate_candidate_explanation(candidate, position, strike_direction)
        }
    
    def _generate_candidate_explanation(
        self, 
        candidate: RollCandidate, 
        position: CurrentPosition,
        strike_direction: str
    ) -> str:
        """Generate human-readable explanation for candidate"""
        
        parts = []
        
        # Credit explanation
        if candidate.credit_received > 0:
            parts.append(f"Net credit ${candidate.credit_received:.2f} per share (${candidate.credit_received*100:.2f} per contract)")
        
        # Yield explanation
        if candidate.yield_boost:
            parts.append(f"{candidate.yield_boost:.2f}% yield boost ({candidate.annualized_yield:.1f}% annualized)")
        
        # Probability
        if candidate.probability_otm:
            prob_text = f"{candidate.probability_otm:.0f}% chance of expiring OTM"
            parts.append(prob_text)
        
        # Strike adjustment context
        if position.option_type == OptionType.PUT:
            if strike_direction == "up":
                parts.append("Raising strike to capture more upside")
            elif strike_direction == "down":
                parts.append("Lowering strike for more downside protection")
        else:  # Covered call
            if strike_direction == "up":
                parts.append("Rolling up to allow more upside potential")
            elif strike_direction == "down":
                parts.append("Rolling down to generate more income")
        
        # Time extension
        time_added = candidate.dte - position.days_to_expiry
        if time_added > 0:
            parts.append(f"Adding {time_added} days of theta decay")
        
        return ". ".join(parts)
    
    def _generate_recommendation(
        self, 
        position: CurrentPosition,
        candidates: List[Dict[str, Any]],
        direction: RollDirection
    ) -> str:
        """Generate final recommendation"""
        
        if not candidates:
            return "No suitable roll candidates found with net credit"
        
        best = candidates[0]
        
        if direction == RollDirection.DEFENSIVE:
            return (f"DEFENSIVE ROLL: Consider {best['option']} for ${best['credit_per_contract']} credit. "
                   f"This lowers your risk while adding {best['dte'] - position.days_to_expiry} days. "
                   f"New breakeven: {best['new_breakeven']} (from {best['current_breakeven']})")
        
        elif direction == RollDirection.OFFENSIVE:
            return (f"OFFENSIVE ROLL: {best['option']} offers {best['annualized']} annualized yield. "
                   f"Projected P/L improves to {best['projected_pl']} from ${position.profit_amount:.2f}")
        
        else:  # INCOME
            return (f"INCOME ROLL: Roll to {best['option']} for {best['yield_boost']} yield boost "
                   f"({best['annualized']} annualized). Best balance of credit and probability.")

# Example usage with NVDA data
def analyze_nvda_roll_example():
    """Example analysis for NVDA puts (based on your position)"""
    
    current_position = CurrentPosition(
        symbol="NVDA",
        option_type=OptionType.PUT,
        strike=175.0,
        expiry=date(2026, 3, 13),
        entry_price=4.9948,
        quantity=-1,
        stock_price=184.10,
        days_to_expiry=28,
        delta=-0.3253,
        theta=-0.16676,
        iv_rank=44.74,
        current_ask=5.95
    )
    
    # Simulated available options data (would come from FMP)
    available_options = [
        # March 20th options
        {"expiry": "2026-03-20", "strike": 175, "bid": 7.50, "ask": 7.85, "delta": -0.35, "theta": -0.19, "iv": 48.5, "option_type": "PUT"},
        {"expiry": "2026-03-20", "strike": 170, "bid": 5.20, "ask": 5.50, "delta": -0.28, "theta": -0.16, "iv": 47.8, "option_type": "PUT"},
        {"expiry": "2026-03-20", "strike": 165, "bid": 3.40, "ask": 3.65, "delta": -0.21, "theta": -0.13, "iv": 47.2, "option_type": "PUT"},
        
        # March 27th options [citation:2]
        {"expiry": "2026-03-27", "strike": 175, "bid": 8.95, "ask": 9.30, "delta": -0.38, "theta": -0.16, "iv": 49.2, "option_type": "PUT"},
        {"expiry": "2026-03-27", "strike": 170, "bid": 6.40, "ask": 6.70, "delta": -0.31, "theta": -0.14, "iv": 48.5, "option_type": "PUT"},
        {"expiry": "2026-03-27", "strike": 165, "bid": 4.30, "ask": 4.55, "delta": -0.24, "theta": -0.11, "iv": 48.0, "option_type": "PUT"},
        
        # April 17th options
        {"expiry": "2026-04-17", "strike": 175, "bid": 12.50, "ask": 13.00, "delta": -0.42, "theta": -0.12, "iv": 51.5, "option_type": "PUT"},
        {"expiry": "2026-04-17", "strike": 170, "bid": 9.80, "ask": 10.20, "delta": -0.36, "theta": -0.10, "iv": 50.8, "option_type": "PUT"},
    ]
    
    analyzer = RollAnalyzer()
    result = analyzer.analyze_roll_options(current_position, available_options)
    
    print(f"\n{'='*60}")
    print(f"NVDA PUT ROLL ANALYSIS")
    print(f"{'='*60}")
    print(f"Current: -1 NVDA Mar-13-26 $175 Put @ ${current_position.entry_price}")
    print(f"Stock: ${current_position.stock_price} | DTE: {current_position.days_to_expiry}")
    print(f"Current P/L: ${result.current_pl:.2f} ({current_position.profit_pct*100:.1f}%)")
    print(f"Direction: {result.roll_direction.value} - {result.reason}")
    print(f"\nTOP 3 ROLL CANDIDATES:")
    print(f"{'-'*60}")
    
    for i, cand in enumerate(result.candidates, 1):
        print(f"\n{i}. {cand['option']}")
        print(f"   Net Credit: {cand['net_credit']} ({cand['credit_per_contract']} per contract)")
        print(f"   DTE: {cand['dte']} | Delta: {cand['delta']} | Theta: {cand['theta']}")
        print(f"   Yield: {cand['yield_boost']} ({cand['annualized']} annualized)")
        print(f"   Prob OTM: {cand['prob_otm']}")
        print(f"   Projected P/L: {cand['projected_pl']}")
        print(f"   {cand['explanation']}")
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {result.recommendation}")
    print(f"{'='*60}")

# Example for covered call analysis
def analyze_nvda_covered_call_example():
    """Example analysis for NVDA covered calls"""
    
    # Assume you own 100 shares of NVDA at $184.10
    current_position = CurrentPosition(
        symbol="NVDA",
        option_type=OptionType.CALL,
        strike=190.0,  # You sold $190 calls
        expiry=date(2026, 3, 13),
        entry_price=3.50,  # Received $3.50 premium
        quantity=-1,
        stock_price=184.10,
        days_to_expiry=28,
        delta=0.28,  # Positive delta for calls
        theta=-0.12,
        iv_rank=44.74,
        current_ask=2.80  # Cost to buy back
    )
    
    available_options = [
        # March 20th calls
        {"expiry": "2026-03-20", "strike": 190, "bid": 4.20, "ask": 4.50, "delta": 0.32, "theta": -0.14, "iv": 46.5, "option_type": "CALL"},
        {"expiry": "2026-03-20", "strike": 195, "bid": 2.80, "ask": 3.10, "delta": 0.24, "theta": -0.11, "iv": 45.8, "option_type": "CALL"},
        
        # March 27th calls [citation:2]
        {"expiry": "2026-03-27", "strike": 190, "bid": 5.80, "ask": 6.20, "delta": 0.35, "theta": -0.12, "iv": 47.2, "option_type": "CALL"},
        {"expiry": "2026-03-27", "strike": 195, "bid": 4.10, "ask": 4.40, "delta": 0.28, "theta": -0.10, "iv": 46.5, "option_type": "CALL"},
        {"expiry": "2026-03-27", "strike": 200, "bid": 2.90, "ask": 3.20, "delta": 0.21, "theta": -0.08, "iv": 46.0, "option_type": "CALL"},
        
        # April 17th calls
        {"expiry": "2026-04-17", "strike": 195, "bid": 7.20, "ask": 7.60, "delta": 0.38, "theta": -0.09, "iv": 49.5, "option_type": "CALL"},
    ]
    
    analyzer = RollAnalyzer()
    result = analyzer.analyze_roll_options(current_position, available_options)
    
    print(f"\n{'='*60}")
    print(f"NVDA COVERED CALL ROLL ANALYSIS")
    print(f"{'='*60}")
    print(f"Current: -1 NVDA Mar-13-26 $190 Call @ ${current_position.entry_price}")
    print(f"Stock: ${current_position.stock_price} | DTE: {current_position.days_to_expiry}")
    print(f"Current P/L: ${result.current_pl:.2f} ({current_position.profit_pct*100:.1f}%)")
    print(f"Direction: {result.roll_direction.value} - {result.reason}")
    print(f"\nTOP 3 ROLL CANDIDATES:")
    print(f"{'-'*60}")
    
    for i, cand in enumerate(result.candidates, 1):
        print(f"\n{i}. {cand['option']}")
        print(f"   Net Credit: {cand['net_credit']} ({cand['credit_per_contract']} per contract)")
        print(f"   DTE: {cand['dte']} | Delta: {cand['delta']} | Theta: {cand['theta']}")
        print(f"   Yield: {cand['yield_boost']} ({cand['annualized']} annualized)")
        print(f"   Prob OTM: {cand['prob_otm']}")
        print(f"   New Breakeven: {cand['new_breakeven']}")
        print(f"   {cand['explanation']}")
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {result.recommendation}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Run NVDA put analysis (matching your position)
    analyze_nvda_roll_example()
    
    # Run covered call analysis for comparison
    analyze_nvda_covered_call_example()