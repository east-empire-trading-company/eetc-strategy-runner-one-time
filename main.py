from fastapi import FastAPI

from runner.context import Context
from strategies.calculate_optimal_position_size import CalculateOptimalPositionSize
from strategies.market_implied_volatility import MarketImpliedVolatility

app = FastAPI()


@app.get("/api/strategy/calculate_optimal_position_sizes")
async def execute_calculate_optimal_position_sizes_strategy():
    context = Context(strategies=[CalculateOptimalPositionSize])

    await context.execute_strategies_async()

    return {"status": "OK"}


@app.get("/api/strategy/market_implied_volatility")
async def execute_market_implied_volatility_strategy():
    context = Context(strategies=[MarketImpliedVolatility])

    await context.execute_strategies_async()

    return {"status": "OK"}
