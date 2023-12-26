from fastapi import FastAPI

from runner.context import Context
from strategies.calculate_optimal_position_size import CalculateOptimalPositionSize
from strategies.market_implied_volatility import MarketImpliedVolatility
from strategies.animal_spirits import AnimalSpirits
from strategies.net_gamma import NetGamma
from strategies.recession_indicators import RecessionIndicators
from strategies.ta_screener import TAScreener

app = FastAPI()


@app.get("/api/strategy/calculate_optimal_position_sizes")
async def execute_calculate_optimal_position_sizes_strategy():
    context = Context(strategies=[CalculateOptimalPositionSize])

    await context.execute_strategies_async()

    return {"status": "OK"}


@app.get("/api/strategy/ta_screener")
async def execute_calculate_optimal_position_sizes_strategy():
    context = Context(strategies=[TAScreener])

    await context.execute_strategies_async()

    return {"status": "OK"}


@app.get("/api/strategy/market_implied_volatility")
async def execute_market_implied_volatility_strategy():
    context = Context(strategies=[MarketImpliedVolatility])

    await context.execute_strategies_async()

    return {"status": "OK"}


@app.get("/api/strategy/recession_indicators")
async def execute_recession_indicators_strategy():
    context = Context(strategies=[RecessionIndicators])

    await context.execute_strategies_async()

    return {"status": "OK"}


@app.get("/api/strategy/roguetrader_pre_market_report")
async def execute_roguetrader_pre_market_report_strategies():
    context = Context(strategies=[])
    context.add_strategy(MarketImpliedVolatility)
    context.add_strategy(NetGamma)
    await context.execute_strategies_async()

    return {"status": "OK"}


@app.get("/api/strategy/animal_spirits")
async def execute_animal_spirits_strategy():
    context = Context(strategies=[AnimalSpirits])

    await context.execute_strategies_async()

    return {"status": "OK"}
