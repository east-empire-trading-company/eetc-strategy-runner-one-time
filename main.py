from fastapi import FastAPI

from runner.context import Context
from strategies.calculate_optimal_position_size import CalculateOptimalPositionSize

app = FastAPI()


@app.get("/calculate_optimal_position_sizes")
async def execute_calculate_optimal_position_sizes_strategy():
    context = Context()
    context.add_strategy(CalculateOptimalPositionSize)

    await context.execute_strategies_async()

    return {"status": "OK"}
