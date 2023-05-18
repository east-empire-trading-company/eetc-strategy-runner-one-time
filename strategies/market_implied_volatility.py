from datetime import datetime, timedelta

from runner.strategy import OneTimeStrategy


class MarketImpliedVolatility(OneTimeStrategy):
    async def _algorithm(self):
        """
        Calculate Market Implied Volatility of the underlying based on
        ATM options for the specified expiration date. In this case, we want to
        calculate Intraday Market Implied Volatility based on 0DTE ATM options.

        We have to make this async due to the Telegram runner code which needs
        to be awaited.
        """

        symbol = "SPY"
        df = self.parent_context.data_client.get_price_data(symbol=symbol)
        strike = df["close"].iloc[-1]  # latest price
        expiration = datetime.today()

        today = datetime.today()
        for exp_date in [today + timedelta(days=x) for x in range(3)]:
            if 1 <= exp_date.isoweekday() <= 5:
                expiration = exp_date
                break  # only get the next earliest possible weekday

        iv = self.parent_context.ibkr_client.get_market_implied_volatility(
            symbol,
            strike,
            expiration,
        )

        expiration = expiration.strftime("%d.%m.%Y")

        message = f"Expected Market Implied Volatility {strike}C {expiration} Expo for {symbol}:\n{round(iv * 100, 2)}%"
        await self.parent_context.telegram_channel.send_message(message)

        move = round(strike * iv, 2)

        message = f"Expected Intraday SPY Move {expiration}:\n{move}"
        await self.parent_context.telegram_channel.send_message(message)

        low, high = strike - move, strike + move

        message = f"Expected Intraday SPY Range {expiration}:\n{low} - {high}"
        await self.parent_context.telegram_channel.send_message(message)

        self.parent_context.shared_data["market_implied_volatility"] = {
            "daily_range_low": low,
            "daily_range_high": high,
            "daily_move": move,
        }
