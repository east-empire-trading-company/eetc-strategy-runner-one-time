from datetime import datetime

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

        iv = self.parent_context.ibkr_client.get_market_implied_volatility(
            symbol,
            strike,
            expiration,
        )

        expiration = expiration.strftime("%d.%m.%Y")

        message = f"Expected Market Implied Volatility {strike}C {expiration} Expo for {symbol}:\n{iv}"
        await self.parent_context.telegram_channel.send_message(message)

        move = round(strike * iv, 2)

        message = f"Expected Intraday SPY Move {expiration}:\n{move}"
        await self.parent_context.telegram_channel.send_message(message)

        low, high = strike - move, strike + move

        message = f"Expected Intraday SPY Range {expiration}:\n{low} - {high}"
        await self.parent_context.telegram_channel.send_message(message)
