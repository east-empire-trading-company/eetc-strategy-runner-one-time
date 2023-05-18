from datetime import datetime, timedelta

from runner.strategy import OneTimeStrategy

import plotly.graph_objects as go


class NetGamma(OneTimeStrategy):
    async def _algorithm(self):
        """
        Calculate Net Gamma of the underlying based on options for the specified
        expiration dates and strikes.

        We have to make this async due to the Telegram runner code which needs
        to be awaited.
        """

        symbol = "SPY"
        # df = self.parent_context.data_client.get_price_data(symbol=symbol)
        # current_price = df["close"].iloc[-1]  # latest price
        strikes = list(
            range(
                int(
                    self.parent_context.shared_data["market_implied_volatility"][
                        "daily_range_low"
                    ]
                ),
                int(
                    self.parent_context.shared_data["market_implied_volatility"][
                        "daily_range_high"
                    ]
                )
                + 1,
            )
        )
        # strikes = list(range(407, 419))
        expiration_dates = []
        today = datetime.today()
        for exp_date in [today + timedelta(days=x) for x in range(3)]:
            if 1 <= exp_date.isoweekday() <= 5:
                expiration_dates.append(exp_date)
                break  # only get the next earliest possible weekday

        df = self.parent_context.ibkr_client.get_net_gamma(
            symbol,
            strikes,
            expiration_dates,
        )

        df = df.drop("expiration_date", axis=1)
        df = df.groupby(["strike"], as_index=False).sum()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["strike"], y=df["call_gex"], name="Call GEX"))
        fig.add_trace(go.Bar(x=df["strike"], y=df["put_gex"], name="Put GEX"))
        fig.update_xaxes(dtick=1)
        fig.update_layout(barmode="relative")
        image = fig.to_image(format="png")

        today = today.strftime("%d.%m.%Y")

        message = f"{today} 0DTE Option Gamma"

        await self.parent_context.telegram_channel.send_image(image, message)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["strike"], y=df["net_gamma"], name="Net Gamma"))
        fig.update_xaxes(dtick=1)
        image = fig.to_image(format="png")

        message = f"{today} 0DTE Option Net Gamma levels"

        await self.parent_context.telegram_channel.send_image(image, message)
