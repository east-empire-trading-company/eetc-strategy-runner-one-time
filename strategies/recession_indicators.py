import plotly.graph_objects as go

from runner.strategy import OneTimeStrategy


class RecessionIndicators(OneTimeStrategy):
    async def _algorithm(self):
        """
        Look at all leading indicators that have proven themselves as reliable
        in predicting recessions/crashes.

        We have to make this async due to the Telegram runner code which needs
        to be awaited.
        """

        # Consumer Staples vs Consumer Discretionary rising
        xlp_df = self.parent_context.data_client.get_price_data(
            symbol="XLP",
        )[["date", "open", "high", "low", "close", "volume"]]
        xlp_df = xlp_df.sort_values(by=["date"])

        xly_df = self.parent_context.data_client.get_price_data(
            symbol="XLY",
        )[["date", "open", "high", "low", "close", "volume"]]
        xly_df = xly_df.sort_values(by=["date"])

        if xly_df.shape[0] > xlp_df.shape[0]:
            diff = xly_df.shape[0] - xlp_df.shape[0]
            xly_df = xly_df.tail(xly_df.shape[0] - diff)
        elif xlp_df.shape[0] > xly_df.shape[0]:
            diff = xlp_df.shape[0] - xly_df.shape[0]
            xlp_df = xlp_df.tail(xlp_df.shape[0] - diff)

        xlp_df = xlp_df.reset_index(drop=True)
        xly_df = xly_df.reset_index(drop=True)
        xlp_df["close"] = xlp_df["close"] / xly_df["close"]
        xlp_df = xlp_df.rename({"close": "value"}, axis="columns")

        xlp_df["144_ma"] = xlp_df["value"].rolling(window=144).mean()
        xlp_df["55_ma"] = xlp_df["value"].rolling(window=55).mean()
        xlp_df["21_ma"] = xlp_df["value"].rolling(window=21).mean()

        long_term_ma = round(xlp_df["144_ma"].iloc[-1], 2)
        short_term_ma = round(xlp_df["21_ma"].iloc[-1], 2)

        if short_term_ma >= long_term_ma:
            df = xlp_df.tail(365)  # only plot the last 365 days
            caption = (
                "Consumer Staples vs Consumer Discretionary is trending up \n"
                f"21MA = {short_term_ma} 144MA = {long_term_ma}"
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df["date"], y=df["144_ma"], name="144MA", mode="lines"),
            )
            fig.add_trace(
                go.Scatter(x=df["date"], y=df["55_ma"], name="55MA", mode="lines"),
            )
            fig.add_trace(
                go.Scatter(x=df["date"], y=df["21_ma"], name="21MA", mode="lines"),
            )
            image = fig.to_image(format="png")

            await self.parent_context.telegram_channel.send_image(image, caption)

        # 2Y10Y Yield Curve inverting
        df = self.parent_context.data_client.get_indicator_data(
            "US - 10 Year Treasury Yield minus 2 Year Treasury Yield",
        )
        df = df.sort_values(by=["date"])

        yc_inversion = df["value"].iloc[-1]

        if yc_inversion < 0:
            caption = f"US 10Y2Y Yield Curve is inverting ({yc_inversion})"
            df = df.tail(365 * 10)  # only plot the last 10 years

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["value"],
                    name="10Y Yield minus 2Y Yield",
                    mode="lines",
                ),
            )
            fig.layout.yaxis.zeroline = True
            image = fig.to_image(format="png")

            await self.parent_context.telegram_channel.send_image(image, caption)

        # FED Funds Rate is rising
        df = self.parent_context.data_client.get_indicator_data(
            "US - Federal Funds Rate"
        )
        df = df.sort_values(by=["date"])
        df["12_ma"] = df["value"].rolling(window=12).mean()

        ma = df["12_ma"].iloc[-1]
        fed_funds_rate = df["value"].iloc[-1]

        if fed_funds_rate >= ma:
            caption = f"FED Funds Rate is going up ({fed_funds_rate})"

            df = df.tail(12 * 10)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df["date"], y=df["12_ma"], name="12MA", mode="lines")
            )
            fig.add_trace(
                go.Scatter(x=df["date"], y=df["value"], name="FFR", mode="lines")
            )
            image = fig.to_image(format="png")

            await self.parent_context.telegram_channel.send_image(image, caption)

        # TODO US CPI above 5%
        # TODO Buffet Indicator above threshold
        # TODO University of Michigan - Consumer Sentiment Index

        # US - ISM PMI bellow 50
        df = self.parent_context.data_client.get_indicator_data("PMI")

        pmi = df["value"].iloc[-1]
        if pmi < 50:
            caption = "US - ISM PMI is bellow 50"

            df = df.tail(12 * 3)  # plot only last 3 years for better visibility

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["value"],
                    name="US - PMI",
                    mode="lines",
                ),
            )
            image = fig.to_image(format="png")

            await self.parent_context.telegram_channel.send_image(image, caption)
