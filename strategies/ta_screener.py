import datetime
import logging

import pandas as pd
import talib

from runner.strategy import OneTimeStrategy


class TAScreener(OneTimeStrategy):
    async def _algorithm(self):
        """
        Screens stocks for strong long-term Bear/Bull trends and generates a
        report sent via email.
        """

        sp500_stocks = self.parent_context.data_client.get_companies(index="S&P500")

        stocks_with_ideal_bullish_setup = []
        stocks_with_ideal_bearish_setup = []

        for stock in sp500_stocks:
            # get price data
            symbol = stock["symbol"]

            try:
                from_date = datetime.datetime.now() - datetime.timedelta(days=365)
                price_data = self.parent_context.data_client.get_price_data(
                    symbol=symbol,
                    from_date=from_date.strftime("%Y-%m-%d"),
                )
            except Exception as e:
                logging.info(f"Failed to fetch price data for {symbol}")
                continue

            # calculate 144, 55, 21 day moving averages
            price_data["144_ma"] = talib.SMA(price_data["close"], 144)
            price_data["55_ma"] = talib.SMA(price_data["close"], 55)
            price_data["21_ma"] = talib.SMA(price_data["close"], 21)

            price_data.round(2)

            price = price_data["close"].iloc[-1]
            ma_144 = price_data["144_ma"].iloc[-1]
            ma_55 = price_data["55_ma"].iloc[-1]
            ma_21 = price_data["21_ma"].iloc[-1]
            ma_55_prev = price_data["55_ma"].iloc[-2]
            ma_21_prev = price_data["21_ma"].iloc[-2]

            # ideal bullish setup is if the price is above all the MAs, with
            # all the MAs rising or flat, while 21 MA is above 55 MA, which is
            # above 144 MA
            moving_averages_rising = False
            if ma_55_prev <= ma_55 and ma_21_prev <= ma_21:
                moving_averages_rising = True

            # check if there was a "favorable" MA crossing in the past n days
            price_data["21_ma_lt_55_ma"] = price_data["21_ma"].lt(price_data["55_ma"])
            price_data["55_ma_lt_144_ma"] = price_data["55_ma"].lt(price_data["144_ma"])

            if ma_144 <= ma_55 <= ma_21 and price >= ma_55 and moving_averages_rising:
                # presence of a bullish MA crossing in the last n days
                bullish_ma_crossing_present = (
                    True in price_data.tail(10)["21_ma_lt_55_ma"].values
                    or True in price_data.tail(10)["55_ma_lt_144_ma"].values
                )
                if bullish_ma_crossing_present:
                    stocks_with_ideal_bullish_setup.append(
                        {
                            "Symbol": symbol,
                            "Name": stock["name"],
                            "Price": price,
                            "144MA": ma_144,
                            "55MA": ma_55,
                            "21MA": ma_21,
                            "Sector": stock["sector"],
                        },
                    )
                    continue

            # ideal bearish setup is if the price is bellow all the MAs, with
            # all the MAs falling or flat, while 144 MA is above 51 MA, which is
            # above 21 MA
            moving_averages_falling = False
            if ma_55_prev >= ma_55 and ma_21_prev >= ma_21:
                moving_averages_falling = True

            # check if there's a favorable MA crossing in the past n days
            if ma_21 <= ma_55 <= ma_144 and price <= ma_55 and moving_averages_falling:
                bearish_ma_crossing_present = (
                    False in price_data.tail(10)["21_ma_lt_55_ma"].values
                    or False in price_data.tail(10)["55_ma_lt_144_ma"].values
                )
                if bearish_ma_crossing_present:
                    stocks_with_ideal_bearish_setup.append(
                        {
                            "Symbol": symbol,
                            "Name": stock["name"],
                            "Price": price,
                            "144MA": ma_144,
                            "55MA": ma_55,
                            "21MA": ma_21,
                            "Sector": stock["sector"],
                        },
                    )
                    continue

        if stocks_with_ideal_bullish_setup:
            df = pd.DataFrame.from_records(stocks_with_ideal_bullish_setup)
            message = df.to_html()

            self.parent_context.email_client.send_email(
                subject=f"Stocks in long-term Bull Trend {datetime.datetime.now().strftime('%Y-%m-%d')}",
                body_html=message,
                recipients=["eastempiretradingcompany2019@gmail.com"],
            )

        if stocks_with_ideal_bearish_setup:
            df = pd.DataFrame.from_records(stocks_with_ideal_bearish_setup)
            message = df.to_html()
            self.parent_context.email_client.send_email(
                subject=f"Stocks in long-term Bear Trend {datetime.datetime.now().strftime('%Y-%m-%d')}",
                body_html=message,
                recipients=["eastempiretradingcompany2019@gmail.com"],
            )
            await self.parent_context.telegram_channel.send_message(message)
