import datetime
import logging

import numpy as np
import pandas as pd
import talib

from runner.strategy import OneTimeStrategy


class MarkMinervini(OneTimeStrategy):
    async def _algorithm(self):
        """
        Based on the trading strategy of Mark Minervini, which is based on the
        SEPA(Specific Entry Point Analysis) methodology. Links:
            - https://substack.com/home/post/p-142665034
            - https://www.marketsmith.hk/v2/blog/7720?lang=en-US
        """

        df = self._prepare_dataset()
        df = df[df["sepa_stage_2"] == True]

        self.parent_context.email_client.send_email(
            subject=f"Mark Minervini Model {datetime.datetime.now().strftime('%Y-%m-%d')}",
            body_html=df.to_html(),
            recipients=["eastempiretradingcompany2019@gmail.com"],
        )

    def _prepare_dataset(self) -> pd.DataFrame:
        """
        Gather data from various sources & process it.
        :return: Final dataset as a pandas DataFrame.
        """

        columns_to_keep = [
            "symbol",
            "close",
            "sector",
            "sepa_stage_2",
        ]
        df = pd.DataFrame()

        # get the dataset with the prediction column
        companies = self.parent_context.data_client.get_companies(index="S&P500")
        for company in companies:
            symbol_df = self._get_ohlc_price_dataset(company["symbol"])
            if symbol_df is None:
                continue

            # calculate features
            symbol_df = self._calculate_sepa_stages(symbol_df)
            # TODO symbol_df = self._analyze_fundamentals(symbol_df)
            symbol_df["symbol"] = company["symbol"]
            symbol_df["sector"] = company["sector"]
            symbol_df = symbol_df.loc[:, columns_to_keep]

            # append to output DataFrame
            df = pd.concat([df, symbol_df.tail(1)], ignore_index=True)

        df = df.sort_values(by=["sector", "symbol"])

        return df

    def _get_ohlc_price_dataset(self, symbol: str) -> pd.DataFrame | None:
        """
        Get historical OHLC price data for a specific symbol.
        :symbol: Stock or Index symbol to get data for.
        :return: DataFrame with the OHLC price data.
        """

        try:
            df = self.parent_context.data_client.get_price_data(symbol=symbol)
        except Exception:
            logging.info(f"Failed to fetch price data for {symbol}")
            return None

        return df

    def _calculate_sepa_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Trend & Momo features using 21 day SMA & 84 day SMA & RoC,
        respectively. Trend and Momo are represented as discretionary values for
        each regime.
        :return: DataFrame with the new "trend" and "momo" columns.
        """

        df["200_ma"] = talib.SMA(df["close"], 200)
        df["150_ma"] = talib.SMA(df["close"], 150)
        df["50_ma"] = talib.SMA(df["close"], 50)
        df["200_ma_21_ma"] = talib.SMA(df["close"], 21)
        df["200_ma_84_ma"] = talib.SMA(df["close"], 84)
        df['52_week_high'] = df["close"].rolling(window=252, center=False).max()
        df['52_week_low'] = df['close'].rolling(window=252, center=False).min()
        df["200_ma_trend"] = np.where(df["200_ma_21_ma"] > df["200_ma_84_ma"], "Up", "Down")
        df["chaikin_oscillator"] = talib.ADOSC(df["high"], df["low"], df["close"], df["volume"])

        def satisfies_trend_model(row):
            if not (row["close"] > max([row["200_ma"], row["150_ma"], row["50_ma"]])):
                return False
            if not (row["150_ma"] > row["200_ma"]):
                return False
            if not (row["close"] >= row["52_week_low"] * 1.3):
                return False
            if not (row["close"] >= 0.7 * row["52_week_high"]):
                return False
            if row["200_ma_trend"] == "Down":
                return False
            if row["chaikin_oscillator"] < 0:
                return False

            return True

        df["sepa_stage_2"] = df.apply(satisfies_trend_model, axis=1)

        return df
