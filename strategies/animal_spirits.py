import datetime
import logging
from math import sqrt
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import talib
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

from runner.strategy import OneTimeStrategy


class AnimalSpirits(OneTimeStrategy):
    async def _algorithm(self):
        """
        Use a random forest classifier to classify the current market cycle
        based on historical data and features covering various aspects associated
        with Bull and Bear markets such as price action, growth, inflation,
        liquidity, monetary and fiscal policy, consumer behavior, corporate profits, etc.

        This model is one of the main foundations for making further trade
        decisions and trade idea generation.

        TODO: Model predicts Bear markets and a lot of corrections perfectly,
            but it exits a too late on some of them, probably because we are
            lacking features that are tied to Bear market bottoms, which are
            different from the ones preceding the Bear markets and corrections.
            Most likely, it's the liquidity-related features, since the FED
            always pumps a ton of liquidity to save the market, which usually
            marks the bottom.
        """

        df, label_encoder_per_col_map = self._prepare_dataset()

        df, prediction = self._train_and_execute_model(df, label_encoder_per_col_map)
        # uncomment to test the model
        # self._train_and_test_model(df, label_encoder_per_col_map)

        # rename columns for presentation purposes
        df = self._rename_columns(df)

        # send out the report
        message = df.iloc[-1:].to_html()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        prediction = df.iloc[-1]["Cycle"]
        subject = f"Animal Spirits Model Prediction {today} - {prediction}"
        self.parent_context.email_client.send_email(
            subject=subject,
            body_html=message,
            recipients=["eastempiretradingcompany2019@gmail.com"],
        )
        # TODO Generate a Telegram text message containing all the data that was
        #   used to make the prediction.
        await self.parent_context.telegram_channel.send_message(subject)

    def _prepare_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Gather data from various sources, process it and create the train and
        test datasets for the ML model.
        :return: Train and Test datasets, in that order.
        """

        # get the dataset with the prediction column
        df = self._get_sp500_bull_bear_dataset()

        # calculate features
        # TODO split training-test set before doing anything (data leakage)
        #   when doing this, make sure to use bias correction
        df = self._calculate_trend_momo_features(df)
        df = self._calculate_interest_rate_features(df)
        df = self._calculate_gdp_growth_features(df)
        df = self._calculate_unemployment_features(df)
        df = self._calculate_pmi_features(df)
        df = self._calculate_inflation_features(df)
        df = self._calculate_consumer_behavior_features(df)
        df = self._calculate_corporate_behavior_features(df)
        # TODO df = self._calculate_liquidity_features(df)

        # clean up dataset
        columns_to_keep = [
            "date",  # not used as a feature column
            "cycle",  # prediction column
            "trend",
            "momo",
            "interest_rates_lvl",
            "interest_rates_direction",
            "gdp_growth_vol",
            "cpi_roc",
            "cpi_vol",
            "cpi_lvl",
            "sahm_rule",
            "ism_pmi",
            "ig_regime",
            "pce_roc",
            "pce_vol",
            "pdi_roc",
            "pdi_vol",
            "pdi",
            "pce",
            "corporate_profits",
            "corporate_profits_vol",
        ]
        df = self._clean_up_dataset(df, columns_to_keep)

        # transform categorical columns to numerical
        columns_to_transform = [
            "cycle",
            "trend",
            "momo",
            "interest_rates_lvl",
            "interest_rates_direction",
            "gdp_growth_vol",
            "cpi_vol",
            "cpi_lvl",
            "ig_regime",
            "pce_vol",
            "pdi_vol",
            "corporate_profits_vol",
        ]
        (
            df,
            label_encoder_per_col_map,
        ) = self._transform_categorical_columns_to_numerical(df, columns_to_transform)

        self._check_multicollinearity(df)

        return df, label_encoder_per_col_map

    def _get_sp500_bull_bear_dataset(self) -> pd.DataFrame:
        """
        Get historical S&P500 OHLC price data and calculate the Bull and Bear
        cycles using the standard definition, but with a 17% threshold instead
        of the standard 20%.
        Values for the "cycle" column:
            - "Bull"
            - "Bear"
        This is the column we will be doing the predictions on.
        :return: DataFrame with the new "cycle" column.
        """

        symbol = "^GSPC"  # S&P500

        try:
            df = self.parent_context.data_client.get_price_data(
                symbol=symbol,
            )
        except Exception as e:
            logging.info(f"Failed to fetch price data for {symbol}")
            raise e

        # calculate daily returns
        df["return"] = df["close"].pct_change()

        # calculate Bull and Bear cycles
        df["dd"] = df["close"].div(df["close"].cummax()).sub(1)
        df["ddn"] = ((df["dd"] < 0.0) & (df["dd"].shift() == 0.0)).cumsum()
        df["ddmax"] = df.groupby("ddn")["dd"].transform("min")
        df["bear"] = (df["ddmax"] <= -0.17) & (
            df["ddmax"] < df.groupby("ddn")["dd"].transform("cummin")
        )
        df["bearn"] = ((df["bear"] == True) & (df["bear"].shift() == False)).cumsum()
        df["cycle"] = df["bear"].apply(lambda bear: "Bear" if bear is True else "Bull")

        # uncomment to plot bull and bear markets
        # df_pred = x_test.copy()
        # df_pred["date"] = x_test_dates
        # df_pred["cycle"] = y_test
        # df_pred["prediction"] = model.predict(x_test)
        # print(df_pred)

        # plt.figure(figsize=(12, 6))
        # plt.plot(df['date'], df['close'], label='Price', color='b')
        # plt.fill_between(
        #     df['date'],
        #     df['close'],
        #     where=df['cycle'] == 'Bear',
        #     color='grey',
        #     alpha=0.5,
        #     label='Bear Market',
        # )
        # plt.title('S&P 500 Bull and Bear cycles')
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.show()
        #
        # image = io.BytesIO()
        # image.name = 'image.png'
        # plt.savefig(image, format='PNG')
        # image.seek(0)
        # image = image.read()
        #
        # message = f"S&P 500 Bull and Bear cycles"
        #
        # await self.parent_context.telegram_channel.send_image(image, message)

        return df

    def _calculate_trend_momo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Trend & Momo features using 21 day SMA & 84 day SMA & RoC,
        respectively. Trend and Momo are represented as discretionary values for
        each regime.
        :return: DataFrame with the new "trend" and "momo" columns.
        """

        # calculate Trend using 21 day SMA and 84 day SMA
        df["21_ma"] = talib.SMA(df["close"], 21)
        df["84_ma"] = talib.SMA(df["close"], 84)
        df["trend"] = np.where(df["21_ma"] > df["84_ma"], "Up", "Down")

        # calculate Momentum using 21 day ROC and 84 day ROC
        df["21_roc"] = (df["close"] / df["close"].shift(21) - 1) * 100
        df["84_roc"] = (df["close"] / df["close"].shift(84) - 1) * 100

        def get_momo_regime(row):
            if row["84_roc"] > 0:
                return "Up" if row["21_roc"] > 0 else "Weak Down"
            else:
                return "Weak Up" if row["21_roc"] > 0 else "Down"

        df["momo"] = df.apply(get_momo_regime, axis=1)

        return df

    def _get_missing_dates_for_joining(self, df: pd.DataFrame) -> pd.Index:
        """
        Used when left joining based on the "date" column in cases where the
        left DataFrame does not have all the dates in the right DataFrame.
        :return: Index column containing all the dates from the left DataFrame.
        """

        all_dates = pd.date_range(start=min(df["date"]), end=max(df["date"]))
        return all_dates.strftime("%Y-%m-%dT%H:%M:%S")

    def _join_feature_df_with_main_df(
        self,
        main_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left join feature DataFrame to the main DataFrame based on the "date" column.
        :param main_df: LEFT DataFrame in the JOIN.
        :param feature_df: RIGHT DataFrame in the JOIN.
        :return: Main DataFrame with the new feature columns.
        """

        all_dates = self._get_missing_dates_for_joining(main_df)
        feature_df = feature_df.set_index("date")
        feature_df = feature_df.reindex(all_dates, method="ffill")
        feature_df = feature_df.reset_index().rename(columns={"index": "date"})

        return pd.merge(main_df, feature_df, on="date", how="left").ffill()

    def _clean_up_dataset(
        self,
        df: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> pd.DataFrame:
        """
        Perform clean up operations on the dataset such as rounding, removing
        invalid values, removing duplicate rows, etc.
        """

        df = df.loc[:, columns_to_keep]
        df = df.round(2)
        df = df.replace([np.inf, -np.inf], 0, inplace=False)
        df = df.dropna()
        df = df.drop_duplicates(subset=df.columns.difference(["date"]), keep="first")

        return df

    def _calculate_interest_rate_features(self, df: pd.DataFrame):
        """
        Calculate the features related to interest rates.
        :return: DataFrame with new columns "interest_rates", "interest_rates_lvl"
        and "interest_rates_direction".
        """

        indicator = "US - Federal Funds Rate"
        try:
            df_ffr = self.parent_context.data_client.get_indicator_data(
                name=indicator,
            )
            df_ffr = df_ffr.loc[:, ["date", "value"]]
            df_ffr = df_ffr.rename(columns={"value": "interest_rates"})
        except Exception as e:
            logging.info(f"Failed to fetch indicator data for {indicator}")
            raise e

        def _get_interest_rates_direction(row):
            if row["interest_rates_direction"] > 0:
                return "Rising"
            elif row["interest_rates_direction"] < 0:
                return "Falling"

            return "Unchanged"

        average_interest_rate = df_ffr.loc[:, "interest_rates"].median()
        df_ffr["interest_rates_direction"] = df_ffr["interest_rates"].pct_change()
        df_ffr["interest_rates_direction"] = df_ffr.apply(
            _get_interest_rates_direction,
            axis=1,
        )
        df_ffr["interest_rates_lvl"] = df_ffr["interest_rates"].apply(
            lambda val: "Low" if val < average_interest_rate else "High",
        )

        df = self._join_feature_df_with_main_df(df, df_ffr)

        return df

    def _calculate_gdp_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Real GDP Growth features such as Real GDP Growth, etc.
        :return: DataFrame with the new "gdp_growth" and "gdp_growth_vol" columns.
        """

        indicator = "US - Real GDP Growth Rate"
        try:
            df_gdp = self.parent_context.data_client.get_indicator_data(
                name=indicator,
                frequency="Quarterly",
            )
            df_gdp = df_gdp.loc[:, ["date", "value"]]
            df_gdp = df_gdp.rename(columns={"value": "gdp_growth"})
        except Exception as e:
            logging.info(f"Failed to fetch indicator data for {indicator}")
            raise e

        gdp_growth_std = df_gdp.loc[:, "gdp_growth"].std()
        df_gdp["gdp_growth_vol"] = df_gdp["gdp_growth"].rolling(8).std().fillna(0)
        df_gdp["gdp_growth_vol"] = df_gdp["gdp_growth_vol"].apply(
            lambda val: "Low" if val <= gdp_growth_std else "High",
        )

        df = self._join_feature_df_with_main_df(df, df_gdp)

        return df

    def _calculate_unemployment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Unemployment features such as Sahm Rule, etc.
        :return: DataFrame with the new "sahm_rule" column.
        """

        indicator = "US - Sahm Rule Indicator"
        try:
            df_sahm = self.parent_context.data_client.get_indicator_data(
                name=indicator,
            )
            df_sahm = df_sahm.loc[:, ["date", "value"]]
            df_sahm = df_sahm.rename(columns={"value": "sahm_rule"})
        except Exception as e:
            logging.info(f"Failed to fetch indicator data for {indicator}")
            raise e

        df = self._join_feature_df_with_main_df(df, df_sahm)

        return df

    def _calculate_pmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PMI features such as ISM PMI, etc.
        :return: DataFrame with the new "ism_pmi" column.
        """

        indicator = "PMI"
        try:
            df_pmi = self.parent_context.data_client.get_indicator_data(
                name=indicator,
            )
            df_pmi = df_pmi.loc[:, ["date", "value"]]
            df_pmi = df_pmi.rename(columns={"value": "ism_pmi"})
        except Exception as e:
            logging.info(f"Failed to fetch indicator data for {indicator}")
            raise e

        df = self._join_feature_df_with_main_df(df, df_pmi)

        return df

    def _calculate_inflation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Inflation features such as CPI, Inflation/Growth regime, etc.
        :return: DataFrame with the new "cpi", "cpi_vol", "cpi_lvl" and
        "ig_regime" columns.
        """

        indicator = "US - CPI YoY"
        try:
            df_cpi = self.parent_context.data_client.get_indicator_data(
                name=indicator,
                frequency="Monthly",
            )
            df_cpi = df_cpi.loc[:, ["date", "value"]]
            df_cpi = df_cpi.rename(columns={"value": "cpi"})
            df_cpi["cpi_roc"] = df_cpi["cpi"].pct_change()
        except Exception as e:
            logging.info(f"Failed to fetch indicator data for {indicator}")
            raise e

        def _get_cpi_vol(row):
            if row["cpi_std_3m"] > row["cpi_std_12m"]:
                return "High"
            elif row["cpi_std_3m"] < row["cpi_std_12m"]:
                return "Low"

            return "Low"

        def _get_cpi_lvl(row):
            if row["cpi"] > 0.04:
                return "Plebflation"
            elif row["cpi"] < 0:
                return "Deflation"

            return "Wealthflation"

        df_cpi["cpi_std_12m"] = df_cpi["cpi"].rolling(12).std().fillna(0)
        df_cpi["cpi_std_3m"] = df_cpi["cpi"].rolling(3).std().fillna(0)
        df_cpi["cpi_vol"] = df_cpi.apply(_get_cpi_vol, axis=1)
        df_cpi["cpi_lvl"] = df_cpi.apply(_get_cpi_lvl, axis=1)
        df = self._join_feature_df_with_main_df(df, df_cpi)

        # calculate IG (I=Inflation, G=Growth) Regime
        def _get_ig_regime(row):
            """
            Regimes: I+G+, I-G-, I+G-, I-G+.
            I = Inflation, G = Growth
            """

            if row["cpi_roc"] <= 0 and row["gdp_growth"] <= 0:
                return "I-G-"
            elif row["cpi_roc"] > 0 and row["gdp_growth"] <= 0:
                return "I+G-"
            elif row["cpi_roc"] <= 0 and row["gdp_growth"] > 0:
                return "I-G+"

            return "I+G+"

        df["ig_regime"] = df.apply(_get_ig_regime, axis=1)

        return df

    def _calculate_consumer_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate consumer behavior features such as Consumption, etc.
        :return: DataFrame with the new "pce" and "pce_vol" columns.
        """

        indicator = "US - Real Personal Consumption Expenditures YoY"
        try:
            df_pce = self.parent_context.data_client.get_indicator_data(
                name=indicator,
                frequency="Monthly",
            )
            df_pce = df_pce.loc[:, ["date", "value"]]
            df_pce = df_pce.rename(columns={"value": "pce"})
            df_pce["pce_roc"] = df_pce["pce"].pct_change()
        except Exception as e:
            logging.info(f"Failed to fetch indicator data for {indicator}")
            raise e

        def get_pce_vol(row):
            if row["pce_std_3m"] > row["pce_std_12m"]:
                return "High"
            elif row["pce_std_3m"] < row["pce_std_12m"]:
                return "Low"

            return "Low"

        df_pce["pce_std_12m"] = df_pce["pce"].rolling(12).std().fillna(0)
        df_pce["pce_std_3m"] = df_pce["pce"].rolling(3).std().fillna(0)
        df_pce["pce_vol"] = df_pce.apply(get_pce_vol, axis=1)

        df = self._join_feature_df_with_main_df(df, df_pce)

        return df

    def _calculate_corporate_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate corporate behavior features such as Investment, Profit, etc.
        :return: DataFrame with the new "pdi", "pdi_vol", "pdi_roc",
        "corporate_profits" and "corporate_profits_vol" columns.
        """

        # Real Gross Private Domestic Investment YoY
        indicator = "US - Real Gross Private Domestic Investment YoY"
        try:
            df_pdi = self.parent_context.data_client.get_indicator_data(
                name=indicator,
                frequency="Quarterly",
            )
            df_pdi = df_pdi.loc[:, ["date", "value"]]
            df_pdi = df_pdi.rename(columns={"value": "pdi"})
            df_pdi["pdi_roc"] = df_pdi["pdi"].pct_change()
        except Exception as e:
            logging.info(f"Failed to fetch indicator data for {indicator}")
            raise e

        pdi_std = df_pdi.loc[:, "pdi"].std()
        df_pdi["pdi_vol"] = df_pdi["pdi"].rolling(8).std().fillna(0)
        df_pdi["pdi_vol"] = df_pdi["pdi_vol"].apply(
            lambda val: "Low" if val <= pdi_std else "High",
        )

        df = self._join_feature_df_with_main_df(df, df_pdi)

        # US Corporate profits after Tax YoY
        indicator = "US - Corporate profits after Tax YoY"
        try:
            df_cp = self.parent_context.data_client.get_indicator_data(
                name=indicator,
                frequency="Quarterly",
            )
            df_cp = df_cp.loc[:, ["date", "value"]]
            df_cp = df_cp.rename(columns={"value": "corporate_profits"})
        except Exception as e:
            logging.info(f"Failed to fetch indicator data for {indicator}")
            raise e

        corporate_profits_std = df_cp.loc[:, "corporate_profits"].std()
        df_cp["corporate_profits_vol"] = (
            df_cp["corporate_profits"].rolling(8).std().fillna(0)
        )
        df_cp["corporate_profits_vol"] = df_cp["corporate_profits_vol"].apply(
            lambda val: "Low" if val <= corporate_profits_std else "High",
        )

        df = self._join_feature_df_with_main_df(df, df_cp)

        # TODO feature to see if personal interest/debt payments are affecting
        #  personal consumption (YoY increase leads to YoY increase), overlaid
        #  with Interest Rate lvl and trend/RoC

        return df

    def _transform_categorical_columns_to_numerical(
        self,
        df: pd.DataFrame,
        columns_to_transform: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Transform categorical columns to numerical.
        :return: DataFrame with transformed columns and the mapping of each col
        to the corresponding LabelEncoder used to transform it, which is needed
        to reverse the transformation later on.
        """

        label_encoder_per_col_map = {}

        for col in columns_to_transform:
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            label_encoder_per_col_map[col] = label_encoder

        return df, label_encoder_per_col_map

    def _train_and_execute_model(
        self,
        df: pd.DataFrame,
        label_encoder_per_col_map: Dict[str, LabelEncoder],
    ) -> Tuple[pd.DataFrame, str]:
        """
        Train the model and execute it. The ML Algorithm used for this is the
        Random Forest Classifier.
        :return: Original dataset with the new prediction in the last row, for
        presentation purposes, along with the prediction value itself.
        """

        # prepare train and predict input data
        x_train = df.drop(["cycle", "date"], axis=1)  # features
        y_train = df["cycle"]  # prediction column
        x_pred = x_train.iloc[-1:]  # prediction dataset (last row only)
        x_train = x_train.iloc[:1857]  # drop last (prediction) row
        y_train = y_train.iloc[:1857]  # drop last (prediction) row

        # parameters
        n_estimators = 256  # above 128 trees, the RoI is negligible
        # TODO experiment for max_features = 1-x_train.shape[1]
        max_features = int(sqrt(x_train.shape[1])) + 1

        # initialize and train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            n_jobs=-1,  # -1 to ensure all CPU cores are utilized
        )
        model.fit(x_train, y_train)

        # execute the model to generate a prediction
        # TODO use predict_proba to get probs for each prediction class
        prediction = model.predict(x_pred)

        # add the prediction back to the original dataset
        df.loc[df.index[-1], "cycle"] = prediction

        # undo the discretionary column transformations
        for col, label_encoder in label_encoder_per_col_map.items():
            df[col] = label_encoder.inverse_transform(df[col])

        return df, prediction

    def _train_and_test_model(
        self,
        df: pd.DataFrame,
        label_encoder_per_col_map: Dict[str, LabelEncoder],
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Train and test the model. The ML Algorithm used for this is the Random
        Forest Classifier.
        :return: Original dataset with the predicted values, alongside test
        accuracy and train accuracy values.
        """

        # prepare train and test input data
        x = df.drop(["cycle"], axis=1)  # features
        y = df["cycle"]  # prediction column
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,  # roughly from 2007 onward
            random_state=None,
            shuffle=False,
            stratify=None,
        )
        x_train = x_train.drop(["date"], axis=1)
        x_test = x_test.drop(["date"], axis=1)

        # parameters
        n_estimators = 256  # above 128 trees, the RoI is negligible
        # TODO experiment for max_features = 1-x_train.shape[1]
        max_features = int(sqrt(x_train.shape[1]))

        # initialize and train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            n_jobs=-1,  # -1 to ensure all CPU cores are utilized
        )
        model.fit(x_train, y_train)

        # test the model
        train_acc = model.score(x_train, y_train)
        test_acc = model.score(x_test, y_test)
        print(
            f"Accuracy on training set for n_estimators={n_estimators} max_features={max_features}: {train_acc}"
        )
        print(
            f"Accuracy on test set for n_estimators={n_estimators} max_features={max_features}: {test_acc}"
        )
        print(f"Feature importances: {model.feature_importances_}")

        def _plot_feature_importances(tree):
            n_features = x_train.shape[1]
            plt.barh(np.arange(n_features), tree.feature_importances_, align="center")
            plt.yticks(np.arange(n_features), list(x_train.columns))
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
            plt.ylim(-1, n_features)
            plt.show()

        _plot_feature_importances(model)

        # undo the discretionary column transformations
        for col, label_encoder in label_encoder_per_col_map.items():
            df[col] = label_encoder.inverse_transform(df[col])

        return df, test_acc, train_acc

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename the columns for presentation purposes."""

        return df.rename(
            columns={
                "date": "Date",
                "cycle": "Cycle",
                "trend": "Trend Regime",
                "momo": "Momo Regime",
                "interest_rates_lvl": "Interest Rate Level",
                "interest_rates_direction": "Interest Rate Direction",
                "gdp_growth_vol": "Real GDP Growth Volatility",
                "cpi_roc": "CPI Rate of Change",
                "cpi_vol": "CPI Volatility",
                "cpi_lvl": "CPI Level",
                "sahm_rule": "Sahm Rule",
                "ism_pmi": "ISM PMI",
                "ig_regime": "Inflation/Growth Regime",
                "pce_roc": "PCE Rate of Change",
                "pce_vol": "PCE Volatility",
                "pdi_roc": "Private Domestic Investment Rate of Change",
                "pdi_vol": "Private Domestic Investment Volatility",
                "pdi": "Private Domestic Investment YoY",
                "pce": "PCE YoY",
                "corporate_profits": "Corporate profits YoY",
                "corporate_profits_vol": "Corporate profits Volatility",
            },
            inplace=False,
        )
    
    def _check_multicollinearity(self, df: pd.DataFrame):
        # independent variables DataFrame
        x = df.drop(["date", "cycle"], axis=1)

        # VIF DataFrame
        vif_data = pd.DataFrame()
        vif_data["feature"] = x.columns

        # calculate VIF for each feature
        vif_data["vif"] = [
            variance_inflation_factor(x.values, i)
            for i in range(len(x.columns))
        ]

        print(vif_data)
