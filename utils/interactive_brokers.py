import logging
import threading
import time
from datetime import datetime
from math import sqrt
from typing import List

import pandas as pd

import settings
from utils.ibkr.contracts import Option
from utils.ibkr.wrapper import IB


def run_message_loop(ib):
    """
    Used as a Thread routine for running the Message Loop.
    :param ib: Object instance of utils.ibkr.wrapper.IB.
    """

    ib.run()


class InteractiveBrokersClient:
    def __init__(self):
        self._port = settings.IB_GATEWAY_PORT
        self._host = settings.IB_GATEWAY_HOST
        self._client_id = int(time.time())
        self._ib = None  # set by self.connect_to_tws()
        self._msg_loop_thread = None  # set by self.connect_to_tws()

    def connect_to_tws(self):
        """
        Connect to TWS and start the Message Loop Thread.

        Every call to this method should be followed by a call to
        self.disconnect_from_tws().
        """

        ib = IB()
        ib.connect(self._host, self._port, self._client_id)

        # start Message Loop Thread (stops when self.disconnect_from_tws()
        # is called)
        self._msg_loop_thread = threading.Thread(
            target=run_message_loop,
            args=(ib,),
            daemon=True,
        )
        self._msg_loop_thread.start()

        self._ib = ib

    def disconnect_from_tws(self):
        """
        Call this at the end of each request, to make sure that the Message Loop
        Thread is stopped and doesn't continue to run in the background.

        Basically for every call to self.connect_to_tws() there should be a call
        to this method.
        """

        # when this is called, the while loop inside run() will end and run()
        # will return, so the msg_loop_thread will stop as well
        if self._ib is not None:
            self._ib.disconnect()

    def get_market_implied_volatility(
        self,
        symbol: str,
        price: float,
        expiration_date: datetime,
    ) -> float:
        """
        Calculate Market Implied Volatility of the underlying based on
        ATM options for the specified expiration date.

        :param symbol: Symbol of the underlying, i.e. "SPY".
        :param price: Current price, cause these are ATM options or in case we
            want to test different scenarios, any desired strike price.
        :param expiration_date: datetime object, i.e. datetime.today().
        :return: Market Implied Volatility.
        """

        try:
            self.connect_to_tws()

            expiration = expiration_date.strftime("%Y%m%d")
            strike = int(price)

            contract = Option(symbol, expiration, strike, "C", "SMART")
            details = self._ib.reqContractDetails(int(time.time()), contract)

            self._ib.reqMarketDataType(2)

            option_price = self._ib.reqMktData_last_price(
                int(time.time()),
                details.contract,
                "",
                False,
                False,
                [],
            )

            market_iv = 0

            # call multiple times cause sometimes we get botched values on the
            # first call
            for i in range(2):
                # https://interactivebrokers.github.io/tws-api/option_computations.html
                iv = self._ib.calculateImpliedVolatility(
                    int(time.time()),
                    details.contract,
                    option_price,
                    strike,
                    [],
                )
                market_iv = iv / sqrt(252)

                time.sleep(1)

            return market_iv
        finally:
            self.disconnect_from_tws()

    def get_net_gamma(
        self,
        symbol: str,
        strike_prices: List[int],
        expiration_dates: List[datetime],
    ) -> pd.DataFrame:
        try:
            self.connect_to_tws()

            df = pd.DataFrame()  # output

            for expiration_date in expiration_dates:
                expiration = expiration_date.strftime("%Y%m%d")

                for strike_price in strike_prices:
                    strike = int(strike_price)

                    # call
                    contract = Option(symbol, expiration, strike, "C", "SMART")
                    details = self._ib.reqContractDetails(int(time.time()), contract)

                    # call gamma
                    self._ib.reqMarketDataType(2)
                    greeks = self._ib.reqMktData_greeks(
                        int(time.time()),
                        details.contract,
                        "",
                        True,
                        False,
                        [],
                    )
                    call_gamma = greeks["gamma"]

                    # call open-interest
                    self._ib.reqMarketDataType(2)
                    call_open_interest = self._ib.reqMktData_call_option_open_interest(
                        int(time.time()),
                        details.contract,
                        "mdoff,101",
                        False,
                        False,
                        [],
                    )
                    if call_open_interest is None:
                        continue

                    call_gex = call_gamma * call_open_interest * 100 * strike

                    # put
                    contract = Option(symbol, expiration, strike, "P", "SMART")
                    details = self._ib.reqContractDetails(int(time.time()), contract)

                    # put gamma
                    self._ib.reqMarketDataType(2)
                    greeks = self._ib.reqMktData_greeks(
                        int(time.time()),
                        details.contract,
                        "",
                        True,
                        False,
                        [],
                    )
                    put_gamma = greeks["gamma"]

                    # put open-interest
                    self._ib.reqMarketDataType(2)
                    put_open_interest = self._ib.reqMktData_put_option_open_interest(
                        int(time.time()),
                        details.contract,
                        "mdoff,101",
                        False,
                        False,
                        [],
                    )
                    if put_open_interest is None or put_gamma is None:
                        continue

                    put_gex = put_gamma * put_open_interest * 100 * strike * -1

                    data = {
                        "expiration_date": expiration_date,
                        "strike": strike,
                        "net_gamma": round(call_gex + put_gex, 2),
                        "call_gex": call_gex,  # positive number
                        "put_gex": put_gex,  # negative number
                    }
                    print(data)
                    logging.info(str(data))

                    # df = df.append(data, ignore_index=True)
                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                    time.sleep(1.5)

            return df
        finally:
            self.disconnect_from_tws()
