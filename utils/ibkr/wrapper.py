import queue
from _decimal import Decimal

from ibapi.client import EClient
from ibapi.common import TickerId, TagValueList, TickAttrib
from ibapi.contract import Contract, ContractDetails
from ibapi.ticktype import TickType
from ibapi.wrapper import EWrapper


class IB(EWrapper, EClient):
    """
    Contains overriden methods from EWrapper and EClient. These methods have to
    be overriden, so we can send and receive messages in a way we want.
    """

    def __init__(self):
        EClient.__init__(self, self)
        self._response_queue = queue.Queue()

    def reqContractDetails(self, reqId: int, contract: Contract):
        """
        Call this function to download all details for a particular
        underlying. The contract details will be received via the
        contractDetails() method.
        """

        super().reqContractDetails(reqId, contract)
        # will block until self.contract_details returns
        while data := self._response_queue.get(block=True, timeout=None):
            if isinstance(data, dict) and "contractDetails" in data:
                return data["contractDetails"]

        return None

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        """
        Receives the full contract's definitions. This method will return all
        contracts matching the requested via EEClientSocket::reqContractDetails.
        For example, one can obtain the whole option chain with it.
        """

        if self._response_queue.empty():
            self._response_queue.put({"contractDetails": contractDetails})

    def reqMktData_greeks(
        self,
        reqId: TickerId,
        contract: Contract,
        genericTickList: str,
        snapshot: bool,
        regulatorySnapshot: bool,
        mktDataOptions: TagValueList,
    ):
        """
        Call this function to request market data. The market data will be
        returned by the tickPrice and tickSize events.
        """

        super().reqMktData(
            reqId,
            contract,
            genericTickList,
            snapshot,
            regulatorySnapshot,
            mktDataOptions,
        )
        # will block until self.contract_details returns
        while data := self._response_queue.get(block=True, timeout=None):
            if isinstance(data, dict) and "gamma" in data:
                super().cancelMktData(reqId)
                return data

        return None

    def reqMktData_last_price(
        self,
        reqId: TickerId,
        contract: Contract,
        genericTickList: str,
        snapshot: bool,
        regulatorySnapshot: bool,
        mktDataOptions: TagValueList,
    ):
        """
        Call this function to request market data. The market data will be
        returned by the tickPrice and tickSize events.
        """

        super().reqMktData(
            reqId,
            contract,
            genericTickList,
            snapshot,
            regulatorySnapshot,
            mktDataOptions,
        )
        # will block until self.contract_details returns
        while data := self._response_queue.get(block=True, timeout=None):
            if isinstance(data, dict) and "last_price" in data:
                super().cancelMktData(reqId)
                return data["last_price"]

        return None

    def reqMktData_call_option_open_interest(
        self,
        reqId: TickerId,
        contract: Contract,
        genericTickList: str,
        snapshot: bool,
        regulatorySnapshot: bool,
        mktDataOptions: TagValueList,
    ):
        """
        Call this function to request market data. The market data will be
        returned by the tickPrice and tickSize events.
        """

        super().reqMktData(
            reqId,
            contract,
            genericTickList,
            snapshot,
            regulatorySnapshot,
            mktDataOptions,
        )
        # will block until self.contract_details returns
        while data := self._response_queue.get(block=True, timeout=None):
            if isinstance(data, dict) and "call_option_open_interest" in data:
                super().cancelMktData(reqId)
                return data["call_option_open_interest"]

        return None

    def reqMktData_put_option_open_interest(
        self,
        reqId: TickerId,
        contract: Contract,
        genericTickList: str,
        snapshot: bool,
        regulatorySnapshot: bool,
        mktDataOptions: TagValueList,
    ):
        """
        Call this function to request market data. The market data will be
        returned by the tickPrice and tickSize events.
        """

        super().reqMktData(
            reqId,
            contract,
            genericTickList,
            snapshot,
            regulatorySnapshot,
            mktDataOptions,
        )
        # will block until self.contract_details returns
        while data := self._response_queue.get(block=True, timeout=None):
            if isinstance(data, dict) and "put_option_open_interest" in data:
                super().cancelMktData(reqId)
                return data["put_option_open_interest"]

        return None

    def tickPrice(
        self,
        reqId: TickerId,
        tickType: TickType,
        price: float,
        attrib: TickAttrib,
    ):
        """
        Market data tick price callback. Handles all price related ticks.
        """

        # https://interactivebrokers.github.io/tws-api/tick_types.html
        if tickType in {4, 9} and self._response_queue.empty():
            self._response_queue.put({"last_price": price})

    def tickSize(self, reqId: TickerId, tickType: TickType, size: Decimal):
        """
        Market data tick size callback. Handles all price related ticks.
        """

        # https://interactivebrokers.github.io/tws-api/tick_types.html
        if tickType == 27:
            self._response_queue.put({"call_option_open_interest": int(size)})

        if tickType == 28:
            self._response_queue.put({"put_option_open_interest": int(size)})

    def calculateImpliedVolatility(
        self,
        reqId: TickerId,
        contract: Contract,
        optionPrice: float,
        underPrice: float,
        implVolOptions: TagValueList,
    ):
        """
        Call this function to calculate volatility for a supplied option price
        and underlying price. Result will be delivered via
        EWrapper.tickOptionComputation()
        """

        super().calculateImpliedVolatility(
            reqId, contract, optionPrice, underPrice, implVolOptions
        )
        # will block until self.contract_details returns
        while data := self._response_queue.get(block=True, timeout=None):
            if isinstance(data, dict) and "impliedVol" in data:
                super().cancelCalculateImpliedVolatility(reqId)
                return data["impliedVol"]

    def tickOptionComputation(
        self,
        reqId: TickerId,
        tickType: TickType,
        tickAttrib: int,
        impliedVol: float,
        delta: float,
        optPrice: float,
        pvDividend: float,
        gamma: float,
        vega: float,
        theta: float,
        undPrice: float,
    ):
        """
        This function is called when the market in an option or its underlier
        moves. TWS's option model volatilities, prices, and deltas, along with
        the present value of dividends expected on that options underlier are
        received.
        """

        if tickType in {13, 53} and self._response_queue.empty():
            greeks = {
                "impliedVol": impliedVol,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
            }
            self._response_queue.put(greeks)
