from ibapi.contract import Contract


class Option(Contract):
    def __init__(
        self,
        ticker: str,
        expiration: str,
        strike: int,
        right: str = "C",
        exchange: str = "SMART",
        currency: str = "USD",
    ):
        super().__init__()
        self.symbol = ticker
        self.secType = "OPT"
        self.lastTradeDateOrContractMonth = expiration
        self.strike = strike
        self.right = right
        self.exchange = exchange
        self.currency = currency
        self.multiplier = "100"
        self.includeExpired = True
