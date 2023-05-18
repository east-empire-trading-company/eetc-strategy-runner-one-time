import asyncio
import logging
from typing import List, Type

from eetc_data_client.client import EETCDataClient

import settings
from runner.strategy import StrategyBase
from utils.email import EmailClient
from utils.interactive_brokers import InteractiveBrokersClient
from utils.telegram import EETCTelegramChannel
from utils.vault import EETCVaultClient


class Context:
    def __init__(self, strategies: List[Type[StrategyBase]] = None):
        if not strategies:
            strategies = []

        self._strategies = [s(context=self) for s in strategies]
        self._zmq_context = None  # TODO create ZMQ context here
        self._telegram_channel = EETCTelegramChannel()
        self._data_client = EETCDataClient(api_key=settings.EETC_API_KEY)
        self._vault_client = EETCVaultClient()
        self._email_client = EmailClient()
        self._ibkr_client = InteractiveBrokersClient()

        self.shared_data = {}  # share data between strategies

    @property
    def email_client(self) -> EmailClient:
        return self._email_client

    @property
    def strategies(self) -> List[StrategyBase]:
        return self._strategies

    @property
    def vault_client(self) -> EETCVaultClient:
        return self._vault_client

    @property
    def data_client(self) -> EETCDataClient:
        return self._data_client

    @property
    def telegram_channel(self) -> EETCTelegramChannel:
        return self._telegram_channel

    @property
    def ibkr_client(self) -> InteractiveBrokersClient:
        return self._ibkr_client

    def add_strategy(self, strategy: Type[StrategyBase]):
        self._strategies.append(strategy(context=self))

    def execute_strategies(self):
        """
        Execute strategies in the order they were added.

        This is important because a strategy may set some attributes on the
        parent context that might be used by subsequent strategies.

        For example, one strategy comes up with some positions to take, another
        one calculates optimal position sizes for them and executes them. This
        is much better than writing one big complex strategy.
        """

        for strategy in self._strategies:
            logging.info(f"Executing strategy: {strategy.__class__}")
            strategy.execute()

    async def execute_strategies_async(self):
        """
        Execute strategies in parallel.
        """

        for strategy in self._strategies:
            logging.info(f"Executing strategy: {strategy.__class__}")
            await strategy.execute_async()
            await asyncio.sleep(5)
