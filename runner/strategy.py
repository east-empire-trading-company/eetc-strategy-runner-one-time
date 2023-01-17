from typing import Iterable


class StrategyBase:
    def __init__(self, context):
        self._data = dict()
        self._context = context
        self._orders = list()
        self._positions = list()

    def _algorithm(self):
        raise NotImplementedError()

    def execute(self):
        """
        Execute the self._algorithm() method defined by the developer.
        """

        self._algorithm()

    async def execute_async(self):
        """
        Execute the self._algorithm() method defined by the developer in case
        it has async code in it that needs to be awaited.
        """

        await self._algorithm()

    @property
    def orders(self):
        return self._orders

    @property
    def positions(self):
        return self._positions

    @property
    def parent_context(self):
        return self._context


class OneTimeStrategy(StrategyBase):
    """
    Inherit this class to write your own class that encapsulates a one-time
    strategy.
    """

    def __init__(self, context, zmq_context=None):
        super().__init__(context=context)
        # TODO create ZMQ context if not provided
        self._zmq_context = zmq_context if zmq_context else None

    def execute(self):
        """
        Execute the self._algorithm() method defined by the developer.
        """

        self._algorithm()

    async def execute_async(self):
        """
        Execute the self._algorithm() method defined by the developer in case
        it has async code in it that needs to be awaited.
        """

        await self._algorithm()

    def _algorithm(self):
        raise NotImplementedError()


class EventBasedStrategy(StrategyBase):
    """
    Inherit this class to write your own class that encapsulates an event-based
    strategy.

    Make sure to set the self._pub_sub_topics attribute.
    """

    def __init__(self, context, zmq_context=None):
        super().__init__(context=context)
        # make sure this attribute is set in your implementation
        self._pub_sub_topics: Iterable = None
        # TODO create ZMQ context if not provided
        self._zmq_context = zmq_context if zmq_context else None

    def pre_execution_checks(func):
        def inner(self):
            if self._pub_sub_topics is None:
                raise ValueError("Attribute self._pub_sub_topics is not defined.")

            func(self)

        return inner

    @pre_execution_checks
    def execute(self):
        """
        Execute the self._algorithm() method defined by the developer.
        """
        # TODO create sockets from ZMQ context and subscribe to all topics
        pass

    @pre_execution_checks
    async def execute_async(self):
        """
        Execute the self._algorithm() method defined by the developer in case
        it has async code in it that needs to be awaited.
        """
        # TODO create sockets from ZMQ context and subscribe to all topics
        pass

    def _algorithm(self):
        raise NotImplementedError()
