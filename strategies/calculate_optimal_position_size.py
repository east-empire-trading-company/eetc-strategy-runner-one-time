from eetc_utils.finance import optimal_leverage_kelly_criterion

from runner.strategy import OneTimeStrategy


class CalculateOptimalPositionSize(OneTimeStrategy):
    async def _algorithm(self):
        """
        Fetch current positions and calculate optimal position sizes for them.
        Then send out alerts for those whose position size is more than 5%
        different from the optimal size.

        We have to make this async due to the Telegram runner code which needs
        to be awaited.
        """

        default_position_size = 2000

        portfolio = self.parent_context.vault_client.get_current_positions()

        for position in portfolio:
            symbol = position["symbol"]

            price_data = self.parent_context.data_client.get_price_data(symbol=symbol)

            current_position_size = round(
                int(float(position["amount"])) * float(position["price"]), 2
            )

            optimal_leverage = optimal_leverage_kelly_criterion(
                price_data,
                position["start_date"],
                position["position_type"],
                use_fractional_kelly=True,
                use_garch=True,
            )
            optimal_position_size = round(optimal_leverage * default_position_size, 2)

            change_pct = (
                (float(optimal_position_size) - current_position_size)
                / current_position_size
            ) * 100

            # don't bother if the difference is less than 5%
            if -5 < change_pct < 5:
                continue

            message = (
                f"{symbol} - Current Position Size is {current_position_size}, "
                f"but Optimal Position Size should be {optimal_position_size}"
            )

            # TODO automatically rebalance position instead of sending alerts
            self.parent_context.email_client.send_email(
                subject=f"{symbol} - Position Size Recommendation",
                body_html=message,
                recipients=["eastempiretradingcompany2019@gmail.com"],
            )
            await self.parent_context.telegram_channel.send_message(message)
