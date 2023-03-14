from typing import List, Dict

import requests
from requests import Response

import settings


class EETCVaultClient:
    def __init__(self):
        self._base_url = "https://eetc-vault-service-ciyunepj4a-ue.a.run.app"
        self._api_key = settings.EETC_VAULT_API_KEY

    def _send_http_request(self, url: str, params: dict) -> Response:
        response = requests.get(url, params=params, headers={"API_KEY": self._api_key})

        if response.status_code != 200:
            response.raise_for_status()

        return response

    def get_current_positions(self) -> List[Dict]:
        """
        Get current positions from Google Sheets via EETC Vault REST API.
        """

        url = self._base_url + "/api/trading/current_positions"
        response = self._send_http_request(url, {})

        return response.json()
