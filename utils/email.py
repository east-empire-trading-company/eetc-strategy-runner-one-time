import logging
from typing import List

import boto3
from botocore.exceptions import ClientError

import settings


class EmailClient:
    def __init__(self, ses_client=None):
        self._aws_ses_client = (
            ses_client
            if ses_client
            else boto3.client(
                "ses",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
        )

    def send_email(self, subject: str, body_html: str, recipients: List[str]):
        assert body_html, "HTML body is required"

        sender = f"East Empire Trading Company <{settings.AWS_EMAIL_SENDER}>"
        encoding = "UTF-8"

        logging.info(f"Sending email via AWS SES to: {str(recipients)}")

        try:
            self._aws_ses_client.send_email(
                Destination={"ToAddresses": recipients},
                Message={
                    "Body": {"Html": {"Charset": encoding, "Data": body_html}},
                    "Subject": {"Charset": encoding, "Data": subject},
                },
                Source=sender,
            )
        except ClientError as e:
            logging.error(f"Failed to send email via AWS SES: {e}")
