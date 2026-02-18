from __future__ import annotations
from typing import Optional, Any, Dict
from botocore.config import Config
import logging
import boto3
from .services.s3 import S3
import sys


class AWS:

    def __init__(self,
                 profile: Optional[str] = None,
                 region: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 verbose: bool = False,
                 retries: int = 10,
                 connect_timeout_s: int = 5,
                 read_timeout_s: int = 60):
        
        self.profile = profile
        self.region = region
        self.verbose = verbose
        self.retries = retries
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        
        self._config_logger(logger)

        self._session_cache: Optional[boto3.Session] = None
        self._s3: Optional[S3] = None

    @property
    def s3(self) -> S3:
        """
        Returns
        -------
        An instance of the S3 service wrapper for interacting with Amazon S3.
        """
        if self._s3 is None:
            self._s3 = S3(self)
        return self._s3

    # --------------------------------------------------------
    # |                   Internal Helpers                   |
    # --------------------------------------------------------

    def _config_logger(self, logger: Optional[logging.Logger]) -> None:
        """
        Configure the logger for the AWS instance.
            - If a logger is provided by the user, it will be used directly as it is.
            - If no logger is provided, a default logger will be created that outputs to stdout with INFO level.
            - The botocore.credentials logger is set to WARNING level to reduce noise from AWS credential loading.

        Parameters
        ----------
        logger: Optional[logging.Logger]
            An optional logger to use.
        """
        logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("better_aws._internal")
            self.logger.propagate = False
            self.logger.setLevel(logging.INFO)

            if not self.logger.handlers:
                h = logging.StreamHandler(sys.stdout)
                h.setLevel(logging.INFO)
                h.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(h)

    def _session(self) -> boto3.Session:
        """
        Get a boto3 Session object for AWS interactions.

        Returns
        -------
        boto3.Session
            A boto3 Session configured with the specified profile and region.
        """
        if self._session_cache is None:
            self._session_cache = boto3.Session(profile_name=self.profile, region_name=self.region)
        return self._session_cache

    def _config(self) -> Config:
        """
        Minimal botocore configuration for AWS clients.

        Returns
        -------
        botocore.config.Config
            A botocore Config object with the specified retry and timeout settings.
        """
        return Config(
            retries={"max_attempts": self.retries, "mode": "standard"},
            connect_timeout=self.connect_timeout_s,
            read_timeout=self.read_timeout_s,
        )

    # --------------------------------------------------------
    # |                   Exposed Methods                    |
    # --------------------------------------------------------

    def info(self, msg: str, *args: Any) -> None:
        """
        Log an message according to the logging configuration of the AWS instance.

        Parameters
        ----------
        msg: str
            The message to log.
        *args: Any
            Additional arguments to format the message with.
        """
        if self.verbose:
            self.logger.info(msg, *args)

    def identity(self, print_info: bool = False) -> Dict[str, Any]:
        """
        Get the AWS identity of the current session.

        Parameters
        ----------
        print_info: bool, default False
            If True, print the identity information to stdout.

        Returns
        -------
        dict
            The AWS identity information, including the ARN, account ID, and user ID.
        """
        sts = self._session().client("sts", config=self._config())
        id = sts.get_caller_identity()

        if print_info:
            self.info("AWS Identity:")
            self.info(f"  ARN: {id.get('Arn')}")
            self.info(f"  Account ID: {id.get('Account')}")
            self.info(f"  User ID: {id.get('UserId')}")

        return id