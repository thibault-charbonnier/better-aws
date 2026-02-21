from __future__ import annotations
import sys
import boto3
import logging
import botocore.session
from .services import S3
from dotenv import dotenv_values
from botocore.config import Config
from typing import Optional, Any, Dict


class AWS:

    def __init__(self,
                 profile: Optional[str] = None,
                 region: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 verbose: bool = False,
                 retries: int = 10,
                 connect_timeout_s: int = 5,
                 read_timeout_s: int = 60,
                 *,
                 credentials_file: Optional[str] = None,
                 config_file: Optional[str] = None,
                 env_file: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None
                 ):
        
        self.profile = profile
        self.region = region
        self.verbose = verbose
        self.retries = retries
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        
        self.credentials_file = credentials_file
        self.config_file = config_file
        self.env_file = env_file
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token

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

    def _read_env_file(self) -> Dict[str, str]:
        """
        Env file reader.

        Returns
        -------
        dict
            A dictionary containing the key-value pairs from the env file.
        """
        values = dotenv_values(self.env_file)
        return {k: v for k, v in values.items() if k and v is not None}
    
    def _session(self) -> boto3.Session:
        """
        Get a boto3 Session object for AWS interactions.

        The session is created based on the following priority:
            1) Static credentials provided directly to the AWS constructor.
            2) An env file containing AWS credentials and region.
            3) Custom credentials or/and config files specified by the user.
            4) Default boto3 session using the specified profile and region
               See : https://docs.aws.amazon.com/boto3/latest/guide/credentials.html

        Returns
        -------
        boto3.Session
            A boto3 Session configured with the specified profile and region.
        """
        
        if self._session_cache is not None:
            return self._session_cache

        # 1) Static credentials
        if self.aws_access_key_id and self.aws_secret_access_key:
            self._session_cache = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.region,
            )
            return self._session_cache

        # 2) Env file
        if self.env_file:
            env = self._read_env_file()
            ak = env.get("AWS_ACCESS_KEY_ID")
            sk = env.get("AWS_SECRET_ACCESS_KEY")
            tok = env.get("AWS_SESSION_TOKEN")
            reg = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or self.region

            if not ak or not sk:
                raise ValueError(f"env_file provided but missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: {self.env_file}")

            self._session_cache = boto3.Session(
                aws_access_key_id=ak,
                aws_secret_access_key=sk,
                aws_session_token=tok,
                region_name=reg,
            )
            return self._session_cache
        
        # 3) Custom config files
        if self.credentials_file or self.config_file:
            bc = botocore.session.get_session()
            if self.credentials_file:
                bc.set_config_variable("credentials_file", self.credentials_file)
            if self.config_file:
                bc.set_config_variable("config_file", self.config_file)

            self._session_cache = boto3.Session(
                botocore_session=bc,
                profile_name=self.profile,
                region_name=self.region,
            )
            return self._session_cache

        # 4) Default boto3 session
        self._session_cache = boto3.Session(
            profile_name=self.profile,
            region_name=self.region,
        )
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

    def reset_session(self) -> None:
        """
        Reset the cached boto3 session.
        """
        self._session_cache = None

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