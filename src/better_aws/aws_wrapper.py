from __future__ import annotations
from typing import Optional, Any, Dict
from botocore.config import Config
import logging
import boto3


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
        self.logger = logger or logging.getLogger("better_aws")
        self.verbose = verbose
        self.retries = retries
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s

    def session(self) -> boto3.Session:
        return boto3.Session(profile_name=self.profile, region_name=self.region)

    def _config(self) -> Config:
        return Config(
            retries={"max_attempts": self.retries, "mode": "standard"},
            connect_timeout=self.connect_timeout_s,
            read_timeout=self.read_timeout_s,
        )

    def identity(self) -> Dict[str, Any]:
        sts = self.session().client("sts", config=self._config())
        id = sts.get_caller_identity()
        self.logger.debug(f"UserId: {id['UserId']}")
        return id