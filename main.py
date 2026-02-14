from src.better_aws.aws_wrapper import AWS
import logging
logging.basicConfig(level=logging.INFO)

aws = AWS("s3admin")
aws.identity()
