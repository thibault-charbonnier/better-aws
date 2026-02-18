from src.better_aws import AWS
import logging
import pandas as pd
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    force=True,
)
logger = logging.getLogger(__name__)

logger.info("Starting AWS operations...")

aws = AWS("s3admin", logger=logger, verbose=True)
aws.identity(print_info=True)

aws.s3.config(bucket="thib-quant")
# files = aws.s3.list(prefix="test/")
# print(files)

# # test_data = pd.read_csv(r"C:\Users\thibc\Downloads\ECB Data Portal_20260210153658.csv", sep=";")
# # print(test_data.head())
# # aws.s3.upload(test_data, key="test/test_pd_data.parquet", overwrite=True)
# # aws.s3.upload(r"C:\Users\thibc\Downloads\ECB Data Portal_20260210153658.csv", key="test/test_csv_data.csv", overwrite=True)

# # res = aws.s3.download("test/test_csv_data.csv", to=r"C:\Users\thibc\Downloads\test_csv_data.csv")
# # print(res)
# # res_2 = aws.s3.load("test/test_pd_data.parquet")
# # print(res_2)

aws.s3.exists("test/test_csv_datafefef.csv")
# aws.s3.delete("test/test_pd_data.parquet")
