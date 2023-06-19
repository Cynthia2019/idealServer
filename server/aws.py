import boto3
from botocore.config import Config
import os
session = boto3.Session(
    profile_name='ideal-server-profile'
)
config = Config(
    region_name = 'us-east-2',
    signature_version = 'v4',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)
s3 = session.client('s3', config=config)