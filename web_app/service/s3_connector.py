import boto3
from config.s3_config import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_REGION, AWS_S3_BUCKET_NAME

def s3_connection():
    s3 = boto3.client(
        service_name='s3',
        region_name=AWS_S3_BUCKET_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    return s3

def upload_image(s3, filepath, access_key):
    s3.upload_file(
        filepath,
        AWS_S3_BUCKET_NAME,
        access_key,
        ExtraArgs={'ContentType': "application/png", 'ACL': "public-read"}
    )

def download_image(s3, filepath, access_key):
    return s3.download_file(
        AWS_S3_BUCKET_NAME,
        access_key,
        filepath
    )