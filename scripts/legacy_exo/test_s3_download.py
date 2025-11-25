import boto3
import os
import sys

def test_download():
    s3_path = "s3://ego4d-unict-milan/public/v1/imu/canonical/3e08beb0-9108-4e77-b2ae-80f91ceac474.csv"
    bucket = "ego4d-unict-milan"
    key = "public/v1/imu/canonical/3e08beb0-9108-4e77-b2ae-80f91ceac474.csv"
    
    print(f"Attempting to download {s3_path}...")
    
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="us-west-1" # Ego4D is usually in us-west-1
    )
    
    s3 = session.client('s3')
    
    try:
        s3.download_file(bucket, key, "test_download.csv")
        print("Download successful!")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    test_download()
