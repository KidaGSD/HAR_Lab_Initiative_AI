import boto3
import os

def list_s3_keys():
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="us-west-1"
    )
    s3 = session.client('s3')
    bucket = "ego4d-consortium-sharing"
    prefix = "public/v1/imu/canonical/"
    
    print(f"Listing keys in {bucket}/{prefix}...")
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(obj['Key'])
        else:
            print("No objects found.")
            
        # Try v2 path just in case
        prefix_v2 = "public/v2/imu/"
        print(f"\nListing keys in {bucket}/{prefix_v2}...")
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix_v2, MaxKeys=10)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(obj['Key'])
        else:
            print("No objects found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_s3_keys()
