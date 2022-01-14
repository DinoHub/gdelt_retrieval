#!/usr/bin/env python
# coding: utf-8
# Adapted from https://github.com/cshikai/onboarding_demo/blob/main/model/src/aiplatform/s3utility.py
import os
import sys
import time
import boto3
import zipfile
import requests
from io import BytesIO
from typing import Union
from pathlib import Path
from botocore.client import Config
from boto3.s3.transfer import TransferConfig

class S3Utils():
    
    MAX_ATTEMPT = 5 # No. attempts to connect to S3
    WAIT_TIME = 1   # How long to wait per connection attempt (seconds)

    def __init__(
        self,
        bucket: str,
        endpoint_url: str, 
        aws_access_key_id: str,
        aws_secret_access_key: str,
        upload_multi_part=False,
        signature_version: str = 's3v4',
        region_name: str = 'us-east-1', 
        verify: Union[bool, str] = '/usr/share/ca-certificates/extra/ca.dsta.ai.crt',
        ) -> None:

        '''
        Parameters
        ----------
        bucket : str
            Main S3 Bucket to link to
        endpoint_url : str
            Endpoint URL to private S3 storage
        aws_access_key_id : str
            S3 Access Key (Ask your SA for the key)
        aws_secret_access_key :  str
            S3 Secret Access Key (Ask your SA for the secret)
        signature_version : str
            Version of signature used for S3 Storage (default = 's3v4')
        region_name : str
            Region where the S3 storage is situated (default = 'us-east-1')
        verify : bool | str
            Verify with SSL
            if boolean: False - Do not verify
            if boolean: True - checks the environment variable REQUESTS_CA_BUNDLE for SSL Cert path
            Otherwise, insert SSL Cert path manually as string
        '''
        if upload_multi_part:
            self.config = TransferConfig(max_concurrency=5)
        else:
            GB = 1024**3
            self.config = TransferConfig(multipart_threshold=20*GB)

        self.bucket = bucket
        if type(verify) is str and not os.path.exists(verify):
            print(f'{verify} does not exist. Downloading from relevant repository...')

            OUTPUT_DIR = '/usr/share/ca-certificates/extra'
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            DOWNLOAD_URL = 'https://gitlab.dsta.ai/ai-platform/getting-started/raw/master/config/ca.dsta.ai.crt'
            r = requests.get(DOWNLOAD_URL)
            
            verify = os.path.join(OUTPUT_DIR, 'ca.dsta.ai.crt')
            open(verify, 'wb').write(r.content)
            print(f'{DOWNLOAD_URL} has been downloaded to {verify}')

        self.s3 = boto3.resource(
            's3',
            endpoint_url = endpoint_url,
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key,
            config = Config(signature_version = signature_version),
            region_name = region_name,
            verify = verify)

    def download_file(
        self, 
        s3_key: str, 
        local_path: str, 
        bucket: str = None,
        verbose: int = 1,
        ) -> None:

        '''
        Downloads a file from the S3 Storage to a local path.

        Parameters
        ----------
        s3_key : str
            Key of the file in the target bucket of the S3 Storage
        local_path : str
            Local path where the file will be downloaded to
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        verbose : int
            Decide whether to print logs [0: Print exceptions, 1: Print all] (default = 1)
        '''

        if bucket == None:
            bucket = self.bucket

        if verbose == 1:
            print(f'S3 - Downloading from s3://{bucket}/{s3_key} to {local_path}')

        attempt_no = 1
        success = False

        while attempt_no <= self.MAX_ATTEMPT and (not success):
            try:
                self.s3.Bucket(bucket).download_file(s3_key, local_path)
                success = True
                if verbose == 1:
                    print(f'S3 - File successfully downloaded to {local_path}')
            except Exception as e:
                if attempt_no == self.MAX_ATTEMPT:
                    print('S3 - Max attempts reached. Connection to S3 not successful. Error is shown below.')
                    print(e)
                    sys.exit(1)
                else:
                    print('S3 - connection failed. Retrying...')
                    time.sleep(self.WAIT_TIME)
                attempt_no += 1
        
        return None

    def download_zipped_file_v1(
        self, 
        s3_key: str, 
        local_path: str, 
        bucket: str = None,
        temp_dir: str = '/tmp',
        remove_temp: bool = True
        ) -> None:

        '''
        V1 Method - Extracts the contents of a S3 zipped file with 3 steps. (Time & Memory consuming, but stable)
        1. Downloads a zipped file from the S3 Storage to a temporary path.
        2. and then extracts the file into the target path.
        3. (Optional) Remove temporary zipped file.

        Parameters
        ----------
        s3_key : str
            Key of the file in the target bucket of the S3 Storage
        local_path : str
            Local path where the contents of the zipped file will be extracted to
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        temp_dir : str
            Temporary directory where the zipped file will be stored (default = '/tmp')
        remove_temp : bool
            Boolean to remove temporary zip file (default = True)
        '''

        # download file to temp directory
        temp_path = os.path.join(temp_dir, os.path.basename(s3_key))
        self.download_file(s3_key=s3_key, local_path=temp_path, bucket=bucket)

        # unzip file to target directory
        print(f'Local - Extracting contents from {temp_path} to {local_path}')
        with zipfile.ZipFile(temp_path, 'r') as ref:
            ref.extractall(local_path)
        print(f'Local - File contents have been unzipped to {local_path}')

        # remove temporary zipped file
        if remove_temp:
            os.remove(temp_path)
            print(f'Local - {temp_path} has been deleted.')

        return None

    def download_zipped_file_v2(
        self, 
        s3_key: str, 
        local_path: str, 
        bucket: str = None
        ) -> None:

        '''
        V2 Method - Reads and extracts the contents of a S3 zipped file as a bytes stream. 
        (Only works for smaller zip files (<= 2GB), otherwise will overflow)

        Parameters
        ----------
        s3_key : str
            Key of the file in the target bucket of the S3 Storage
        local_path : str
            Local path where the contents of the zipped file will be extracted to
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        '''

        if bucket == None:
            bucket = self.bucket

        # instantiate variable
        zipped_object = None

        attempt_no = 1
        success = False

        while attempt_no <= self.MAX_ATTEMPT and (not success):
            try:
                zipped_object = self.s3.Object(bucket_name=bucket, key=s3_key)
                success = True
            except Exception as e:
                if attempt_no == self.MAX_ATTEMPT:
                    print('S3 - Max attempts reached. Connection to S3 not successful. Error is shown below.')
                    print(e)
                    sys.exit(1)
                else:
                    print('S3 - connection failed. Retrying...')
                    time.sleep(self.WAIT_TIME)
                attempt_no += 1
        
        buffer = BytesIO(zipped_object.get()['Body'].read())
        ref = zipfile.ZipFile(buffer)
        for filename in ref.namelist():
            ref.extract(filename, local_path)

        print(f'Local - File contents from S3 have been extracted to {local_path}')
        return None

    def download_folder(
        self,
        s3_prefix: str,
        local_dir: str,
        bucket: str = None,
        verbose: int = 1,
        ) -> None:

        '''
        Downloads all files from S3 Storage with the prefix to a local directory.

        Parameters
        ----------
        s3_prefix : str
            Prefix of the files in the target bucket of the S3 Storage
        local_dir : str
            Local directory where the files will be downloaded to
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        verbose : int
            Decide whether to print logs [0: Only print exceptions, 1: Only print main method logs, 2: Print all] (default = 1)
        '''

        if bucket == None:
            bucket = self.bucket

        if verbose > 0:
            print(f'S3 - Downloading files with the prefix: {s3_prefix} to the local directory: {local_dir}')

        for obj in self.s3.Bucket(bucket).objects.filter(Prefix=s3_prefix):
            # skip directories
            if obj.key[-1] == '/':
                continue
            # path where each file will be locally downloaded to
            target_path = os.path.join(local_dir, os.path.relpath(obj.key, s3_prefix))
            # create nested directory if doesn't exist
            Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)
            # download file
            sub_verbose = 0 if verbose <= 1 else 1
            self.download_file(s3_key=obj.key, local_path=target_path, bucket=bucket, verbose=sub_verbose)
        
        if verbose > 0:
            print(f'S3 - All files successfully downloaded to {local_dir}')

        return

    def upload_file(
        self, 
        local_path: str,
        s3_key: str,
        bucket: str = None,
        verbose: int = 1,
        ) -> None:

        '''
        Upload a single file to S3
        
        Parameters
        ----------
        local_path : str
            Local path of the file for upload
        s3_key : str
            Key of the file to the target bucket of the S3 Storage
        bucket : str
            Manually state another bucket to upload to, otherwise download from main bucket (default = None)
        verbose : int
            Decide whether to print logs [0: Print exceptions, 1: Print all] (default = 1)
        '''

        if bucket == None:
            bucket = self.bucket

        if verbose == 1:
            print(f'S3 - Uploading from {local_path} to s3://{bucket}/{s3_key}')

        attempt_no = 1
        success = False

        while attempt_no <= self.MAX_ATTEMPT and (not success):
            try:
                self.s3.Bucket(bucket).upload_file(local_path, s3_key)
                success = True
                if verbose == 1:
                    print(f'S3 - File has been successfully uploaded to s3://{bucket}/{s3_key}')

            except Exception as e:
                if attempt_no == self.MAX_ATTEMPT:
                    print('S3 - Max attempts reached. Connection to S3 not successful. Error is shown below.')
                    print(e)
                    sys.exit(1)
                else:
                    print('S3 - connection failed. Retrying...')
                    time.sleep(self.WAIT_TIME)
                attempt_no += 1

        return None
    
    def upload_folder(
        self,
        local_dir: str,
        s3_prefix: str,
        bucket: str = None,
        verbose: int = 1,
        ) -> None:

        '''
        Uploads all files from a local directory to S3 Storage with the inserted prefix.

        Parameters
        ----------
        local_dir : str
            Local directory where the files will be uploaded from
        s3_prefix : str
            Prefix of the files in the target bucket of the S3 Storage
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        verbose : int
            Decide whether to print logs [0: Only print exceptions, 1: Only print main method logs, 2: Print all] (default = 1)
        '''

        if bucket == None:
            bucket = self.bucket

        if verbose > 0:
            print(f'S3 - Uploading files from the local directory: {local_dir} to S3 with prefix: {s3_prefix}')

        for dirx, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(dirx, file)
                s3_key = os.path.join(s3_prefix, os.path.relpath(local_path, local_dir))

                sub_verbose = 0 if verbose <= 1 else 1
                self.upload_file(local_path=local_path, s3_key=s3_key, bucket=bucket, verbose=sub_verbose)
        
        if verbose > 0:
            print(f'S3 - All files have been successfully uploaded to s3://{bucket}/{s3_prefix}')

        return
