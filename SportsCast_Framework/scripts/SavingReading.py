import logging
import boto3
from botocore.exceptions import *
import pandas as pd
import pickle
from pickle import *

class SavingReading:
    
    def __init__(self,bucket_name='insight-prj-bucket',region=None):
        try:
            # Create an S3 client
            self.s3 = boto3.client('s3')
            #Try to validate bucket
            response = self.s3.list_buckets()# Call S3 to list current buckets
            buckets = [bucket['Name'] for bucket in response['Buckets']] # Get a list of all bucket names from the response
            if bucket_name in buckets:
                self.bucket_save_read = True
                self.s3_bucket_name = bucket_name
            else:
                self.bucket_save_read = False
                logging.warning(f'Couldnt find bucket {bucket_name}! All data will be saved/read-from locally')
                self.s3_bucket_name = None
        except BotoCoreError as err:
            self.bucket_save_read = False
            logging.warning(f'Couldnt start S3: {err}! All data will be saved/read-from locally')
        except Exception as err:
            self.bucket_save_read = False
            logging.warning(f'Other error in creating a saver-reader: {err}! All data will be saved/read-from if possible')

    def upload_to_s3(self,filename:str,full_local_dir:str,full_s3_dir:str,region=None):
        if self.bucket_save_read:
            try:
                self.s3.upload_file(full_local_dir+"/"+filename, self.s3_bucket_name, full_s3_dir+"/"+filename)
                return True
            except ClientError as e:
                logging.error(f'Error uploading: {e}')
                return None
            except Exception as err:
                logging.error(f'Error uploading: {err}')
                return None
        else:
            logging.warning('Cant save to S3. No bucket')
            return False

    #TODO: delete local copies after uploaded
    def save_to_s3(self,obj,save_name:str,full_save_dir:str):
        try:
            if isinstance(obj,pd.DataFrame):
                return self.save_df_to_csv(obj,save_name,full_save_dir) and \
                        self.upload_to_s3(save_name,full_save_dir,full_save_dir)
            else:
                return self.save_pickle(obj,full_save_dir+"/"+save_name) and \
                        self.upload_to_s3(save_name,full_save_dir,full_save_dir)
        except Exception as err:
            logging.error(f'Error: {err}')
            return None


    #TODO: test overwriting
    def save_df_to_csv(self,df:pd.DataFrame,save_name:str,full_save_dir:str):
        try:
            df.to_csv(full_save_dir + "/" + save_name + ".csv")
            return True
        except Exception as err:
            logging.error(f'Error: {err}')
            return None

    def save_pickle(self,obj,loc:str):
        try:
            f = open(loc+".p","wb")
            pickle.dump(obj,f)
            return True
        except NameError as err:
            logging.error(f'Couldn\'t open save loc {loc+".p"}. Error: {err}')
            return None
        except PickleError as err:
            logging.error(f'Couldn\'t save. Error pickling object: {err}')
            return None

    def save(self,obj,save_name:str,full_save_dir:str, bool_save_s3=False):
        if bool_save_s3:
            return self.save_to_s3(obj,save_name,full_save_dir) 
        elif isinstance(obj,pd.DataFrame):
            return self.save_df_to_csv(obj,save_name,full_save_dir)
        else:
            logging.info('Using pickle by default')
            return self.save_pickle(obj,full_save_dir+"/"+save_name)