import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd

class SavingReading:
    
    def __init__(self,bucket_name='insight-prj-bucket',region=None):
        # Create an S3 client
        self.s3 = boto3.client('s3')
        #Try to validate bucket
        response = self.s3.list_buckets()# Call S3 to list current buckets
        buckets = [bucket['Name'] for bucket in response['Buckets']] # Get a list of all bucket names from the response
        if bucket_name in buckets:
            self.bucket_save = True
            self.s3_bucket_name = bucket_name
        else:
            self.bucket_save = True
            logging.warning(f'Couldnt find bucket {bucket_name}! All data will be saved locally')
            self.s3_bucket_name = None

    def upload_to_s3(self,filename:str,full_local_path:str,full_s3_path:str,region=None):
        if self.bucket_save:
            try:
                self.s3.upload_file(full_local_path+"/"+filename, self.s3_bucket_name, full_s3_path+"/"+filename)
            except ClientError as e:
                logging.error(f'Error uploading: {e}')
            except Exception as err:
                logging.error(f'Error uploading: {err}')
        else:
            logging.warning('Cant save to S3. No bucket')


    #TODO: delete local copies after uploaded
    def save_to_s3(self,obj,save_name:str,full_save_dir:str):
        if isinstance(obj,pd.DataFrame):
            self.save_df_to_csv(obj,save_name,full_save_dir)
            self.upload_to_s3(save_name+".csv",full_save_dir,full_save_dir)
        else:
            self.save_pickle(obj,full_save_dir+"/"+save_name+".p")
            self.upload_to_s3(save_name+".p",full_save_dir,full_save_dir)

    def save_df_to_csv(self,df:pd.DataFrame,save_name:str,full_save_dir:str):
        df.to_csv(full_save_dir + "/" + save_name)

    def save_pickle(self,obj,loc:str):
        try:
            f = open(loc,"rb")
            pickle.dump(obj,f)
        except NameError:
            print('Couldn\'t open save loc')

    def save(self,obj,save_name:str,full_save_dir:str, bool_save_s3=False):
        if bool_save_s3:
            self.save_to_s3(obj,save_name,full_save_dir)
        elif isinstance(obj,pd.DataFrame):
            self.save_df_to_csv(obj,save_name,full_save_dir)
        else:
            logging.info('Using pickle by default')
            self.save_pickle(obj,full_save_dir+"/"+save_name+".p")

    #TODO: implement reader. Move pickle.loads in other modules here