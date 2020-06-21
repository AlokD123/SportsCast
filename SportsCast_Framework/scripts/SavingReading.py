import logging
import boto3
from botocore.exceptions import *
import pandas as pd
import pickle
from pickle import *
import os
import glob

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

    def upload_to_s3(self,filename_with_ext:str,full_local_dir:str,full_s3_dir:str,region=None):
        if self.bucket_save_read:
            try:
                self.s3.upload_file(full_local_dir+"/"+filename_with_ext, self.s3_bucket_name, full_s3_dir+"/"+filename_with_ext)
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

    def save_to_s3(self,obj,save_name:str,full_save_dir:str):
        try:
            if isinstance(obj,pd.DataFrame):
                return self.save_df_to_csv(obj,save_name,full_save_dir) and \
                        self.upload_to_s3(save_name+".csv",full_save_dir,full_save_dir) and \
                        self.delete_local(save_name+".csv",full_save_dir)
            else:
                return self.save_pickle(obj,full_save_dir+"/"+save_name) and \
                        self.upload_to_s3(save_name+".p",full_save_dir,full_save_dir) and \
                        self.delete_local(save_name+".p",full_save_dir)
        except Exception as err:
            logging.error(f'Error: {err}')
            return None

    
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
        logging.debug(f"Arguments:\nsave_name={save_name},\nfull_save_dir={full_save_dir},\nbool_save_s3={bool_save_s3}")
        if bool_save_s3:
            ret = self.save_to_s3(obj,save_name,full_save_dir)
            if ret is True:
                logging.info(f'Saved {full_save_dir+save_name} to S3 with appropriate extension')
            return  
        elif isinstance(obj,pd.DataFrame):
            ret = self.save_df_to_csv(obj,save_name,full_save_dir)
            logging.info(f'Saved {full_save_dir+save_name+".csv"} to local disk')
            return ret
        else:
            logging.info('Using pickle by default')
            ret = self.save_pickle(obj,full_save_dir+"/"+save_name)
            if ret is True:
                logging.info(f'Saved {full_save_dir+save_name+".p"} to local disk')
            return ret


    def delete_local(self,filename_with_ext:str,full_save_dir:str):
        files_to_delete = glob.glob(full_save_dir+"/"+filename_with_ext)
        if len(files_to_delete)==0:
            logging.warn(f'The file {full_save_dir+"/"+filename_with_ext} to delete does not exist')
            return False
        else:
            logging.info(f'Deleting file: {files_to_delete[0]}')
            os.remove(files_to_delete[0])
            return True

    def create_dir(self,full_local_dir:str):
        try: 
            os.makedirs(full_local_dir, exist_ok=True)
            return True
        except OSError as error:
            logging.error(f"Directory {full_local_dir} can not be created: {error}")
            return None

    def download_from_s3(self,filename_with_ext:str,full_local_dir:str,full_s3_dir:str,region=None):
        if self.bucket_save_read:
            try:
                return self.create_dir(full_local_dir) and \
                        self.s3.download_file(self.s3_bucket_name, full_local_dir+"/"+filename_with_ext, full_s3_dir+"/"+filename_with_ext)
            except ClientError as e:
                logging.error(f'Error downloading: {e}')
                return None
            except Exception as err:
                logging.error(f'Error downloading: {err}')
                return None
        else:
            logging.warning('Cant download from S3. No bucket')
            return False

    
    def read_from_s3(self,file_ext:str,read_name:str,full_read_dir:str,pickle_obj_cls=None):
        try:
            ret = self.download_from_s3(read_name+file_ext,full_read_dir,full_read_dir)
            if file_ext == ".csv":
                df = (ret is True) and self.read_df_from_csv(read_name,full_read_dir)
                df = (df is not None) and (df is not False) and self.delete_local(read_name+".csv",full_read_dir)
                return df
            elif file_ext == ".p":
                obj = (ret is True) and self.read_pickle(full_read_dir+"/"+read_name,pickle_obj_cls)
                obj = (obj is not None) and (obj is not False) and self.delete_local(read_name+".p",full_read_dir)
                return obj
        except Exception as err:
            logging.error(f'Error: {err}')
            return None

    def read_df_from_csv(self,read_name:str,full_read_dir:str):
        try: 
            df = pd.read_csv(full_read_dir + "/" + read_name + ".csv")
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            return df
        except Exception as err:
            logging.error(f'Error: {err}')
            return None

    def read_pickle(self,loc:str,obj_cls=None):
        try:
            f = open(loc+".p","rb")
            obj = pickle.load(f)
            if (obj_cls is not None) and (not isinstance(obj,obj_cls)):
                logging.warning('Didnt unpickle expected class. Discarded obj')
                return False
            return obj
        except NameError as err:
            logging.error(f'Couldn\'t open read loc {loc+".p"}. Error: {err}')
            return None
        except PickleError as err:
            logging.error(f'Couldn\'t read. Error unpickling object: {err}')
            return None

    def read(self,file_ext:str,read_name:str,full_read_dir:str,pickle_obj_cls=None, bool_read_s3=False):
        logging.debug(f"Arguments:\nfile_ext: {file_ext},\nread_name={read_name},\nfull_read_dir={full_read_dir},\npickle_obj_cls={pickle_obj_cls},\nbool_read_s3={bool_read_s3}")
        if file_ext == ".csv":
            if bool_read_s3:
                df = self.read_from_s3(file_ext,read_name,full_read_dir)
            else:
                df = self.read_df_from_csv(read_name,full_read_dir)
            assert isinstance(df,pd.DataFrame), "CSV could not be read"
            logging.info(f'Read {full_read_dir+read_name+file_ext} from {"S3" if bool_read_s3 else "local disk"}')
            return df
        elif file_ext == ".p":
            if bool_read_s3:
                obj = self.read_from_s3(file_ext,read_name,full_read_dir,pickle_obj_cls)
            else:
                obj = self.read_pickle(full_read_dir+"/"+read_name,pickle_obj_cls)
            assert isinstance(obj,pickle_obj_cls) if pickle_obj_cls is not None else True, "Pickle could not be read"
            logging.info(f'Read {full_read_dir+read_name+file_ext} from {"S3" if bool_read_s3 else "local disk"}')
            return obj
        else:
            assert False, f'Reading extension {file_ext} not supported'

if __name__=="__main__":
    logging.getLogger().setLevel(logging.DEBUG)