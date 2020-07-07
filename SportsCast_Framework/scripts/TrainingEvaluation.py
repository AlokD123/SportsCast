from SavingReading import SavingReading
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from Model import Model
from MultiARIMA import MultiARIMA
import logging
import os

#ENVIRONMENT VARIABLES
'''
PRJ_PATH = 
DATA_DIR = PRJ_PATH + "/data/inputs"
DATA_FILENAME = "full_dataset_4_seasons"        #.csv
UPDATED_DATA_DIR = PRJ_PATH + "/data/inputs"
UPDATED_DATA_FILENAME = "full_dataset_updated"  #.csv
ROSTER_DIR = PRJ_PATH + "/data/inputs"
ROSTER_FILENAME = "full_roster_4_seasons"       #.csv
RETRAIN_DS_DIR = PRJ_PATH + "/data/retrain_ds"
RETRAIN_DS_FILENAME = "retrain_ds_all"          #.p
TRAIN_DS_DIR = PRJ_PATH + "/data/train_ds"
TRAIN_DS_FILENAME = "train_ds_all"              #.p
TEST_DS_DIR = PRJ_PATH + "/data/test_ds"
TEST_DS_FILENAME = "test_ds_all"               #.p
MODELTRAIN_DIR = PRJ_PATH + "/data/models/trained"
MODELTRAIN_FILENAME = "model_trained"           #.p
MODELRESULT_DIR = PRJ_PATH + "/data/models"
MODELRESULT_FILENAME = "arima_results"          #.p     #Join model type (arima or deepar) string as well as hparam string to this
'''
#TODO: change "arima_results" to "model_results"


class TrainingEvaluation():

    def __init__(self,Model_Cls:Model,use_exog_feat:bool,hparams:str="",train_ds_all:ListDataset=None):
        self.saver_reader = SavingReading()
        self.Model_Cls = Model_Cls
        self.use_exog_feat = use_exog_feat
        self.model = None if train_ds_all is None else Model_Cls(train_ds_all,transform='none')
        #Set hparams str    
        #hparams = f"transform=__,..." #TODO
        self.hparams = hparams

    def train(self,train_ds_dir:str=os.getcwd()+"/data/train_ds",train_ds_fname:str="train_ds_all", \
                model_train_dir:str=os.getcwd+"/data/models/trained", model_train_save_filename:str="model_trained", \
                trn_params_sffix:str=""):

        if self.model is None:
            try:
                train_ds_all = self.saver_reader.read(file_ext='.p',read_name=train_ds_fname+trn_params_sffix,full_read_dir=train_ds_dir,bool_read_s3=False)
            except AssertionError as err:
                logging.error(f'Couldn\'t open path: {err}. Can\'t create model to train')
                return None
            
            #Create model
            self.model = self.Model_Cls(train_ds_all,transform='none') #TODO: ADD MORE HPARAM ARGUMENTS

        
        self.model.create(use_exog_feat=self.use_exog_feat)
        #Fit to dataset
        self.model.fit()
        #Save model
        if not self.saver_reader.save(self.model,model_train_save_filename+self.hparams,model_train_dir):
            logging.warn(f'Failed to save trained model!')
        #TODO: possibly evaluate training error
        return True


    def evaluate(self,model_train_dir:str=os.getcwd+"/data/models/trained", model_train_save_filename:str="model_trained",\
                test_ds_dir:str=os.getcwd()+"/data/test_ds",test_ds_fname:str="test_ds_all",\
                modelresult_dir:str=os.getcwd+"/data/models", modelresult_filename:str="model_results",
                test_params_sffix:str="", retrain_horizon:int=0):

        if model_train_save_filename == "model_trained":
            model_train_save_filename = model_train_save_filename + self.hparams
        if modelresult_filename == "model_results":
            modelresult_filename = modelresult_filename + self.hparams

        assert self.model is not None, "Create model first! (and ideally also train)"

        try:
            test_ds_all = self.saver_reader.read(file_ext='.p',read_name=test_ds_fname+test_params_sffix,full_read_dir=test_ds_dir,bool_read_s3=False)
        except AssertionError as err:
            logging.error(f'Couldn\'t open path: {err}. Can\'t create model to train')
            return None

        self.model.evaluate(test_ds_all,horizon=retrain_horizon)

        #Save results
        if not self.saver_reader.save(self.model.model_results_df,modelresult_filename+self.hparams,modelresult_dir):
            logging.warn(f'Failed to save trained model!')