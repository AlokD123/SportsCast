from SavingReading import SavingReading
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from Model import Model
from MultiARIMA import MultiARIMA

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
MODELRESULT_DIR = PRJ_PATH + "/data/models"
MODELRESULT_FILENAME = "arima_results"          #.p     #Join model type (arima or deepar) string as well as hparam string to this
'''
#TODO: change "arima_results" to "model_results"

saver_reader = SavingReading()


def train(model,transform,full_save_dir:str,trn_ds_relative_path:str,results_save_dir:str, results_save_filename:str):
    
    assert isinstance(model,Model), "Improper model provided for training"

    #TODO: replace with saver_reader.read()
    try:
        f = open(full_save_dir+trn_ds_relative_path,"rb")
        train_list_ds_all_players = pickle.load(f)
    except NameError:
        print(f'Couldn\'t open {full_save_dir+trn_ds_relative_path}')
        #raise Exception (f"Couldnt open ")
        return None

    #Set hparams str
    hparams = f"transform={transform}+___" #TODO

    #Create model
    mar = MultiARIMA(train_list_ds_all_players) #TODO: ADD HPARAM ARGUMENTS
    mar.create(use_exog_feat=True)
    #Fit to dataset
    mar.fit()
    #Save results
    saver_reader.save(mar.models_results_df,MODELRESULT_FILENAME+hparams,MODELRESULT_DIR)
    #TODO: possibly evaluate training error


def evaluate():

    pass

#test_ds_relative_path:str