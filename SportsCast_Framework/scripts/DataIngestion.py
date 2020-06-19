import api
import pandas as pd
import time
import schedule
from SavingReading import SavingReading
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
MODELRESULT_DIR = PRJ_PATH + "/data/outputs"
MODELRESULT_FILENAME = "arima_results"          #.p     #Join model type (arima or deepar) string as well as hparam string to this
'''
#TODO: change "arima_results" to "model_results"


class DataIngestion:

    def __init__(self):
        self.saver_reader = SavingReading()
        
    def load_some_data(self,save_path=None,save_name=None):
        season_id, season_start, season_end = api.get_current_season()
        df = pd.DataFrame({"season_id":season_id,"season_start":season_start,"season_end":season_end})
        if (save_path is None) or (save_name is None):
            save_path = UPDATED_DATA_DIR
            save_name = 'some_data'                 #.csv
        self.saver_reader.save(df,save_name,save_path)
        

    def ingest_league_data(self,save_dir=None,save_name=None):
        all_rosters = api.get_all_rosters(streamlit=False)
        full_df = api.assemble_multiplayer_stat_dataframe(player_id_list=list(all_rosters.index), stat_list=[], boolAddSimulated=False)
        if (save_dir is None) or (save_name is None):
            save_dir = UPDATED_DATA_DIR
            save_name = UPDATED_DATA_FILENAME       #.csv
        self.saver_reader.save(full_df,save_name,save_dir)
            
    def ingest_new_league_data(self):
        #TODO: same as ingest_league_data(), using Season Game data per player, but drop data for games before last saved date / current date
        pass
