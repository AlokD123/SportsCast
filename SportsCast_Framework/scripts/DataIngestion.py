import api
import pandas as pd
import time
import schedule
from SavingReading import SavingReading
import os

#global PRJ_PATH =

class DataIngestion:

    def __init__(self):
        self.saver_reader = SavingReading()
        
    def load_some_data(self,save_path=None,save_name=None):
        season_id, season_start, season_end = api.get_current_season()
        df = pd.DataFrame({"season_id":season_id,"season_start":season_start,"season_end":season_end})
        if (save_path is None) or (save_name is None):
            save_path = PRJ_PATH + "/" + 'data/inputs'
            save_name = 'some_data.csv'
        self.saver_reader.save(df,save_name,save_path)
        

    def ingest_league_data(self,save_path=None,save_name=None):
        all_rosters = api.get_all_rosters(streamlit=False)
        full_df = api.assemble_multiplayer_stat_dataframe(player_id_list=list(all_rosters.index), stat_list=[], boolAddSimulated=True)
        if (save_path is None) or (save_name is None):
            save_path = PRJ_PATH + "/" + 'data/inputs'
            save_name = 'some_data.csv'
        self.saver_reader.save(full_df,save_name,save_path)
            
    def ingest_new_league_data(self):
        pass
