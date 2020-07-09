import api as api
import pandas as pd
import time
from SavingReading import SavingReading
import os
import pdb
import logging


class DataIngestion:
    ''' Class to ingest new data using API module '''
    def __init__(self,initial_ingestion:bool=True,saver_reader=None):
        '''
        Parameters
        ----
        initial_ingestion: whether to ingest immediately after creating ingestor

        saver_reader: an instance to save and read from all files generically (see SavingReading.py)
        '''
        self.saver_reader = saver_reader if saver_reader is not None else SavingReading()
        self.last_ingestion_time = None
        if initial_ingestion:
            self.ingest_league_data(boolSaveData=True)
        
    def load_season_data(self,save_path:str=os.getcwd()+"/data/inputs",save_name:str='some_data'): #.csv
        '''
        Saves season data to a CSV

        Parameters
        ----
        save_path: location to save NHL season data

        save_name: filename for same

        Returns
        ----
        None
        '''
        season_id, season_start, season_end = api.get_current_season()
        df = pd.DataFrame({"season_id":season_id,"season_start":season_start,"season_end":season_end})         
        self.saver_reader.save(df,save_name,save_path)
        
    #TODO: implement updating roster each season
    def ingest_league_data(self,save_dir:str=os.getcwd()+"/data/inputs",save_name:str="full_dataset_updated", roster_name:str="full_roster_4_seasons", roster_dir:str=os.getcwd()+"/data/inputs", boolSaveData:bool=True): #.csv
        '''
        Saves league game data and roster data to a CSV for training and testing models

        Parameters
        ----
        save_path: location to save league game data

        save_name: filename for same

        roster_dir: location to save league roster data

        roster_name: filename for same

        boolSaveData: whether to save

        Returns
        ----
        all_rosters: list of all rosters for all teams during the seasons in season_id_list

        full_df: game by game data during the seasons in season_id_list
        '''
        season_id_list = [20152016, 20162017, 20172018, 20182019]           #TODO: make user-defined
        all_rosters = api.get_all_rosters(season_id_list=season_id_list)

        full_df = api.assemble_multiplayer_stat_dataframe(player_id_list=list(all_rosters.index), boolAddSimulated=False, season_id_list=season_id_list)
        test_season = season_id_list[-1] + 10001 #Try getting next season
        try:
            while True:
                df = api.assemble_multiplayer_stat_dataframe(player_id_list=list(all_rosters.index), boolAddSimulated=False, season_id_list=[test_season])
                full_df = pd.concat([full_df,df],ignore_index=True)
                test_season = test_season + 10001 #Try getting next season
        except Exception as err:
            logging.error(f'Data for {test_season} not available: {err}')
        if boolSaveData:
            self.saver_reader.save(full_df,save_name,save_dir,bool_save_s3=False)
            self.saver_reader.save(all_rosters,roster_name,roster_dir,bool_save_s3=False)

        #Store last ingestion time
        dates = full_df.sort_values(by=['date'],axis='rows')['date']
        self.last_ingestion_time = list(dates.tail(1))[0]
        return all_rosters,full_df
            
    def ingest_new_league_data(self,old_read_dir:str=os.getcwd()+"/data/inputs",old_read_name:str="full_dataset_updated", \
                                save_dir:str=os.getcwd()+"/data/inputs",save_name:str="full_dataset_updated", \
                                roster_name:str="full_roster_4_seasons", roster_dir:str=os.getcwd()+"/data/inputs", boolSaveData:bool=True, \
                                new_read_dir:str=os.getcwd()+"/data/inputs",new_read_name:str=None):
        
        '''
        Saves NEW league data to a CSV, for retraining models periodically. 

        Finds difference between past data used for (re-)training and all data now available... i.e. the new data. Then saves the new data for re-training

        Parameters
        ----
        old_read_dir: directory for data PREVIOUSLY used in training/testing (CSV)

        old_read_name: filename for same

        save_dir: directory for currently ingested data (a superset of the old data)

        save_name: filename for same

        new_read_dir: location to save just NEW data (difference)

        new_read_name: filename for same

        roster_name: see above

        roster_dir: see above

        boolSaveData: see above

        Returns
        ----
        all_rosters: see above

        new_full_df: new game by game data
        '''

        try:
            try:
                full_df_old = self.saver_reader.read('.csv',old_read_name,old_read_dir,bool_read_s3=False)
                all_rosters = self.saver_reader.read('.csv',roster_name,roster_dir,bool_read_s3=False)
            except AssertionError as err:
                logging.error(f'Error reading old data: {err}. Downloading latest')
                all_rosters,full_df = self.ingest_league_data(save_dir=save_dir,save_name=save_name,roster_name=roster_name, roster_dir=roster_dir)
                return all_rosters,full_df
            #Get latest
            if self.last_ingestion_time is None:
                old_dates = full_df_old.sort_values(by=['date'],axis='rows')['date']
                self.last_ingestion_time = list(old_dates.tail(1))[0]
            old_last_ingestion_time = self.last_ingestion_time

            #Two ways to get updated data
            if new_read_dir is not None and new_read_name is not None: #1) Try to read new data saved in a CSV
                try:
                    full_df = self.saver_reader.read('.csv',new_read_name,new_read_dir,bool_read_s3=False)
                except Exception as err:
                    logging.error(f'Error in reading new data saved to specified path: {err}. Will try to ingest new data from online')
                    __,full_df = self.ingest_league_data(save_dir=save_dir,save_name=save_name, roster_name=roster_name, roster_dir=roster_dir)
            else:
                __,full_df = self.ingest_league_data(save_dir=save_dir,save_name=save_name, roster_name=roster_name, roster_dir=roster_dir) #2) Try to ingest by downloading from online
            #Discard old dates
            new_full_df = full_df[full_df['date']>old_last_ingestion_time]
            
            #Overwrite old data with new in SAME file (to avoid iterative data storage)
            if boolSaveData:
                self.saver_reader.save(new_full_df,save_name,save_dir,bool_save_s3=False)
                logging.info("Overwrote with new data")
            return all_rosters, new_full_df

        except Exception as err:
            logging.error(f'Could not ingest new data: {err}. Using old data')
            if boolSaveData:
                self.saver_reader.save(full_df_old,save_name,save_dir,bool_save_s3=False)
            return all_rosters,full_df_old