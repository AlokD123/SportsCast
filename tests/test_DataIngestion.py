from SportsCast_Framework.scripts.DataIngestion import DataIngestion

import unittest
from parameterized import parameterized, parameterized_class

import pdb
import logging
import os

class TestDataIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.debug('Testing data ingestion')

    @classmethod
    def tearDownClass(cls):
        logging.debug('Finished testing')

    def setUp(self):
        self.ingestor = DataIngestion(initial_ingestion=False)
        self.currSeason_metadata = {'dir':os.getcwd()+"/data/inputs", "fname":"currSeason_metadata"}
        self.roster_data = {'dir':os.getcwd()+"/data/inputs","fname":"full_roster_4_seasons"}
        self.gameIngestion_data = {'dir':os.getcwd()+"/data/inputs",'fname':"full_dataset_updated_copy"}
        self.newGameIngestion_old_data = {'dir':os.getcwd()+"/data/inputs","fname":"full_dataset_updated"}
        self.newGameIngestion_simulated_latest_data = {'dir':os.getcwd()+"/data/inputs","fname":"full_dataset_sim-latest"}
        self.newGameIngestion_new_data = {'dir':os.getcwd()+"/data/inputs","fname":"full_dataset_new"}

    def tearDown(self):
        pass

    def testIngestSeasonMetadata(self):
        currSeason_metadata_dir = self.currSeason_metadata['dir']
        currSeason_metadata_fname = self.currSeason_metadata['fname']
        self.ingestor.load_season_data(save_path=currSeason_metadata_dir,save_name=currSeason_metadata_fname)
        new_data_read_val = self.ingestor.saver_reader.read('.csv',currSeason_metadata_fname,currSeason_metadata_dir,bool_read_s3=False)
        self.assertTrue((new_data_read_val is not None) and (new_data_read_val is not False), "Could not ingest current season's metadata")
        self.ingestor.saver_reader.delete_local(currSeason_metadata_fname+'.csv',currSeason_metadata_dir)

    @parameterized.expand([
       ("season_id", [30203021]), #No data available for this future season
    ])
    def testIngestion_DataUnavailable(self,name,season_id):
        future_season_id = season_id
        save_dir = self.gameIngestion_data['dir']
        save_fname = self.gameIngestion_data['fname']

        all_rosters,full_df = self.ingestor.ingest_league_data(save_dir=save_dir,save_name=save_fname, \
                                                        roster_name=self.roster_data['fname'], roster_dir=self.roster_data['dir'], \
                                                        boolSaveData=True,season_id_list=future_season_id)
        self.assertTrue(all_rosters.empty and full_df.empty, "Saved non-zero data even though season data unavailable")

    def ingest_new_data(self,old_data_dir,old_data_fname,latest_data_dir,latest_data_fname,new_data_dir,new_data_fname,roster_dir,roster_fname):
        return self.ingestor.ingest_new_league_data(old_read_dir=old_data_dir,old_read_name=old_data_fname, \
                                                    save_dir=new_data_dir,save_name=new_data_fname, \
                                                    roster_name=roster_fname, roster_dir=roster_dir, \
                                                    boolSaveData=True,new_read_dir=latest_data_dir,new_read_name=latest_data_fname)

    def testNewIngestion(self):
        old_data_dir = self.newGameIngestion_old_data['dir']
        old_data_fname = self.newGameIngestion_old_data['fname']
        latest_data_dir = self.newGameIngestion_simulated_latest_data['dir']
        latest_data_fname = self.newGameIngestion_simulated_latest_data['fname']
        new_data_dir = self.newGameIngestion_new_data['dir']
        new_data_fname = self.newGameIngestion_new_data['fname']

        all_rosters, new_full_df = self.ingest_new_data(old_data_dir,old_data_fname,latest_data_dir,latest_data_fname,new_data_dir,new_data_fname,self.roster_data['dir'],self.roster_data['fname'])
        
        self.assertTrue(len(new_full_df)>0, f"Could not get subset of latest data that is new! Old ingestion tracking not working? Newest read = {new_full_df.head()}")
        new_data_read_val = self.ingestor.saver_reader.read('.csv',new_data_fname,new_data_dir,bool_read_s3=False)
        self.assertTrue((new_data_read_val is not None) and (new_data_read_val is not False), "Could not save new data")
        self.ingestor.saver_reader.delete_local(new_data_fname+'.csv',new_data_dir)

    def testNewIngestion_DataUnavailable(self):
        old_data_dir = self.newGameIngestion_old_data['dir']
        old_data_fname = self.newGameIngestion_old_data['fname']
        latest_data_dir = old_data_dir
        latest_data_fname = old_data_fname
        new_data_dir = self.newGameIngestion_new_data['dir']
        new_data_fname = self.newGameIngestion_new_data['fname']

        all_rosters, new_full_df = self.ingest_new_data(old_data_dir,old_data_fname,latest_data_dir,latest_data_fname,new_data_dir,new_data_fname,self.roster_data['dir'],self.roster_data['fname'])
        
        self.assertTrue(len(new_full_df)==0, f"Incorrectly identified new data when none available. Old ingestion tracking not working? Newest read = {new_full_df.head()}")

        try:
            new_data_read_val = self.ingestor.saver_reader.read('.csv',new_data_fname,new_data_dir,bool_read_s3=False)
            self.ingestor.saver_reader.delete_local(new_data_fname+'.csv',new_data_dir)
        except AssertionError:
            new_data_read_val = None    #NOTE: SHOULD reach here (no data to be read)

        self.assertTrue(new_data_read_val is None, "Incorrectly saved data")


if __name__ == '__main__':
    unittest.main()