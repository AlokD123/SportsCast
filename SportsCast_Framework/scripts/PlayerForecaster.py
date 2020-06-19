import fire

import os
import pickle
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from SavingReading import SavingReading


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


#TODO: maybe don't import here? This module could be opaque to model details
import ARIMA

class PlayerForecaster:
    #TODO: replace models_path with models_dir, models_filename
    def __init__(self,models_path=None):
        self.currPlayer = None
        self.currModel = None
        self.models_path = models_path
        self.all_model_results_df = None
        self.saver_reader = SavingReading()

        #TODO: fix logic for path=None or path=absent
        if models_path is not None:
            assert self.load_all_models(models_path) is True, "Failed to create PlayerForecaster"

    #TODO: update default to use ENV VARIABLES
    #TODO: REPLACE open() with saver_reader.read()
    def load_all_models(self,models_path=os.getcwd()+'/../data/outputs/arima_results_m3_fourStep_noFeatures.p'):
    #def load_all_models(self,models_dir=MODELRESULT_DIR,models_filename=MODELRESULT_FILENAME+hparams):
        try:
            f = open(self.models_path,"rb")
            self.all_model_results_df = pickle.load(f)
            print('Loaded models!!')
            return True
        except NameError:
            print('Could not load models!!')
            return False

    def getPlayerModel(self,player_name):
        try:
            self.currPlayer = player_name
            return self.all_model_results_df.loc[player_name,'model']
        except KeyError:
            self.currPlayer = None
            return None
        
    def pred_points(self,player_name:str, num_games: int, print_single_str=False):
        #TODO: add test for self.currModel is None
        if not (self.currPlayer == player_name and (self.currModel is not None)):
            self.currModel = self.getPlayerModel(player_name)
        
        if self.currModel == None:
            return "No such player found"

        prediction, interval = self.currModel.predict(n_periods=num_games, return_conf_int=True)
        #TODO: postprocess here?
        assert len(prediction) == num_games, "Issue with model.predict inp/out dims"

        if print_single_str:
            string_resp = f'For the next {num_games} games, the predicted running sum of points for {player_name} are: '
            for i in range(num_games):
                string_resp = string_resp + f'\n    Game {i}: {prediction[i]:.2f} (lower bound: {interval[i,0]:.2f}, upper bound:{interval[i,1]:.2f})'
            return string_resp
        else:
            return [f'Game {i}: {prediction[i]:.2f} (lower bound: {interval[i,0]:.2f}, upper bound:{interval[i,1]:.2f})' for i in range(num_games)]


    def retrain(self,hparams:str,retrain_ds_all_players:ListDataset):
        if(self.load_all_models(MODELRESULT_DIR+MODELRESULT_FILENAME+hparams+".csv")): #.csv file containing dataframe of ModelResult for all players
            try:
                for retrain_dict in retrain_ds_all_players.list_data:
                    assert 'name' in retrain_dict.keys(), "No field 'name' provided in retrain_ds_all_players"
                    player_name = retrain_dict['name']
                    player_mdl = self.getPlayerModel(player_name)
                    if player_mdl is None:
                        print(f'No model found for {player_name}')
                        continue
                    '''
                    TODO: update the other columns of self.all_model_results_df.loc[player_name,:] by first performing evaluation on the new retrain data?
                    e.g. self.all_model_results_df.loc[player_name,??] = player_mdl.evaluate(retrain_dict['target'],player_mdl.predict(n_periods,exog=) )
                    '''
                    try:
                        if isinstance(player_mdl,ARIMA):
                            player_mdl.update(player_dict=retrain_dict)
                        else:
                            #TODO: complete update for DeepAR
                            pass
                    except Exception as err:
                        print(f'Could not retrain for {player_name}:{err}')
                    self.all_model_results_df.loc[player_name,'model'] = player_mdl

                    #Overwrite df
                    print('Overwriting df after retraining')
                    self.saver_reader.save(self.all_model_results_df,MODELRESULT_FILENAME+hparams,MODELRESULT_DIR)

            except AssertionError as err:
                print(f'Cant retrain at all: {err}')
            
    def retrain_main(self):
        #TODO: copy the below here.
        hparams="" #TODO: see TrainingEvaluation.train()
        self.retrain(hparams=hparams,retrain_ds_all_players=new_list_ds)
        pass

if __name__ == '__main__':
    fire.Fire(PlayerForecaster)


'''
For retrain() input... or TODO: MOVE THIS TO retrain()????

import api
import DataLoading
from DataIngestion import *

ingestor = DataIngestion()
ingestor.ingest_league_data(self,save_dir=UPDATED_DATA_DIR,save_name=UPDATED_DATA_FILENAME) #TODO: change to ingest_new_data()


#TODO: replace data_path with data_dir, data_filename
#TODO: replace roster_path with roster_dir, roster_filename

retrn_params_sfx = "" #TODO
new_list_ds = DataLoading.load_data_main(data_path=UPDATED_DATA_DIR+UPDATED_DATA_FILENAME+'.csv', \
                                        roster_path=ROSTER_DIR+ROSTER_FILENAME+'.csv', \
                                        full_save_dir=TRAIN_TEST_DS_DIR, \
                                        boolSplitTrainTest = False, use_exog_feat=True, boolTransformed=True, \
                                        boolSave=True, stand=False, scale=True, \
                                        fname_params_sffix=retrn_params_sfx)




'''