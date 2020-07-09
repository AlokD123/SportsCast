import fire

import os
import pickle
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from SavingReading import SavingReading

from DataLoading import DataLoading
from DataIngestion import DataIngestion

import logging
import pdb


#TODO: maybe don't import here? This module could be opaque to model details
from ARIMA import ARIMA
import pmdarima as pm       #Only needed here to support legacy saved models. Should remove otherwise to again make module opaque

class PlayerForecaster:
    ''' Main class for inference and retraining pipelines '''
    def __init__(self,models_dir=None,models_filename=None):
        '''
        Parameters
        ----
        models_dir: location of trained model to load

        models_filename: filename for same
        '''
        self.currPlayer = None
        self.currModel = None
        self.models_dir = models_dir
        self.models_filename = models_filename
        self.all_model_results_df = None            #IMPORTANT: see MultiARIMA.py for explanation
        self.saver_reader = SavingReading()
        self.ingestor = DataIngestion(initial_ingestion=False,saver_reader=self.saver_reader)
        self.data_loader = DataLoading(saver_reader=self.saver_reader)

        assert self.saver_reader is not None, "Failed to create a saver-reader"

        #TODO: add cases for path=None or path=''
        if models_dir is not None and models_filename is not None:
            assert self.load_all_models(models_dir=models_dir,models_filename=models_filename,load_pickled=True) is True, "Failed to create PlayerForecaster"


    def load_all_models(self,models_dir=os.getcwd()+'/data/models',models_filename="model_results",load_pickled=False): #TODO: override with hparams suffix from CLI arg
        '''
        Loads trained models for all players

        Parameters
        ----
        models_dir: location of trained model to load

        models_filename: filename for same

        load_pickled: load model files that have been pickled

        Returns
        ----
        Success boolean (True if success, False/None otherwise)
        '''
        try:
            if load_pickled:
                obj = self.saver_reader.read(file_ext='.p',read_name=models_filename,full_read_dir=models_dir,bool_read_s3=False)
                self.all_model_results_df = obj
            else:
                df = self.saver_reader.read(file_ext='.csv',read_name=models_filename,full_read_dir=models_dir,bool_read_s3=False)
                self.all_model_results_df = df
            logging.info('Loaded models!!')
            self.models_dir = models_dir; self.models_filename = models_filename
            return True
        except AssertionError as err:
            logging.error(f'Could not load models!!: {err}')
            return False

    def getPlayerModel(self,player_name):
        '''
        Gets the trained model for a given player

        Returns
        ----
        Model instance for player
        '''
        try:
            self.currPlayer = player_name
            return self.all_model_results_df.loc[player_name,'model']
        except KeyError as err:
            logging.error(f'No player {player_name} found: {err}')
            self.currPlayer = None
            return None
        
    #TODO: replace string return with JSON return. HERE AND IN INFER_SERVE
    def pred_points(self,player_name:str, num_games: int, models_dir:str=os.getcwd()+"/data/models", \
                    models_filename:str="model_results",print_single_str=False):                            #TODO: override with hparams suffix from CLI arg
        '''
        Runs inference

        Parameters
        -----
        player_name: name

        num_games: number of games to forecast

        models_dir: location of model to load

        models_filename: filename for same

        print_single_str: whether to return result as a string or list

        Returns
        ----
        String or list of strings with predicted results

        '''
        
        #TODO: add test for self.currModel is None
        if self.all_model_results_df is None:
            if not self.load_all_models(models_dir=models_dir,models_filename=models_filename,load_pickled=True):
                return "No model available for predictions"
        if not (self.currPlayer == player_name and (self.currModel is not None)):
            self.currModel = self.getPlayerModel(player_name)
        
        if self.currModel == None:
            return "No such player found"

        predictions, intervals = self.currModel.predict(n_periods=num_games, return_conf_int=True)
        if not isinstance(self.currModel,pm.ARIMA):
            __, predictions, low_intervals, high_intervals = self.currModel.postprocess(train_predictions=None, predictions=predictions, intervals=intervals)
        assert len(predictions) == num_games, "Issue with model.predict inp/out dims"

        if print_single_str:
            string_resp = f'For the next {num_games} games, the predicted running sum of points for {player_name} are: '
            for i in range(num_games):
                string_resp = string_resp + f'\n    Game {i}: {predictions[i]:.2f} (lower bound: {low_intervals[i]:.2f}, upper bound:{high_intervals[i]:.2f})'
            return string_resp
        else:
            return [f'Game {i}: {predictions[i]:.2f} (lower bound: {low_intervals[i]:.2f}, upper bound:{high_intervals[i]:.2f})' for i in range(num_games)]


    
    def retrain(self,hparams:str,retrain_ds_all_players:ListDataset,models_dir:str=os.getcwd()+"/data/models",models_fname:str="model_results",use_exog_feats:bool=True):
        '''
        Retrains models for all players in the retrain dataset. Overwrites current model with retrained one

        Parameters
        -----
        hparams: str of hyperparameters to use

        retrain_ds_all_players: ListDataset for retraining (see MultiARIMA.py for definition of ListDataset)

        models_dir: location of model to load

        models_filename: filename for same

        use_exog_feats: use exogenous features when retraining or not

        Returns
        ----
        Success boolean

        '''
        
        if(self.load_all_models(models_dir,models_fname+hparams,load_pickled=True)): #.csv file containing dataframe of ModelResult for all players             #TODO: change load_pickled to false for NEW .csv-saved files
            try:
                #For each player's data in retrain dataset...
                for retrain_dict in retrain_ds_all_players.list_data:
                    assert 'name' in retrain_dict.keys(), "No field 'name' provided in retrain_ds_all_players"
                    player_name = retrain_dict['name']
                    player_mdl = self.getPlayerModel(player_name)
                    if player_mdl is None:
                        logging.warning(f'No model found for {player_name}')
                        continue
                    
                    
                    try:
                        if isinstance(player_mdl,ARIMA):
                            player_mdl.update(player_dict=retrain_dict) #Re-train
                        elif isinstance(player_mdl,pm.ARIMA):
                            ret = ARIMA.update_PMDARIMA(model=player_mdl,player_dict=retrain_dict,use_exog_feats=use_exog_feats,player_name=player_name) #Retrain legacy model
                            assert ret is not None, "PMDARIMA model failed to update"
                            player_mdl, new_targets, exog_feats = ret
                            player_mdl = ARIMA(player_train_labels=new_targets,features_trn=exog_feats,model=player_mdl,player_name=player_name,transform='none') #Create ARIMA.py class model after retrained (wrap)
                            assert isinstance(player_mdl,ARIMA), "Failed to create an ARIMA-class model"
                        else:
                            logging.warning(f'Not an ARIMA model! Unimplemented, so skip')
                            #TODO: complete update for DeepAR
                            continue
                    except Exception as err:
                        logging.error(f'Could not retrain for {player_name}:{err}. Skip')
                        continue

                    self.all_model_results_df.loc[player_name,'model'] = player_mdl

                    #Overwrite models/results dataframe
                    logging.info('Overwriting df after retraining')
                    self.saver_reader.save(self.all_model_results_df,models_fname+hparams,models_dir,bool_save_pickle=True,bool_save_s3=False)
                
                return True

            except AssertionError as err:
                logging.error(f'Cant retrain at all: {err}')
                return None

        else:
            return False

    '''        
    def retrain_main(self,hparams:str,updated_data_dir:str=UPDATED_DATA_DIR,updated_data_fname:str=UPDATED_DATA_FILENAME,\
                    roster_dir:str=ROSTER_DIR, roster_fname:str=ROSTER_FILENAME, load_save_dir:str=RETRAIN_DS_DIR): #TODO: see TrainingEvaluation.train()
    '''
    #NOTE: up to user to appropriately set whether re-training with or without exogenous features, depending on pre-trained model. In most cases, use_exog_feats=True
    def retrain_main(self,hparams:str, use_exog_feats:bool, roster_dir:str=os.getcwd()+"/data/inputs", roster_fname:str="full_roster_4_seasons", \
                    old_ingest_dir=os.getcwd()+"/data/inputs",old_ingest_name="full_dataset_updated", new_ingest_dir:str=os.getcwd()+"/data/inputs",new_ingest_name:str=None, \
                    updated_data_dir:str=os.getcwd()+"/data/inputs",updated_data_fname:str="full_dataset_updated",\
                    load_save_dir:str=os.getcwd()+"/data/retrain_ds", \
                    models_dir:str=os.getcwd()+"/data/models",models_fname:str="model_results"):

        '''
        Calls retrain with newly ingested data file and appropriate parameters

        Parameters
        ---
        Described variously here, in DataIngestion.py, in DataLoading.py

        Returns
        ----
        Success boolean
        '''
        all_rosters, new_full_df = self.ingestor.ingest_new_league_data(roster_name=roster_fname,roster_dir=roster_dir,save_dir=updated_data_dir,save_name=updated_data_fname, \
                                                                        old_read_dir=old_ingest_dir,old_read_name=old_ingest_name,new_read_dir=new_ingest_dir,new_read_name=new_ingest_name)

        if len(new_full_df)==0:
            logging.warning(f'No new data to retrain')
            return False

        retrn_params_sfx = "" #TODO
        new_list_ds = self.data_loader.load_data_main(data_dir=updated_data_dir, data_fname=updated_data_fname, \
                                                        roster_dir=roster_dir,roster_fname=roster_fname, \
                                                        full_save_dir=load_save_dir, \
                                                        boolSplitTrainTest = False, use_exog_feat=use_exog_feats, boolTransformed=True, \
                                                        boolSave=True, stand=False, scale=False, \
                                                        fname_params_sffix=retrn_params_sfx)
        
        if new_list_ds is None or len(new_list_ds.list_data)==0:
            logging.error(f'Couldnt retrain. Data loading not working')
            return False
        else:
            logging.debug('Starting retraining with the following:\n')
            logging.debug(new_list_ds.list_data)
            if self.retrain(hparams=hparams,retrain_ds_all_players=new_list_ds,models_dir=models_dir,models_fname=models_fname,use_exog_feats=use_exog_feats) is not True: #TODO: Add options for non-default re-train save location
                return False
            return True


if __name__ == '__main__':
    fire.Fire(PlayerForecaster) #Start CLI
