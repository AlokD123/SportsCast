import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
import glnts
import pickle
import pandas as pd
#from Preprocessing import Preprocessing
from SavingReading import SavingReading
import logging

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



class DataLoading:
    def __init__(self,saver_reader=None):
        self.saver_reader = saver_reader if saver_reader is not None else SavingReading()

    def preprocessing(self,data):
        #assert 'date' in data.columns
        data['date'] = pd.to_datetime(data['date'])
        return data 

    def generate_list_ds(self,data, targets, targets_meta, targets_raw, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta, player_names:list, boolTransformed=False):
        data_meta = glnts.generate_minimal_metadata_all(data, index=None)
        list_ds = glnts.getListDS(targets if boolTransformed else targets_raw, data_meta,stat_cat_features,dyn_cat_features,dyn_real_features,player_names)
        return list_ds

    def load_data_listDS(self,
                        data,
                        full_save_dir,
                        fname_params_sffix,
                        boolSplitTrainTest,
                        index='date',
                        feature='cumStatpoints',
                        forecast_from='2018-10-03',
                        transform='none',               #TODO: keep? Currently could use in glnts.assembleTargets with stand+scale
                        roster=None,
                        column_list=None,
                        use_exog_feat=False,
                        boolTransformed=False,
                        boolSave=False,
                        stand=False,
                        scale=True):

        '''
        Inputs
        =====
        data: for ALL players

        Outputs
        =====
        x_train_listDS.p + x_test_listDS.p for that player
        '''

        assert len(data)>0,"Missing data"
        assert roster is not None and len(roster)>0, "Missing roster"
        assert all([feat in data.columns for feat in ['name','date']]), f"Missing feature. Data has columns: {data.columns}"


        ret = glnts.prep_df(data, \
                            roster, \
                            split_from=forecast_from, \
                            column_list=column_list, \
                            streamlit=False, \
                            stand=stand, \
                            scale=scale, \
                            boolSplitTrainTest=boolSplitTrainTest \
                            )
                        
        if ret is None: #If any errors in preparing, skip
            logging.warning(f'Error getting data for all players!')
            return None
        
        if boolSplitTrainTest:
            train, test, targets_trn, targets_test, targets_raw_trn, targets_raw_test, targets_meta_trn, targets_meta_test, targets_raw_meta_trn, targets_raw_meta_test, stat_cat_features_trn, \
            stat_cat_features_test, dyn_cat_features_trn, dyn_cat_features_test, dyn_real_features_trn, dyn_real_features_test, dyn_real_features_meta_trn, dyn_real_features_meta_test \
            =  ret 

            #Get names of players with sufficient data
            player_names = train['name'].unique()

            #COPY FEATURE VERIFICATION GOES HERE.... ARIMA 156:196

            #FEATURE_SELECTION
            assert all([feat in train.columns for feat in [feature]]), f'Feat missing in these train columns: {train.columns}'
            assert all([feat in test.columns for feat in [feature]]), f'Feat missing in these test columns: {test.columns}'
            #TODO: add index change (ARIMA 205:210) if index = 'date' needed

            train_list_ds = self.generate_list_ds(train,targets_trn, targets_meta_trn, targets_raw_trn, stat_cat_features_trn, dyn_cat_features_trn, dyn_real_features_trn, dyn_real_features_meta_trn, player_names, boolTransformed=boolTransformed)
            test_list_ds = self.generate_list_ds(test,targets_test, targets_meta_test, targets_raw_test, stat_cat_features_test, dyn_cat_features_test, dyn_real_features_test, dyn_real_features_meta_test, player_names, boolTransformed=boolTransformed)

            if boolSave:
                self.saver_reader.save(train_list_ds,"train_ds_all"+fname_params_sffix,full_save_dir,bool_save_s3=False)
                self.saver_reader.save(test_list_ds,"test_ds_all"+fname_params_sffix,full_save_dir,bool_save_s3=False)

            return train_list_ds, test_list_ds
        else:
            data, targets, targets_meta, targets_raw, targets_raw_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta = ret

            #Get names of players with sufficient data
            player_names = data['name'].unique()

            list_ds = self.generate_list_ds(data, targets, targets_meta, targets_raw, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta, player_names, boolTransformed=boolTransformed)
            if boolSave:
                self.saver_reader.save(list_ds,"retrain_ds_all"+fname_params_sffix,full_save_dir,bool_save_s3=False)

            return list_ds



    #TODO: add 'transform' param input to load_data_listDS()?? See above

    #TODO: move use_exog_feat to TrainEvaluate.py
    def load_data_main(self,data_dir,data_fname,roster_dir,roster_fname, \
                        full_save_dir, fname_params_sffix, boolSplitTrainTest, \
                        use_exog_feat=False, boolTransformed=False, boolSave=False, \
                        column_list = ['date', 'name', 'gameNumber', 'cumStatpoints'], stand=False, \
                        scale=True, index='date',feature='cumStatpoints',forecast_from='2018-10-03'):
        try:
            #TODO: replace with saver_reader.read()
            data = self.saver_reader.read(file_ext='.csv',read_name=data_fname,full_read_dir=data_dir,bool_read_s3=False)
            full_roster = self.saver_reader.read(file_ext='.csv',read_name=roster_fname,full_read_dir=roster_dir,bool_read_s3=False)
            data = self.preprocessing(data)
            
            ret = self.load_data_listDS(data=data,roster=full_roster, use_exog_feat=use_exog_feat, stand=stand, \
                                        boolTransformed=boolTransformed, boolSave=boolSave, \
                                        column_list=column_list,scale=scale,index=index,feature=feature, \
                                        forecast_from=forecast_from,boolSplitTrainTest=boolSplitTrainTest, \
                                        full_save_dir=full_save_dir, fname_params_sffix=fname_params_sffix)
            assert ret is not None, f"Could not load data"
            if boolSplitTrainTest:
                train_list_ds, test_list_ds = ret
                return train_list_ds, test_list_ds
            else:
                data_list_ds = ret
                return data_list_ds
        
        except AssertionError as err:
            logging.error(f'Error in loading data: {err}')
            return None


'''
#CREATE A TRAINING SET HERE
if __name__=="__main__":
    trn_params_sfx = "" #TODO
    load_data_main(DATA_DIR+DATA_FILENAME+".csv",ROSTER_DIR+ROSTER_FILENAME+".csv",TRAIN_DS_DIR, fname_params_sffix=trn_params_sfx, \
                    boolSplitTrainTest=True, use_exog_feat=True, boolTransformed=False, boolSave=True, \
                    column_list = ['date', 'name', 'gameNumber', 'cumStatpoints'], stand=False, \
                    scale=True, index='date',feature='cumStatpoints',forecast_from='2018-10-03')
'''