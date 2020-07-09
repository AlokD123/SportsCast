import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
import glnts
import pickle
import pandas as pd
#from Preprocessing import Preprocessing
from SavingReading import SavingReading
import logging


class DataLoading:
    ''' Class for loading and preprocessing data '''
    def __init__(self,saver_reader=None):
        self.saver_reader = saver_reader if saver_reader is not None else SavingReading() #To save/load (see SavingReading.py)

    @classmethod
    def preprocessing(cls,data):
        data['date'] = pd.to_datetime(data['date']) #Change to datetime
        return data 

    @classmethod
    def generate_list_ds(cls,data, targets, targets_meta, targets_raw, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta, player_names:list, boolTransformed=False):
        ''' Creates a list dataset using static and dynamic (real and categorical) features, as well as train/test labels ('targets') and metadata '''
        data_meta = glnts.generate_minimal_metadata_all(data, index=None)
        list_ds = glnts.getListDS(targets if boolTransformed else targets_raw , data_meta,stat_cat_features,dyn_cat_features,dyn_real_features,player_names)
        return list_ds

    def load_data_listDS(self,
                        data,                           #game-by-game data for ALL players (dataframe)
                        full_save_dir,                  #location to save loaded data
                        fname_params_sffix,             #suffix for parameters to use when saving
                        boolSplitTrainTest,             #whether to split
                        index='date',                   #index variable
                        feature='cumStatpoints',        #name of feature to use as label in dataset
                        forecast_from='2018-10-03',     #train-test split date
                        roster=None,                    #roster data for ALL players (dataframe)
                        column_list=None,               #list of main data/metadata columns to load from dataframe 'data'
                        use_exog_feat=False,            #whether to load the additional exogenous features, e.g. teammate presence during a game (see ARIMA.py)
                        boolTransformed=False,          #transform data or not
                        boolSave=False,                 #save or not
                        stand=False,                    #standardize data or not
                        scale=False):                   #scale (normalize) data or not

        '''
        Loads and preprocesses data. Optionally saves

        Returns
        ---
        train_list_ds: ListDataset for training a SINGLE player model (see MultiARIMA.py for details)

        test_list_ds: ListDataset for testing a SINGLE player model

        list_ds: aggegate of the above, if no train-test split (for retraining)
        '''

        assert len(data)>0,"Missing data"
        assert roster is not None and len(roster)>0, "Missing roster"
        assert all([feat in data.columns for feat in ['name','date']]), f"Missing feature. Data has columns: {data.columns}"

        #Performs preprocessing, splitting, and exogenous feature extraction using raw dataframes
        ret = glnts.prep_df(data, \
                            roster, \
                            split_from=forecast_from, \
                            column_list=column_list, \
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

            #FEATURE_SELECTION
            assert all([feat in train.columns for feat in [feature]]), f'Feat missing in these train columns: {train.columns}'
            assert all([feat in test.columns for feat in [feature]]), f'Feat missing in these test columns: {test.columns}'

            #Create ListDataset instances
            train_list_ds = DataLoading.generate_list_ds(train,targets_trn, targets_meta_trn, targets_raw_trn, stat_cat_features_trn, dyn_cat_features_trn, dyn_real_features_trn, dyn_real_features_meta_trn, player_names, boolTransformed=boolTransformed)
            test_list_ds = DataLoading.generate_list_ds(test,targets_test, targets_meta_test, targets_raw_test, stat_cat_features_test, dyn_cat_features_test, dyn_real_features_test, dyn_real_features_meta_test, player_names, boolTransformed=boolTransformed)

            if boolSave:
                self.saver_reader.save(train_list_ds,"train_ds_all"+fname_params_sffix,full_save_dir,bool_save_s3=False)
                self.saver_reader.save(test_list_ds,"test_ds_all"+fname_params_sffix,full_save_dir,bool_save_s3=False)

            return train_list_ds, test_list_ds
        else:
            data, targets, targets_meta, targets_raw, targets_raw_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta = ret

            #Get names of players with sufficient data
            player_names = data['name'].unique()

            #Create ListDataset instances
            list_ds = DataLoading.generate_list_ds(data, targets, targets_meta, targets_raw, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta, player_names, boolTransformed=boolTransformed)
            if boolSave:
                self.saver_reader.save(list_ds,"retrain_ds_all"+fname_params_sffix,full_save_dir,bool_save_s3=False)

            return list_ds



    #TODO: move use_exog_feat to TrainEvaluate.py
    def load_data_main(self,data_dir,data_fname,roster_dir,roster_fname, \
                        full_save_dir, fname_params_sffix, boolSplitTrainTest, \
                        use_exog_feat=False, boolTransformed=False, boolSave=False, \
                        column_list = ['date', 'name', 'gameNumber', 'cumStatpoints'], stand=False, \
                        scale=False, index='date',feature='cumStatpoints',forecast_from='2018-10-03'):

        '''
        Calls load_data_listDS with ingested data file and appropriate parameters

        Parameters
        ---
        See above + various filepaths previously defined

        Returns
        ---
        Same ListDatasets as above
        '''
        try:
            data = self.saver_reader.read(file_ext='.csv',read_name=data_fname,full_read_dir=data_dir,bool_read_s3=False)
            full_roster = self.saver_reader.read(file_ext='.csv',read_name=roster_fname,full_read_dir=roster_dir,bool_read_s3=False)
            data = DataLoading.preprocessing(data)
            
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