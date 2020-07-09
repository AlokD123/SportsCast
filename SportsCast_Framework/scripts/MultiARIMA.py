import pandas as pd
from Model import Model
from ARIMA import ARIMA
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import pdb
import logging
from Evaluate import Evaluator
import numpy as np


class ModelResultDict:
    ''' Primitive class just to define a special dictionay containing a pre-trained model and its results (for model-type-agnostic inference)'''

    def __init__(self,
                model=None,
                test_targets=None,
                predictions=None,
                forecast_from=None,
                test_mfe=None,
                test_mae=None,
                test_rmse=None,
                test_mase=None,
                test_trackSig=None,
                prediction_residuals=None,
                low_intervals=None,
                high_intervals=None,
                intervals=None
                ):
    
        '''
        Parameters
        -----
        Various parameters, as define below

        '''

        self.dict = {'model': model,            #An ARIMA Model interface instance. Actual PMDARIMA model contained within (see ARIMA.py)
                    'test_targets_unscaled': [test_targets], #Raw test data
                    'predictions': [predictions],   #Out-of-sample predictions
                    'forecastStart':forecast_from,  #Date of train-test split
                    #Various accuracy metrics
                    'testMfe':test_mfe, #Mean forecasting error
                    'testMae':test_mae, #Mean abs error
                    'testRmse':test_rmse,   #RMS error
                    'test_Mase': test_mase, #Mean abs scaled error ***Used in this project***
                    'test_trackSig':test_trackSig,      #A tracking signal metric
                    'testResiduals':[prediction_residuals], #Test residuals
                    'intervalLow':[low_intervals],  #Lower bound on prediction confidence intervals
                    'intervalHigh':[high_intervals],#Upper bound on prediction confidence intervals
                    'interval': [intervals]         #Prediction confidence intervals
                    }

class MultiARIMA(Model):
    ''' 
    Muti-player ARIMA classe: an aggregate class to hold ARIMA models for all players.
    Aggregation allows for intechangeability with other multi-player forecasting classes (e.g. DeepAR), since this also implements the Model interface
    '''

    def __init__(self,train_ds_all:ListDataset,model=None,transform:str='none'):
        '''
        Parameters
        ----
        train_ds_all: a special ListDataset instance for training, as defined in MXNet's GluonTS package. ListDataset contains player_dict dictonaries for each player, as defined in Model.py

        model: optional pre-existing MultiARIMA model.

        transform: transform being applied during pre-/post-processing for ALL ARIMA models. Specify as string. Currently supports 'yj' and 'log' transforms
        '''

        assert train_ds_all is not None, "Missing dataset for training player ARIMAs"
        self.train_ds = train_ds_all
        self.models_results_df = pd.DataFrame()             #IMPORTANT: a dataframe to contain trained models and results for EACH player, AS DEFINED IN ModelResultDict. Most important structure
        assert 'name' in self.train_ds.list_data[0].keys(), 'Missed providing names in train_ds_all'
        self.player_names = list(pd.DataFrame(self.train_ds.list_data)['name'])

        for player_name in self.player_names:
            player_model_results_df = pd.DataFrame(ModelResultDict().dict,index=[player_name]) #Create an empty results dict for the player
            self.models_results_df = pd.concat([self.models_results_df,player_model_results_df]) #Append to df

        #Add hparams
        self.transform = transform

    def create(self,use_exog_feat=False):
        '''
        Creates and trains ARIMA models for ALL the players in the training dataset

        Parameters
        ----
        use_exog_feat: flag for whether to train/infer with ALL models using exogenous features

        Returns
        ----
        boolean indicating success/failure
        '''
        for player_dict in self.train_ds.list_data:
            ret = self.__decompose_player_dict(player_dict,use_exog_feats=True)
            if ret is None:
                return False
            if use_exog_feat:
                player_train_labels, features_trn, player_name = ret
            else:
                player_train_labels, player_name = ret
                features_trn = False
            #Create Model-class instance of class ARIMA and store
            arima = ARIMA(player_train_labels=player_train_labels, features_trn=features_trn, \
                        player_name=player_name if player_name is not '' else None, transform=self.transform)  
            if arima is not None:
                self.models_results_df.loc[player_name,"model"] = arima

            return True   
    
    def preprocess(self,targets,player_mdl:ARIMA):
        '''
        Helper method to preprocess data for a SINGLE player

        Parameters
        ----
        targets: labels for a single player's training

        player_mdl: player's ARIMA model

        Returns
        ----
        preprocessed labels for training
        '''
        return player_mdl.preprocess(targets)


    #NOT NEEDED because done during initialization for each player
    def fit(self):
        pass

    def predict(self,player_dict:dict,num_per:int=None, return_conf_int:bool=True):  #NOTE: ADDED num_per TO ALLOW FOR ITERATIVE TRAINING+EVALUATION
        '''
        Helper method to perform prediction for a SINGLE player

        Parameters
        ----
        player_dict: dictionary of packed retraining data for player. See def'n in Model.py

        num_per: number time periods ahead to forecast

        return_conf_int: boolean indicating whether to also return confidence intervals for prediction

        Returns
        ----
        prediction: vector of forecasts for player

        interval: 2D array containing lower- and upper-bounds for confidence interval (one for each prediction in forecast)
        '''
        n_periods = num_per if num_per is not None else len(player_dict[FieldName.TARGET])
        ret = self.__decompose_player_dict(player_dict)
        if ret is None:
            return False
        actual_targets, features_predict, player_name, player_mdl = ret
        
        ret = self.__predict_player(n_periods,return_conf_int,features_predict,player_name,player_mdl)
        if return_conf_int:
            prediction, interval = ret
        else:
            prediction = ret
        self.models_results_df.loc[player_name,"predictions"] = prediction.reshape(-1,1)
        if not ((actual_targets is None) or (len(actual_targets)==0)):
            self.models_results_df.loc[player_name,"test_targets_unscaled"] = actual_targets.reshape(-1,1)

        return ret, actual_targets


    def postprocess(self,player_name:str,predictions,targets,intervals=None):
        '''
        Helper method to postprocess data for a SINGLE player

        Parameters
        ----
        targets: labels for a single player's training

        intervals: see above

        predictions: see above

        Returns
        ----
        post-processed versions of each of the above
        '''

        player_mdl = self.models_results_df.loc[player_name,"model"]

        proc_predictions = None
        proc_int_low = None; proc_int_high = None
        proc_targets = None
        
        if predictions is not None:
            _, proc_predictions, _ = player_mdl.postprocess(predictions=predictions)
            self.models_results_df.loc[player_name,"predictions"] = proc_predictions
        if intervals is not None:
            _,_, proc_int_low, proc_int_high = player_mdl.postprocess(intervals=intervals)
            self.models_results_df.loc[player_name,"intervalLow"] = proc_int_low
            self.models_results_df.loc[player_name,"intervalHigh"] = proc_int_high
        if (targets is not None) and (len(targets)>0):
            _, proc_targets, _ = player_mdl.postprocess(predictions=targets)
            self.models_results_df.loc[player_name,"test_targets_unscaled"] = proc_targets  #TODO: add dedicated field for test_targets_unscaled OR don't preprocess in glnts.prep_df

        return proc_predictions,proc_int_low,proc_int_high,proc_targets


    def evaluate(self,test_ds_all:ListDataset, horizon:int=0):
        '''
        Performs aggregate evaluation for ALL the players in the testing dataset. IMPORTANT: two modes of testing, depending on horizon

        Parameters
        ----
        test_ds_all: similar to train_ds_all, but for evaluation

        horizon: number of periods to forecast at each iteration in retraining/testing mode of evaluation (Mode 1).
        If horizon=0, doesn't re-fit model - just tests prediction for full duration of test dataset (Mode 2).
        '''

        assert len(self.player_names)==len(test_ds_all.list_data), 'Mismatch in # of players vs # of datasets' #TODO: maybe allow for mismatch later

        #For each player in aggregate model...
        for player_name,player_dict in zip(self.player_names, test_ds_all):
            if horizon==0: #Mode 2
            
                ret = self.predict(player_dict=player_dict,num_per=None,return_conf_int=True)   #Predict
                if ret is None:
                    continue
                predictions, intervals, actual_targets = ret
                proc_predictions,_,_,_ = self.postprocess(player_name,predictions, None, intervals) #Postprocess predictions

                self.__calc_save_error_player(target_data=actual_targets,predictions=proc_predictions,horizon=len(actual_targets),player_name=player_name) #Get error metrics using full horizon

            else: #Mode 1
                #Get testing data
                ret = self.__decompose_player_dict(player_dict)
                if ret is None:
                    continue
                actual_targets, features_test, _, player_mdl = ret

                preproc_targets = self.preprocess(actual_targets,player_mdl) #Must preprocess here (unlike above) since re-fitting

                #Iterate through forecasting+re-training over each batch of train dataset of length <=horizon. Store prediction for later evaluation
                i =0 
                for batch in Evaluator.batch_generator(iterable = zip(features_test,preproc_targets),n = horizon ): #NOTE:can't change order of loops/combine because target vectors not all same length for all players
                    exog_feat = np.array([]); label = np.array([]); #IMPORTANT - reset each time
                    logging.debug(f'Batch #{i}')

                    #Batch labels+features
                    for idx in range(len(batch)):
                        exog_feat = np.vstack((exog_feat,batch[idx][0])) if exog_feat.size else batch[idx][0]
                        label = np.vstack((label,np.array(batch[idx][1]))) if label.size else batch[idx][1]
                    num_pers_horizon = 1 if np.isscalar(label) else len(label)

                    #Predict for batch
                    ret = self.__predict_player(n_periods=num_pers_horizon,return_conf_int=True,features_predict=features_test,player_name=player_name,player_mdl=player_mdl)
                    if ret is None:
                        continue
                    prediction, interval = ret

                    #Re-train
                    if np.isscalar(label):
                        logging.debug(f'Label #{idx} of 1')
                        bat = (exog_feat,label)
                        exg_ft = bat[0]; l = bat[1]
                        player_mdl.update(l,exogenous=exg_ft,maxiter=1)
                    else:
                        for idx, bat in enumerate(zip(exog_feat,label)):
                            logging.debug(f'Label #{idx} of {len(label)}')
                            exg_ft = bat[0]; l = bat[1]
                            player_mdl.update(l,exogenous=exg_ft,maxiter=1)                                             #Update endog vector manually
                    predictions = np.vstack([predictions,prediction]) if predictions.size else prediction
                    intervals = np.vstack([intervals,interval]) if intervals.size else interval
                    i = i+1
                predictions = predictions.reshape(-1,1); intervals = intervals.reshape(-1,2)

                proc_predictions,_,_,_ = self.postprocess(player_name,predictions,None,intervals) #Postprocess predictions

                #Run evaluation for players forecasts in Mode 1
                self.__calc_save_error_player(target_data=actual_targets,predictions=proc_predictions,horizon=horizon,player_name=player_name)


    def update(self,new_data_ds):
        ''' 
        Retraining each player's model using new data

        Parameters
        ----
        new_data_ds: ListDataset for retraining data

        Returns
        ----
        None
        '''
        for player_dict in new_data_ds:
            player_name = player_dict['name']
            player_mdl = self.models_results_df.loc[player_name,"model"]
            player_mdl.update(player_dict=player_dict)

        
    #HELPER FUNCTIONS
    def __decompose_player_dict(self,player_dict:dict,use_exog_feats:bool=None):
        ''' 
        Decomposes data stored for a SINGLE player

        Parameters
        ----
        player_dict: see above

        use_exog_feats: see above

        Returns
        ----
        ret: 2-tuple containing feature,label lists. 

        player_mdl: model for player
        '''
        player_name = player_dict['name']
        #TODO: account for other combinations (e.g. use_exog_feats is None and self.models_results_df.loc[player_name,"model"] is None)
        if use_exog_feats is None:
            player_mdl = self.models_results_df.loc[player_name,"model"]
            ret = Model.decomposeListDS_dict(player_dict,use_exog_feats=player_mdl.use_exog_feats) #Get player's training data
            if ret is None:
                logging.warning(f"Failed to perform prediction for {player_name}")
            return ret, player_name, player_mdl
        else:
            ret = Model.decomposeListDS_dict(player_dict,use_exog_feats=use_exog_feats)
            if ret is None:
                logging.warning(f"Failed to perform prediction for {player_name}")
            return ret, player_name

    def __predict_player(self,n_periods,return_conf_int,features_predict,player_name:str,player_mdl):
        ''' 
        Forecasts for a SINGLE player

        Parameters
        ----
        n_periods: number of forecast periods

        return_conf_int: see above

        features_predict: exogenous features (2D array). See ARIMA.py for details

        Returns
        ----
        prediction: see above

        interval: see above
        '''

        assert player_mdl is not None, f"Missing model for {player_name}. Can't predict"
        ret = player_mdl.predict(n_periods=n_periods,return_conf_int=return_conf_int,exogenous=features_predict)
        if ret is None:
            logging.warning(f"Failed to evaluate for {player_name}")
        if return_conf_int:
            prediction, interval = ret
            self.models_results_df.loc[player_name,"interval"] = interval
            return prediction, interval
        else:
            prediction = ret
            return prediction

    def __calc_save_error_player(self,target_data,predictions,horizon,player_name):
        ''' 
        Calculates testing errors and saves them to the models_results dataframe

        Parameters
        ----
        target_data: labels in test dataset

        predictions: predictions made on test dataset using Mode 1 or Mode 2 testing

        horizon: see above

        Returns
        ----
        None

        '''

        #Calculate metrics indentified in ModelResultDict
        ret = Evaluator.calculate_errors_trackingSig(target_data=target_data,prediction_array=predictions,mase_forecast=horizon) #Use full sequence in calculating MASE when doing full horizon prediction

        #Store evaluation metrics
        if ret is None:
            logging.error(f'Couldnt calculate metrics for {player_name}')
        else:
            test_mfe, test_mae, test_rmse, test_mase, test_trackSig, prediction_residuals = ret
            self.models_results_df.loc[player_name,"testMfe"] = test_mfe
            self.models_results_df.loc[player_name,"testMae"] = test_mae
            self.models_results_df.loc[player_name,"testRmse"] = test_rmse
            self.models_results_df.loc[player_name,"test_Mase"] = test_mase
            self.models_results_df.loc[player_name,"test_trackSig"] = test_trackSig
            self.models_results_df.loc[player_name,"testResiduals"] = prediction_residuals
