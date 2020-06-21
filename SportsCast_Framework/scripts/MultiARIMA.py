import pandas as pd
from Model import Model
from ARIMA import ARIMA
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import pdb
import logging

usePDB=True

class ModelResultDict:
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
    
        self.dict = {'model': model, #An ARIMA Model interface instance. Actual PMDARIMA model contained within the ARIMA instance
                    #'train_targest_scaled':
                    'test_targets_unscaled': [test_targets],
                    'predictions': [predictions],
                    'forecastStart':forecast_from,
                    'testMfe':test_mfe,
                    'testMae':test_mae,
                    'testRmse':test_rmse,
                    'test_Mase': test_mase,
                    'test_trackSig':test_trackSig,
                    'testResiduals':[prediction_residuals],
                    'intervalLow':[low_intervals],
                    'intervalHigh':[high_intervals],
                    'interval': [intervals]
                    }
        

class MultiARIMA(Model):
    def __init__(self,train_ds_all:ListDataset,model=None): #TODO: MUST ENSURE player_names of SAME LENGTH AS train_ds_all
        assert train_ds_all is not None, "Missing dataset for training player ARIMAs"
        self.train_ds = train_ds_all
        self.models_results_df = pd.DataFrame()
        assert 'name' in self.train_ds.list_data[0].keys(), 'Missed providing names in train_ds_all'
        self.player_names = list(pd.DataFrame(self.train_ds.list_data)['name'])
        
        for player_name in self.player_names:
            player_model_results_df = pd.DataFrame(ModelResultDict().dict,index=[player_name]) #Create an empty results dict for the player
            self.models_results_df = pd.concat([self.models_results_df,player_model_results_df]) #Append to df

    def create(self,use_exog_feat=False):
        for player_dict in self.train_ds.list_data:
            player_name = player_dict['name']                           
            ret = Model.decomposeListDS_dict(player_dict,use_exog_feat)                 #Get player's training data
            if ret is None:
                logging.warning(f"Failed to create ARIMA for {player_name}")
                continue
            player_train_labels, features_trn = ret
            #Create Model-class instance of class ARIMA and store
            self.models_results_df.loc[player_name,"model"] = ARIMA(player_train_labels=player_train_labels,features_trn=features_trn,player_name=player_name if player_name is not '' else None)     
    
    #Preprocessing for each ARIMA already done in ARIMA.create()
    def preprocess(self):
        pass

    #NOT NEEDED because done during initialization for each player
    def fit(self):
        pass

    def predict(self,predict_ds_all:ListDataset,return_conf_int:bool=True):
        for player_dict in predict_ds_all:
            n_periods = len(player_dict[FieldName.TARGET])
            player_name = player_dict['name']
            player_mdl = self.models_results_df.loc[player_name,"model"]
            ret = Model.decomposeListDS_dict(player_dict,use_exog_feats=player_mdl.use_exog_feats)
            if ret is None:
                logging.warning(f"Failed to perform prediction for {player_name}")
                continue
            actual_targets, features_predict = ret
            ret = player_mdl.predict(n_periods=n_periods,return_conf_int=return_conf_int,exogenous=features_predict)
            if return_conf_int:
                prediction, interval = ret
                self.models_results_df.loc[player_name,"interval"] = interval
            else:
                prediction = ret
            self.models_results_df.loc[player_name,"predictions"] = prediction.reshape(-1,1)
            if not ((actual_targets is None) or (len(actual_targets)==0)):
                self.models_results_df.loc[player_name,"test_targets_unscaled"] = actual_targets.reshape(-1,1)

    def postprocess(self):
        for player_name in self.player_names:
            player_mdl = self.models_results_df.loc[player_name,"model"]
            predictions = self.models_results_df.loc[player_name,"predictions"]
            intervals = self.models_results_df.loc[player_name,"interval"]
            targets = self.models_results_df.loc[player_name,"test_targets_unscaled"]
            if predictions is not None:
                _, self.models_results_df.loc[player_name,"predictions"], _ = player_mdl.postprocess(predictions=predictions)
            if intervals is not None:
                _,_, self.models_results_df.loc[player_name,"intervalLow"], \
                self.models_results_df.loc[player_name,"intervalHigh"] = \
                player_mdl.postprocess(intervals=intervals)
            if (targets is not None) and (len(targets)>0):
                _, self.models_results_df.loc[player_name,"test_targets_unscaled"], _ = player_mdl.postprocess(predictions=targets) #TODO: add dedicated field for test_targets_unscaled OR don't preprocess in glnts.prep_df
        return

    def evaluate(self):
        pass

    #Unused, since updates will be done model-by-model during retraining
    def update(self,new_data_ds,exog_feats=None):
        pass