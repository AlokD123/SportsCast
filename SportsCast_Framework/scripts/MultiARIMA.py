import pandas as pd
from Model import Model
from ARIMA import ARIMA
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import pdb
import logging
from Evaluate import Evaluator
import numpy as np

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
    def __init__(self,train_ds_all:ListDataset,model=None,transform:str='none'): #TODO: MUST ENSURE player_names of SAME LENGTH AS train_ds_all
        assert train_ds_all is not None, "Missing dataset for training player ARIMAs"
        self.train_ds = train_ds_all
        self.models_results_df = pd.DataFrame()
        assert 'name' in self.train_ds.list_data[0].keys(), 'Missed providing names in train_ds_all'
        self.player_names = list(pd.DataFrame(self.train_ds.list_data)['name'])

        for player_name in self.player_names:
            player_model_results_df = pd.DataFrame(ModelResultDict().dict,index=[player_name]) #Create an empty results dict for the player
            self.models_results_df = pd.concat([self.models_results_df,player_model_results_df]) #Append to df

        #Add hparams
        self.transform = transform

    def create(self,use_exog_feat=False):
        for player_dict in self.train_ds.list_data:
            player_name = player_dict['name']                           
            ret = Model.decomposeListDS_dict(player_dict,use_exog_feat)                 #Get player's training data
            if ret is None:
                logging.warning(f"Failed to create ARIMA for {player_name}")
                continue
            player_train_labels, features_trn = ret
            #Create Model-class instance of class ARIMA and store
            arima = ARIMA(player_train_labels=player_train_labels, features_trn=features_trn, \
                        player_name=player_name if player_name is not '' else None, transform=self.transform)  
            if arima is not None:
                self.models_results_df.loc[player_name,"model"] = arima   
    
    #Preprocessing for each ARIMA already done in ARIMA constructor
    def preprocess(self):
        pass

    #NOT NEEDED because done during initialization for each player
    def fit(self):
        pass

    def predict(self,predict_ds_all:ListDataset,num_per:int=None, return_conf_int:bool=True):                       #NOTE: ADDED num_per TO ALLOW FOR ITERATIVE TRAINING+EVALUATION
        for player_dict in predict_ds_all:
            n_periods = num_per if num_per is not None else len(player_dict[FieldName.TARGET])
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

    #TODO: change to postprocess-player
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

    def evaluate(self,test_ds_all:ListDataset, horizon:int=0):
        if horizon==0:
            self.predict(predict_ds_all=test_ds_all,num_per=None,return_conf_int=True)
            self.postprocess()
            for player_name in self.player_names:
                predictions = self.models_results_df.loc[player_name,"predictions"]
                targets = self.models_results_df.loc[player_name,"test_targets_unscaled"]


                ret = Evaluator.calculate_errors_trackingSig(target_data=targets,prediction_array=predictions,mase_forecast=len(targets)) #Use full sequence in calculating MASE when doing full horizon prediction
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

        else:
            for player_dict in test_ds_all:

                ###REPEATED
                player_name = player_dict['name']
                player_mdl = self.models_results_df.loc[player_name,"model"]
                ret = Model.decomposeListDS_dict(player_dict,use_exog_feats=player_mdl.use_exog_feats)
                if ret is None:
                    logging.warning(f"Failed to evaluate for {player_name}")
                    continue
                actual_targets, features_test = ret
                ###

                i =0 
                for batch in Evaluator.batch_generator(iterable = zip(features_test,actual_targets),n = horizon ): #NOTE:can't change order of loops/combine because target vectors not all same length for all players
                    exog_feat = np.array([]); label = np.array([]); #IMPORTANT - reset each time
                    logging.debug(f'Batch #{i}')
                    for idx in range(len(batch)):
                        exog_feat = np.vstack((exog_feat,batch[idx][0])) if exog_feat.size else batch[idx][0]
                        label = np.vstack((label,np.array(batch[idx][1]))) if label.size else batch[idx][1]
                    num_pers_horizon = 1 if np.isscalar(label) else len(label)

                    ###REPEATED
                    ret = player_mdl.predict(n_periods=num_pers_horizon,return_conf_int=True,exogenous=exog_feat)
                    if ret is None:
                        logging.warning(f"Failed to evaluate for {player_name}")
                        continue
                    prediction, interval = ret
                    ###

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

                ###REPEATED
                self.models_results_df.loc[player_name,"interval"] = intervals
                self.models_results_df.loc[player_name,"predictions"] = predictions
                if not ((actual_targets is None) or (len(actual_targets)==0)):
                    self.models_results_df.loc[player_name,"test_targets_unscaled"] = actual_targets
                ###
                _, self.models_results_df.loc[player_name,"predictions"], _ = player_mdl.postprocess(predictions=predictions)
                _,_, self.models_results_df.loc[player_name,"intervalLow"], self.models_results_df.loc[player_name,"intervalHigh"] = player_mdl.postprocess(intervals=intervals)
                _, self.models_results_df.loc[player_name,"test_targets_unscaled"], _ = player_mdl.postprocess(predictions=targets) #TODO: SEE ABOVEMENTIONED
                ###

                ###REPEATED
                ret = Evaluator.calculate_errors_trackingSig(target_data=actual_targets,prediction_array=predictions,mase_forecast=horizon) #Use full sequence in calculating MASE when doing full horizon prediction
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
                ###

    def update(self,new_data_ds):
        for player_dict in new_data_ds:
            player_name = player_dict['name']
            player_mdl = self.models_results_df.loc[player_name,"model"]
            #if exog_feats is None and player_mdl.use_exog_feats is not None: #NOTE: already check in Model.decomposeListDS_dict
            #    assert False, f"Missing exogenous features for {player_name}"
            player_mdl.update(player_dict=player_dict)