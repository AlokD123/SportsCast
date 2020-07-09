from Model import Model
import pmdarima as pm
import pdb
import logging
import numpy as np
from sklearn.preprocessing import PowerTransformer



class ARIMA(Model): #TODO: decide if inherit Model or pmdarima ARIMA class
    ''' A class for a player's ARIMA forecasting model. Implements Model interface '''

    def __init__(self,player_train_labels,features_trn=None,model=None,player_name=None,transform:str='none'):
        '''
        Parameters
        -----
        player_train_labels: training data for one player (vector of cumulative points over training set timespan)

        features_trn: array of exogenous features, optional. 2D-array, with same length as labels. (Width depends of number of features)

        model: optional pre-existing model. To use if just want to use this class as a wrapper instead of a model instantiator+trainer

        player_name: name of player associated with this model

        transform: transform being applied during pre-/post-processing. Specify as string. Currently supports 'yj' and 'log' transforms
        '''
        super().__init__()
        self.player_name = player_name if player_name is not None else "NoName"
        self.use_exog_feats = False if features_trn is None  else True
        
        self.transform = transform if model is None else 'none'
        self.power_transformer = PowerTransformer() if model is None else None
        #TODO: use hyperparameters to implement pre/postprocessing below
        self.scaling_transformer = None if model is None else None
        self.standardize = False if model is None else False            #stand
        self.stdize_transformer = None if model is None else None
        self.scale = False if model is None else False                  #scale

        if model is None:
            self.model = self.create(player_train_labels=player_train_labels,features_trn=features_trn)
        else:
            self.model = model

    #TODO: make hparams tunable
    def create(self,player_train_labels,features_trn=None):
        '''
        Parameters
        -----
        player_train_labels: see above

        features_trn: see above

        Returns
        ----
        model: trained model
        '''
        player_train_labels = self.preprocess(player_train_labels) #Preprocess training data

        #Create model and associated bounds for hyperparameter grid search (out-of-sample validation)
        model = pm.auto_arima(player_train_labels,
                                exogenous=features_trn,
                                start_p=1,
                                start_q=1,
                                max_p=5,
                                max_q=5,
                                max_d=3,
                                m=3,
                                start_P=0,
                                start_Q=0,
                                seasonal=True,
                                information_criterion='aicc',
                                error_action='ignore',         #TODO: consider 'warn', with return val None
                                trace=False,
                                suppress_warnings=False,
                                stepwise=True,
                                out_of_sample_size=int(0.1*len(player_train_labels)), #Validation set size
                                scoring='mae')

        #MAIN: train and validate model
        model = self.fit(player_train_labels,features_trn=features_trn,model=model)
        return model

    def preprocess(self,player_train_labels): #self.scaling_transformer,self.stdize_transformer transform, stand, scale
        '''
        Parameters
        -----
        player_train_labels: see above

        Returns
        ----
        player_train_labels: preprocessed labels for training
        '''

        #By definition, enforce only one col in df
        try:
            assert np.array(player_train_labels).shape[1]==1
        except:
            logging.warn(f'Horizontal list?')
            assert np.array(player_train_labels).reshape(-1,1) == len(np.array(player_train_labels))
            player_train_labels = np.array(player_train_labels).reshape(-1,1)
        
        if self.transform == 'log':
            player_train_labels.iloc[:,0] = np.log(player_train_labels.iloc[:,0]) #Transform by log to normalize data
        elif self.transform == 'yj':
            transformer = self.power_transformer
            transformer.fit(player_train_labels.iloc[:,0].values.reshape(-1, 1))
            player_train_labels.iloc[:,0] = transformer.transform(player_train_labels.iloc[:,0].values.reshape(-1, 1)) #Transform using Yeo-Johnson to normalize data

        return player_train_labels

    def fit(self,player_train_labels,features_trn=None,model=None):
        '''
        Parameters
        -----
        player_train_labels: see above

        features_trn: see above

        model: see above

        Returns
        -----
        ARIMA instance reference if succeeded. Otherwise, null.
        '''
        logging.info('Model built, fitting...')
        try:
            if model is not None:                               #For a pretrained model, continue training
                model.fit(player_train_labels,features_trn)
                return model
            else:
                self.model.fit(player_train_labels,features_trn) #Train from scratch
                return self
        except ValueError:
            logging.error(f"{self.player_name} doesn't have enough data for fitting ARIMA!")
            return None
        except IndexError:
            logging.error(f'Index error in fitting ARIMA for {self.player_name}')
            return None
        except Exception as err:
            logging.error(f'Other error in fitting ARIMA for {self.player_name}:{err}')
            return None
    

    #TODO: add boolPredictInsample option
    def predict(self,n_periods:int,return_conf_int:bool=True,exogenous=None):
        '''
        Parameters
        -----
        n_periods: number time periods ahead to forecast

        return_conf_int: boolean indicating whether to also return confidence intervals for prediction

        exogenous: exogenous features for testing/inference. Must be provided if trained using exogenous features

        Returns
        -----
        prediction: vector of forecasts for player

        interval: 2D array containing lower- and upper-bounds for confidence interval (one for each prediction in forecast)
        '''
        if self.use_exog_feats and exogenous is None:
            logging.warning(f'Missing exogenous features for prediction')
            return None
        exogenous = None if exogenous is None else exogenous.reshape(n_periods,-1)
        ret = self.model.predict(n_periods=n_periods, return_conf_int=return_conf_int, exogenous=exogenous) #Make prediction(s)
        if return_conf_int:
            prediction, interval = ret
        else:
            prediction = ret
        return prediction, interval

    def postprocess(self,train_predictions=None, predictions=None, intervals=None):
        '''
        Parameters
        -----
        train_predictions: vector of forecasts after training

        predictions: vector of forecasts during inference

        intervals: see above

        Returns
        -----
        train_predictions: post-processed version of the above

        predictions: post-processed version of the above

        low_intervals: vector of lower bounds for confidence intervals

        high_intervals: vector of upper bounds for confidence intervals
        '''
        #TODO: clean up
        #TODO: postprocess for scale/stand

        for pred in [train_predictions,predictions]:
            if pred is not None:

                #Reshape prediction vectors
                pred = np.array(pred).reshape(-1,1)
                if len(np.array(pred).shape)>2:
                    pred = np.array(pred)[0] 

                #Transform
                if self.transform == 'log':
                    pred = np.exp(pred)
                elif self.transform == 'yj':
                    pred = self.power_transformer.inverse_transform(pred.reshape(-1, 1))
        
        if intervals is not None:
            #Reshape array of prediction confidence intervals
            intervals = np.array(intervals).reshape(-1,2)
            if len(np.array(intervals).shape)>3:
                intervals = np.array(intervals)[0]

            #Transform and decompose
            if self.transform == 'yj':
                low_intervals = self.power_transformer.inverse_transform(intervals[:, 0].reshape(-1, 1))
                high_intervals = self.power_transformer.inverse_transform(intervals[:, 1].reshape(-1, 1))
            else:
                if self.transform == 'log':
                    intervals = np.exp(intervals)
                else:
                    pass
                #Decompose into lower and upper bounds
                low_intervals = []; high_intervals = []
                for low, high in intervals:
                    low_intervals.append(low)
                    high_intervals.append(high)
        
            return train_predictions, predictions, low_intervals, high_intervals

        return train_predictions, predictions, intervals



    #Unused because already implemented in MultiARIMA
    def evaluate(self):
        pass

    
    def update(self,player_dict:dict):
        '''
        Parameters
        -----
        player_dict: dictionary of packed retraining data for player. See def'n in Model.py

        Returns
        -----
        None
        '''

        try:
            ret = Model.decomposeListDS_dict(player_dict,self.use_exog_feats)       #Get retraining data
            assert ret is not None, f"Failed to update ARIMA for {self.player_name}"
            new_targets, exog_feats = ret
            if self.use_exog_feats is True:
                for targ, exog in zip(new_targets,exog_feats):
                    self.model.update(targ,exogenous=exog.reshape(1,-1)) #For each retraining example, re-fit model with it
            else:
                for targ in new_targets:
                    self.model.update(targ)
        except AssertionError as err:
            logging.error(f'{err}')

    @classmethod
    def update_PMDARIMA(cls,model:pm.ARIMA,player_dict:dict,use_exog_feats:bool=True,player_name:str='NoName'):
        '''
        Specifically needed to supported legacy non-ARIMA class PMDARIMA models. Retrains model, as above.

        Parameters
        -----
        model: a pretrained legacy model

        player_dict: see above

        use_exog_feats: flag indicating whether to use exogenous features when retraining

        player_name: see above

        Returns
        -----
        None
        '''

        try:
            ret = Model.decomposeListDS_dict(player_dict,use_exog_feats)
            assert ret is not None, f"Failed to update ARIMA for {player_name}"
            new_targets, exog_feats = ret
            if exog_feats is not None:
                for targ, exog in zip(new_targets,exog_feats):
                    model.update(targ,exogenous=exog.reshape(1,-1))
            else:
                for targ in new_targets:
                    model.update(targ)
            return model, new_targets, exog_feats
        except AssertionError as err:
            logging.error(f'{err}')
            return None