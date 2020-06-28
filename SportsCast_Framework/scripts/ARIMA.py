from Model import Model
import pmdarima as pm
import pdb
import logging
import numpy as np
from sklearn.preprocessing import PowerTransformer

usePDB = True


class ARIMA(Model): #TODO: decide if inherit Model or pmdarima ARIMA class

    def __init__(self,player_train_labels,features_trn=None,model=None,player_name=None,transform:str='none'):
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
        player_train_labels = self.preprocess(player_train_labels)
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

        model = self.fit(player_train_labels,features_trn=features_trn,model=model)
        return model

    def preprocess(self,player_train_labels): #self.power_transformer, transform, stand, scale
        #TODO: decide if moving to Preprocessing.py

        #By definition, only one col in df
        try:
            assert np.array(player_train_labels).shape[1]==1
        except:
            logging.warn(f'Horizontal list?')
            assert np.array(player_train_labels).reshape(-1,1) == len(np.array(player_train_labels))
            player_train_labels = np.array(player_train_labels).reshape(-1,1)
        
        if self.transform == 'log':
            # TODO: make this stat agnostic
            player_train_labels.iloc[:,0] = np.log(player_train_labels.iloc[:,0])
        elif self.transform == 'yj':
            transformer = self.power_transformer
            transformer.fit(player_train_labels.iloc[:,0].values.reshape(-1, 1))
            player_train_labels.iloc[:,0] = transformer.transform(player_train_labels.iloc[:,0].values.reshape(-1, 1))
            #player_train_labels.drop(feature, axis=1, inplace=True)

        return player_train_labels

    def fit(self,player_train_labels,features_trn=None,model=None):
        logging.info('Model built, fitting...')
        try:
            if model is not None:
                model.fit(player_train_labels,features_trn)
                return model
            else:
                self.model.fit(player_train_labels,features_trn)
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
        if self.use_exog_feats and exogenous is None:
            logging.warning(f'Missing exogenous features for prediction')
            return None
        exogenous = None if exogenous is None else exogenous.reshape(n_periods,-1)
        ret = self.model.predict(n_periods=n_periods, return_conf_int=return_conf_int, exogenous=exogenous)
        if return_conf_int:
            prediction, interval = ret
        else:
            prediction = ret
        return prediction, interval

    def postprocess(self,train_predictions=None, predictions=None, intervals=None):
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
        try:
            ret = Model.decomposeListDS_dict(player_dict,self.use_exog_feats)
            assert ret is not None, f"Failed to update ARIMA for {self.player_name}"
            new_targets, exog_feats = ret
            if self.use_exog_feats is True:
                for targ, exog in zip(new_targets,exog_feats): #TODO: decide if updating sequentially over scalars targ (current), or as target vector
                    self.model.update(targ,exogenous=exog.reshape(1,-1))
            else:
                for targ in new_targets:
                    self.model.update(targ)
        except AssertionError as err:
            logging.error(f'{err}')
            if usePDB:
                    pdb.set_trace()

    #Needed to supported legacy non-ARIMA class PMDARIMA models
    @classmethod
    def update_PMDARIMA(cls,model:pm.ARIMA,player_dict:dict,use_exog_feats:bool=True,player_name:str='NoName'):
        try:
            ret = Model.decomposeListDS_dict(player_dict,use_exog_feats)
            assert ret is not None, f"Failed to update ARIMA for {player_name}"
            new_targets, exog_feats = ret
            if exog_feats is not None:
                for targ, exog in zip(new_targets,exog_feats): #TODO: same as above
                    model.update(targ,exogenous=exog.reshape(1,-1))
            else:
                for targ in new_targets:
                    model.update(targ)
            return model, new_targets, exog_feats
        except AssertionError as err:
            logging.error(f'{err}')
            if usePDB:
                    pdb.set_trace()
            return None