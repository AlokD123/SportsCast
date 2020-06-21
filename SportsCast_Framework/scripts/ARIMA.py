from Model import Model
import pmdarima as pm
import pdb
import logging

usePDB = True


class ARIMA(Model): #TODO: decide if inherit Model or pmdarima ARIMA class

    def __init__(self,player_train_labels,features_trn=None,model=None,player_name=None):
        super().__init__()
        self.player_name = player_name if player_name is not None else "NoName"
        self.use_exog_feats = False if features_trn is None  else True
        
        if model is None:
            self.model = self.create(player_train_labels=player_train_labels,features_trn=features_trn)
        else:
            self.model = model

        #TODO: use hyperparameters to implement pre/postprocessing below
        #self.transform = transform
        #self.standardize = stand
        #self.scale = scale
        #self.transformer = None

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

    def preprocess(self,player_train_labels): #self.transformer, transform, stand, scale
        #TODO: decide if moving to Preprocessing.py
        '''
        if transform == 'log':
            # TODO: make this stat agnostic
            player_train_labels.loc[:, 'logValues'] = np.log(player_train_labels['cumStatpoints'])
        elif transform == 'yj':
            transformer = PowerTransformer()
            transformer.fit(player_train_labels.values.reshape(-1, 1))
            player_train_labels.loc[:, 'transformedValues'] = transformer \
                                                        .transform(
                                                            player_train_labels[feature] \
                                                            .values.reshape(-1, 1))
            player_train_labels.drop(feature, axis=1, inplace=True)
        '''
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
        #ppSeries = list()
        '''
        if train_predictions is not None:
            if transform == 'log':
                train_predictions = np.exp(train_predictions)
            elif transform == 'yj':
                train_predictions = transformer.inverse_transform(train_predictions.reshape(-1, 1))
        if predictions is not None:
            if transform == 'log':
                predictions = np.exp(predictions)
            elif transform == 'yj':
                predictions = transformer.inverse_transform(predictions.reshape(-1, 1))
        if intervals is not None:
            if transform == 'yj':
                low_intervals = transformer.inverse_transform(intervals[:, 0].reshape(-1, 1))
                high_intervals = transformer.inverse_transform(intervals[:, 1].reshape(-1, 1))
            else:
                if transform == 'log':
                    intervals = np.exp(intervals)
                else:
                    pass
                low_intervals = []; high_intervals = []
                for low, high in intervals:
                    low_intervals.append(low)
                    high_intervals.append(high)
        
            return train_predictions, predictions, low_intervals, high_intervals

        return train_predictions, predictions, intervals
        '''
        #TODO: remove this and decide on above
        if intervals is not None:
            low_intervals = []; high_intervals = []
            for low, high in intervals:
                low_intervals.append(low)
                high_intervals.append(high)
            return train_predictions, predictions, low_intervals, high_intervals
        else:
            return train_predictions, predictions, intervals

    def evaluate(self):
        pass

    
    def update(self,player_dict:dict):
        try:
            ret = Model.decomposeListDS_dict(player_dict,self.use_exog_feats)
            assert ret is not None, f"Failed to update ARIMA for {self.player_name}"
            new_targets, exog_feats = ret
            if exog_feats is True:
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
                for targ, exog in zip(new_targets,exog_feats): #TODO: decide if updating sequentially over scalars targ (current), or as target vector
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