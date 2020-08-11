from .Model import Model
import mxnet as mx
from mxnet import gluon
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import PowerTransformer


class DeepAR(Model):

    def __init__(self,train_ds_all:ListDataset,model=None,transform:str='none',predictor=None):
        '''
        Parameters
        ----
        train_ds_all: a special ListDataset instance for training, as defined in MXNet's GluonTS package. ListDataset contains player_dict dictonaries for each player, as defined in Model.py

        model: optional pre-existing UNTRAINED model.

        predictor: optional pre-existing TRAINED model.

        transform: transform being applied during pre-/post-processing for ALL ARIMA models. Specify as string. Currently supports 'yj' and 'log' transforms
        '''
        super().__init__()
        self.data_train = train_ds_all
        self.estimator = model
        self.predictor = predictor
        
        #Add hparams
        self.transform = transform if self.estimator is None else None
        self.power_transformer = PowerTransformer() if self.estimator is None else None
        

    def create(self,
                data_train, #ListDataset containing training data + metadata    #NOTE: provides: 'feat_dynamic_cat', 'feat_dynamic_real', 'feat_static_cat', 'name','start','target'
                save_path,  #Save location
                use_exog_feat=False, #Whether or not to use the exogenous features for modelling
                num_epochs=50,  #Number of epochs to train
                lr=1e-3,        #Learning rate
                batch_size=64,  #Batch size
                scaling=False,  #Boolean indicating whether to scale data or not
                context_length=3, #Number of samples to roll out LSTM/GRU
                num_layers=3,     #Number of RNN layers
                embedding_dimension=16, #Dimension of embeddings layer
                context='cpu',      #GPU/CPU training setting
                prediction_length=82,   #Forecast horizong
                cardinality=None,   #Number of values in each categorical feature (inferred if None)
                lags_seq=None,      #Indices of the lagged target values to use as inputs of the RNN
                dropout_rate=0.1,   #Dropout rate
                num_cells=40,       #Number of cells in model
                cell_type='lstm',   #Type (LSTM or GRU)
                num_parallel_samples=100):  #Number of parallel predictions to sample from learnt distribution

        '''
        Creates and a model for ALL the players in the training dataset

        Parameters
        ----
        As defined above

        Returns
        ----
        estimator: a DeepAREstimator instance ready to be trained
        '''

        self.data_train = data_train

        freq=data_train.list_data[0]['freq'] #Use metadata for arbitrary player to get frequency, since always same

        trainer = Trainer(batch_size=batch_size,
                            epochs=num_epochs,
                            learning_rate=lr,
                            ctx=context,
                            hybridize=False)
        estimator = DeepAREstimator(freq=freq,
                                    prediction_length=prediction_length,
                                    scaling=scaling,
                                    context_length=context_length,
                                    num_layers=num_layers,
                                    embedding_dimension=embedding_dimension,
                                    trainer=trainer,
                                    use_feat_dynamic_real=True if use_exog_feat else False,
                                    use_feat_static_cat=False,
                                    use_feat_static_real=False,
                                    cardinality=cardinality,
                                    lags_seq=lags_seq,
                                    dropout_rate=dropout_rate,
                                    num_cells=num_cells,
                                    cell_type=cell_type,
                                    num_parallel_samples=num_parallel_samples)

        self.estimator = estimator

        return estimator

    def preprocess(self,player_train_labels): #self.power_transformer, transform, stand, scale
        '''
        Helper method to preprocess data for a SINGLE player

        Parameters
        ----
        player_train_labels: labels for a single player's training

        Returns
        ----
        preprocessed labels for training
        '''

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

        return player_train_labels

    def fit(self):
        '''
        Parameters
        -----
        None

        Returns
        -----
        predictor: trained model
        '''
        self.predictor = self.estimator.train(self.data_train)
        
    
    def predict(self,num_per=None,return_conf_int=True):
        '''
        Parameters
        -----
        num_per: unused, since constant
        return_conf_int: unused, since always True

        Returns
        -----
        pred_generator: predictions generator
        '''
        pred_generator = self.predictor.predict(self.data_train)
        return pred_generator
        #TODO: add boolPredictInsample option

    def postprocess(self,targets=None, predictions=None, intervals=None):
        #TODO: clean up
        #TODO: postprocess for scale/stand

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

        for val in [targets,predictions]:
            if val is not None:

                #Reshape valiction vectors
                val = np.array(val).reshape(-1,1)
                if len(np.array(val).shape)>2:
                    val = np.array(val)[0] 

                #Transform
                if self.transform == 'log':
                    val = np.exp(val)
                elif self.transform == 'yj':
                    val = self.power_transformer.inverse_transform(val.reshape(-1, 1))
        
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
        
            return targets, predictions, low_intervals, high_intervals

        return targets, predictions, intervals


    def process_prediction(self,prediction):
        ''' Processes predictions for all players '''
        mean = prediction.mean_ts
        mean = mean.reset_index()
        mean = mean.rename(columns={0: 'predictions'})
        mean = mean.rename(columns={'index': 'date'})
        mean = mean.drop(columns=['date'])
        mean['gameNumber'] = mean.index + 1
        conf = pd.DataFrame()
        conf.loc[:, 'low'] = prediction.quantile('0.05')
        conf.loc[:, 'high'] = prediction.quantile('0.95')
        full_df = pd.concat([mean, conf], axis=1)
        return full_df

    def generate_prediction_df(self,predictions, data, drop=True, target='cumStatpoints', scaled=None, scaling_loc=None):
        ''' Postprocess predictions for ALL players and return as df '''
        if scaled is not None:
            scaling_meta = pd.read_pickle(scaling_loc)
            print(scaling_meta)
        names = data.loc[:, 'name'].unique()
        full_predictions = pd.DataFrame()
        for prediction, name in zip(predictions, names): #ONE FORECAST OF LENGTH prediction_length PER PLAYER, in order of data['name']
            player_df = pd.DataFrame()
            player_data = data.loc[data.loc[:, 'name'] == name].loc[:, ['name', 'gameNumber', target]] #DF OF 'name', 'date', 'cumStatpoints' for ONE PLAYER
            
            data_length = player_data.shape[0]
            prediction_df = self.process_prediction(prediction)
            if drop:
                prediction_df = prediction_df.iloc[:data_length, :] #Drop excess predictions if no data available for evaluation
            player_data.reset_index(drop=True, inplace=True)
            prediction_df.reset_index(drop=True, inplace=True)
            if scaled == 'ss':
                scale_data = scaling_meta.loc[scaling_meta.loc[:, 'name'] == name]
                for column in ['predictions', 'low', 'high']:
                    prediction_df.loc[:, column] = ((prediction_df.loc[:, column] * scale_data['maxabs']) \
                                                * scale_data['std']) + scale_data['mean']
            elif scaled == 'unit':
                scale_data = scaling_meta.loc[scaling_meta.loc[:, 'name'] == name]
                for column in ['predictions', 'low', 'high']:
                    prediction_df.loc[:, column] = (prediction_df.loc[:, column] - scale_data['min'].values) / scale_data['scale'].values
           
            player_data_df = pd.concat([player_data, prediction_df], axis=1)
            full_predictions = pd.concat([full_predictions, player_data_df])
        return full_predictions


    #NOTE: Not possible to implement at this point. See presentation for details
    def update(self,new_data_ds):
        pass

    #NOTE: not impelemented because update not possibel
    def evaluate(self,test_ds_all:ListDataset, horizon:int=0):
        pass