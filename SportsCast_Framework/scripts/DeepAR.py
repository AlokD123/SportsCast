from Model import Model
import mxnet as mx
from mxnet import gluon
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import PowerTransformer


class DeepAR(Model):

    def __init__(self,estimator=None,predictor=None,transform:str='none'):
        super().__init__()
        self.data_train = None
        self.data_meta = None
        self.estimator = estimator
        self.predictor = predictor
        
        #Add hparams
        self.transform = transform if estimator is None else None
        self.power_transformer = PowerTransformer() if estimator is None else None
        

    def create(self,
                data_train, #NOTE: provides: 'feat_dynamic_cat', 'feat_dynamic_real', 'feat_static_cat', 'name','start','target'
                data_meta,
                save_path,
                use_exog_feat=False,
                num_epochs=50,
                lr=1e-3,
                batch_size=64,
                scaling=False,
                context_length=3,
                num_layers=3,
                embedding_dimension=16,
                context='cpu',
                prediction_length=82,
                cardinality=None,
                lags_seq=None,
                dropout_rate=0.1,
                num_cells=40,
                cell_type='lstm',
                num_parallel_samples=100):

        self.data_train = data_train
        self.data_meta = data_meta

        #TODO: decide if save
        #pd.to_pickle(data_meta, save_path+'/data_meta.p')

        trainer = Trainer(batch_size=batch_size,
                            epochs=num_epochs,
                            learning_rate=lr,
                            ctx=context,
                            hybridize=False)
        estimator = DeepAREstimator(freq=data_meta['freq'],
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

    def fit(self):
        self.predictor = self.estimator.train(self.data_train)
        
    
    def predict(self):
        pred_generator = self.predictor.predict(self.data_train)
        return pred_generator
        #TODO: add boolPredictInsample option

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


    def process_prediction(self,prediction):
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

    def generate_prediction_df(self,predictions, train_data, test_data, drop=True, target='cumStatpoints', scaled=None, scaling_loc=None):
        if scaled is not None:
            scaling_meta = pd.read_pickle(scaling_loc)
            print(scaling_meta)
        names = test_data.loc[:, 'name'].unique()
        full_predictions = pd.DataFrame()
        for prediction, name in zip(predictions, names): #ONE FORECAST OF LENGTH prediction_length PER PLAYER, in order of test_data['name']
            player_df = pd.DataFrame()
            player_test_data = test_data.loc[test_data.loc[:, 'name'] == name].loc[:, ['name', 'gameNumber', target]] #DF OF 'name', 'date', 'cumStatpoints' for ONE PLAYER
            #player_test_data.loc[:, 'date'] = pd.to_datetime(player_test_data.loc[:, 'date'])
            
            test_length = player_test_data.shape[0]
            prediction_df = self.process_prediction(prediction)
            # prediction_df.loc[:, 'name'] = name
            if drop:
                prediction_df = prediction_df.iloc[:test_length, :] #Drop excess predictions if no test data available for evaluation
            player_test_data.reset_index(drop=True, inplace=True)
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
           
            player_test_df = pd.concat([player_test_data, prediction_df], axis=1)
            full_predictions = pd.concat([full_predictions, player_test_df])
        return full_predictions


    #Not possible to implement at this point. See presentation for details
    def update(self):
        pass