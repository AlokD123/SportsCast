from abc import ABC, abstractmethod
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import numpy as np
import pdb
import logging


def reshape_arr_vertical(arr):
    ''' Reshapes horizontal to vertical '''
    if arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr)
    return arr

class Model(ABC):
    '''
    Abstract class to specify the interface for a MULTI_PLAYER forecasting model in this project.
    Allows for intechanging models on-the-fly during training/evaluation (see TrainingEvaluation.py)
    '''
    def __init__(self,*args,**kwargs):
        pass
    @abstractmethod
    def create(self,*args,**kwargs):
        pass
    @abstractmethod
    def preprocess(self,player_train_labels,*args,**kwargs):
        pass
    @abstractmethod
    def fit(self,*args,**kwargs):
        pass
    @abstractmethod
    def predict(self,num_per,return_conf_int,*args,**kwargs):
        pass
    @abstractmethod
    def postprocess(self,targets,predictions,intervals,*args,**kwargs):
        pass
    @abstractmethod
    def evaluate(self,test_ds_all, horizon,*args,**kwargs):
        pass
    @abstractmethod
    def update(self,new_data_ds,*args,**kwargs):
        pass

    @classmethod
    def decomposeListDS_dict(cls,player_dict:dict,use_exog_feats=False):
        '''
        Decomposes a player_dict dictionary contained in a ListDataset instance

        Parameters
        ----
        player_dict: dictionary of labels ('targets'), real-valued and categorical features in dataset for a player. See glnts.py for details

        use_exog_feats: flag of whether to return these exogenous features in addition to the labels.

        Returns
        ----
        player_labels: labels vector

        player_features: features as an nd-array
        '''
        player_labels = player_dict[FieldName.TARGET]
        if use_exog_feats:

            #Preprocess exogenous features
            try:
                dyn_real_features = player_dict[FieldName.FEAT_DYNAMIC_REAL]
                dyn_cat_features = player_dict[FieldName.FEAT_DYNAMIC_CAT]
                assert len(dyn_real_features)>0, "Missing dyn_real_features"
                assert len(dyn_real_features)>0, "Missing dyn_cat_features"
            except AssertionError as err:
                logging.error(f'Error in decomposing DS: {err}')
                return None
            dyn_cat_features = reshape_arr_vertical(np.array(dyn_cat_features))
            dyn_real_features = reshape_arr_vertical(np.array(dyn_real_features))

            #Try to concatenate
            try:
                player_features = np.hstack( ( np.array(dyn_cat_features), np.array(dyn_real_features) ) )
            except ValueError:
                try:
                    player_features = np.hstack( ( np.array(dyn_cat_features).T, np.array(dyn_real_features).T ) ) #Legacy handling
                except Exception as err:
                    raise Exception(f"{err}")
            except Exception as err:
                logging.error(f'Error in decomposing DS: couldnt concatenate feature arrays: {err}')
                logging.error(f'\ndyn_cat_features: {np.array(dyn_cat_features)}. Shape: {np.array(dyn_cat_features).shape}')
                logging.error(f'\ndyn_real_features: {np.array(dyn_real_features)}. Shape: {np.array(dyn_real_features).shape}')
                return None
            return player_labels, player_features
        else:
            return player_labels, None