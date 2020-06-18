from abc import ABC, abstractmethod
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import numpy as np
import pdb

usePDB = True

class Model(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def create(self):
        pass
    @abstractmethod
    def preprocess(self,data):
        pass
    @abstractmethod
    def fit(self,data,labels):
        pass
    @abstractmethod
    def predict(self,data):
        pass
    @abstractmethod
    def postprocess(self,data):
        pass

    @classmethod
    def decomposeListDS_dict(cls,player_dict:dict,use_exog_feats=False):
        player_labels = player_dict[FieldName.TARGET]
        if use_exog_feats:
            try:
                dyn_real_features = player_dict[FieldName.FEAT_DYNAMIC_REAL]
                dyn_cat_features = player_dict[FieldName.FEAT_DYNAMIC_CAT]
                assert len(dyn_real_features)>0, "Missing dyn_real_features"
                assert len(dyn_real_features)>0, "Missing dyn_cat_features"
            except AssertionError as err:
                print(f'Error in decomposing DS: {err}')
                if usePDB:
                    pdb.set_trace()
                return None
            try:
                player_features = np.hstack( ( np.array(dyn_cat_features), np.array(dyn_real_features) ) ) #Try to concatenate
            except ValueError:
                try:
                    player_features = np.hstack( ( np.array(dyn_cat_features).T, np.array(dyn_real_features).T ) ) #Legacy handling
                except Exception as err:
                    raise Exception(f"{err}")
            except Exception as err:
                print(f'Error in decomposing DS: couldnt concatenate feature arrays: {err}')
                print(f'\ndyn_cat_features: {np.array(dyn_cat_features)}. Shape: {np.array(dyn_cat_features).shape}')
                print(f'\ndyn_real_features: {np.array(dyn_real_features)}. Shape: {np.array(dyn_real_features).shape}')
                if usePDB:
                    pdb.set_trace()
                return None
            return player_labels, player_features
        else:
            return player_labels, None