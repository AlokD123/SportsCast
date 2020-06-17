from abc import ABC, abstractmethod
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import numpy as np

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
    def decomposeDS(cls,list_ds:ListDataset,use_exog_feats=False):
        labels = list_ds.list_data[FieldName.TARGET]
        if use_exog_feats:
            dyn_real_features = list_ds.list_data[FieldName.FEAT_DYNAMIC_REAL]
            dyn_cat_features = list_ds.list_data[FieldName.FEAT_DYNAMIC_CAT]
            try:
                features = np.hstack( ( dyn_cat_features, np.transpose(np.array([dyn_real_features])) ) )
            except:
                print('Couldnt concatenate feature arrays.')
                print(f'\ndyn_cat_features: {dyn_cat_features}. Shape: {dyn_cat_features.shape}')
                print(f'\ndyn_real_features: {np.transpose(np.array([dyn_real_features]))}. Shape: {np.transpose(np.array([dyn_real_features])).shape}')
                return None
            return labels, features
        else:
            return labels, None