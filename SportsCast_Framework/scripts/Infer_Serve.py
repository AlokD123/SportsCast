import pandas as pd
import numpy as np
import time
import copy
from pandas.io.json import json_normalize
import pickle
import os

from PlayerForecaster import PlayerForecaster

import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler

@bentoml.artifacts([PickleArtifact('model')])
@bentoml.env(pip_dependencies=["pandas"])
class Forecast_Service(bentoml.BentoService):
    #self.args = {"player_name":0, "num_games":1}

    @bentoml.api(DataframeHandler, typ='series')
    def predict(self, series):
        """
        predict expects pandas.Series as input
        """        
        return self.artifacts.model.pred_points(series[0],series[1])

if __name__=="__main__":
    # 1) create a model
    pf = PlayerForecaster(os.getcwd()+'/./data/outputs/arima_results_m3_fourStep_noFeatures.p')

    # 2) `pack` the service with the model
    bento_service = Forecast_Service()
    bento_service.pack('model', pf)

    # 3) save your BentoSerivce to file archive
    saved_path = bento_service.save()