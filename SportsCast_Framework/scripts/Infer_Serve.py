import pandas as pd
import os
import logging

from PlayerForecaster import PlayerForecaster


import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler



@bentoml.artifacts([PickleArtifact('model')])
@bentoml.env(auto_pip_dependencies=True)
class Forecast_Service(bentoml.BentoService):
    #self.args = {"player_name":0, "num_games":1}

    @bentoml.api(DataframeHandler, typ='series')
    def predict(self, series):
        """
        predict expects pandas.Series as input
        """        
        return self.artifacts.model.pred_points(series[0],series[1])

def create_service(models_dir:str=os.getcwd()+"/data/models", models_fname:str="model_results", bento_dir:str=os.getcwd()+"/serve/bentoml"):
    
    try:
        # 1) create a model
        pf = PlayerForecaster(models_dir=models_dir,models_filename=models_fname)

        # 2) `pack` the service with the model
        bento_service = Forecast_Service()
        bento_service.pack('model', pf)

        # 3) save  BentoSerivce to file archive
        saved_path = bento_service.save_to_dir(bento_dir)

        return True
    except Exception as err:
        logging.error(f'Could not create service:{err}')
        return False

if __name__=="__main__":
    create_service()