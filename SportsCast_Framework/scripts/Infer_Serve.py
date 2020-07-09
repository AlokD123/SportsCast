import pandas as pd
import os
import logging
import fire
import pdb

from PlayerForecaster import PlayerForecaster


import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler



@bentoml.artifacts([PickleArtifact('model')])
@bentoml.env(auto_pip_dependencies=True)
class Forecast_Service(bentoml.BentoService):
    ''' BentoService class. Server prediction request '''

    @bentoml.api(DataframeHandler, typ='series')
    def predict(self, series):
        '''
        Serves prediction request 

        Parameters
        ----
        series: Pandas series, with indices as per this dict (key=type, value=index)
        
        {"player_name":0, 
          "num_games":1}
        '''        
        return self.artifacts.model.pred_points(series[0],series[1])

def create_service(models_dir:str=os.getcwd()+"/data/models", models_fname:str="model_results", bento_dir:str=os.getcwd()+"/serve/bentoml"):
    
    try:
        # 1) create a model
        pf = PlayerForecaster(models_dir=models_dir,models_filename=models_fname)
        assert pf is not None, "Failed to create forecaster"

        # 2) `pack` the service with the model
        bento_service = Forecast_Service()
        bento_service.pack('model', pf)

        # 3) save BentoService to file archive
        saved_path = bento_service.save(bento_dir)

        return True
    except Exception as err:
        logging.error(f'Could not create service:{err}')
        return False

if __name__ == '__main__':
    fire.Fire(create_service)
