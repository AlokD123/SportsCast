import os, time
from timeloop import Timeloop
from datetime import timedelta
import requests
import logging

tl = Timeloop()


def retrain_attempt():
    '''
    Attempt to retrain model by:
    1) sending request to disable inference server
    2) starting retrain pipeline
    3) sending request to enable inference server
    '''
    logging.info('Checking for new data to retrain model')
    #Step 1
    response = requests.post("http://localhost:8000/updateState", data='[1]') #TODO: replace with deployed model URL
    if response.text=='True':
        #Step 2
        #Retrain
        os.system("python3 SportsCast_Framework/scripts/PlayerForecaster.py retrain_main --hparams='' --new_ingest_name='full_dataset_added_sim-data-10_chara' --models_fname='simple_model' --use_exog_feats=False")
        #Once retrained...
        os.system("python3 /home/ubuntu/InsightPrj_BentoML/Infer_Serve.py") #Generate new BentoService
        os.system("bentoml lambda delete Forecast_Service:latest") #Delete old deployed service
        os.system("bentoml lambda deploy Forecast_Service:latest &") #Deploy new service
    else:
        logging.warn(f'Failed to update server state to "Retraining"')
    #Step 3
    response = requests.post("http://localhost:8000/updateState", data='[0]') #TODO: replace with deployed model URL
    if not response.text=='True':
        logging.warn(f'Failed to update server state to "Ready"')

@tl.job(interval=timedelta(seconds=20))
def sample_job_every_20s():
    retrain_attempt()

if __name__=="__main__":
    os.system("bentoml lambda deploy Forecast_Service:latest &")
    tl.start(block=True)