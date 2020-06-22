import os, time
from timeloop import Timeloop
from datetime import timedelta

tl = Timeloop()

def retrain_attempt():
    print('Checking for new data to retrain model')
    os.system("pkill bentoml")
    os.system("python3 SportsCast_Framework/scripts/PlayerForecaster.py retrain_main --hparams='' --new_ingest_name='full_dataset_added_sim-data-10_chara' --models_fname='arima_results_m3_fourStep_noFeatures' --use_exog_feats=False")
    os.system("python3 /home/ubuntu/InsightPrj_BentoML/Infer_Serve.py") #TODO: update
    os.system("bentoml serve Forecast_Service:latest &")

@tl.job(interval=timedelta(seconds=20))
def sample_job_every_20s():
    retrain_attempt()

if __name__=="__main__":
    os.system("bentoml serve Forecast_Service:latest &")
    tl.start(block=True)