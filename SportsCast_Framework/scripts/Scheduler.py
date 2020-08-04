import os, time
import schedule

def retrain_attempt():
    print('Checking for new data to retrain model')
    os.system("bentoml lambda delete player_forecaster")
    os.system("python3 SportsCast_Framework/scripts/PlayerForecaster.py retrain_main --hparams='' --new_ingest_name='full_dataset_added_sim-data-10_chara' --models_fname='simple_model' --use_exog_feats=False")
    os.system("python3 /home/ubuntu/InsightPrj_BentoML/Infer_Serve.py") #TODO: update
    os.system("bentoml lambda deploy player_forecaster -b Forecast_Service:20200713050642_947E16 --api-name predict --verbose")

if __name__=="__main__":
    os.system("bentoml lambda deploy player_forecaster -b Forecast_Service:20200713050642_947E16 --api-name predict --verbose")
    schedule.every(1).minutes.do(retrain_attempt)
    while True:
        schedule.run_pending()
        time.sleep(1)