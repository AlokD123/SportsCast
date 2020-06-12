from DataIngestion import *
import os, time
import schedule

def receive_data():
    print('Collecting new data')
    os.system("pkill bentoml")
    load_some_data()
    os.system("bentoml serve /home/ubuntu/bentoml/repository/Forecast_Service/20200611163906_07ECBA &")

if __name__=="__main__":
    os.system("bentoml serve /home/ubuntu/bentoml/repository/Forecast_Service/20200611163906_07ECBA &")
    schedule.every(1).minutes.do(receive_data)
    while True:
        schedule.run_pending()
        time.sleep(1)