from DataIngestion import *
import os, time
from timeloop import Timeloop
from datetime import timedelta

#import schedule
tl = Timeloop()

def data_received():
    print('Received data')
    os.system("pkill bentoml")
    load_some_data()
    os.system("bentoml serve /home/ubuntu/bentoml/repository/Forecast_Service/20200611163906_07ECBA &")

@tl.job(interval=timedelta(seconds=20))
def sample_job_every_20s():
    data_received()

if __name__=="__main__":
    os.system("bentoml serve /home/ubuntu/bentoml/repository/Forecast_Service/20200611163906_07ECBA &")
    tl.start(block=True)