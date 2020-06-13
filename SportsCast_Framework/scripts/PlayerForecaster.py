import fire

import os
import pickle

#TODO: maybe don't import here? This module could be opaque to model details
import pmdarima as pm

class PlayerForecaster:
    def __init__(self,models_path=None):
        self.currPlayer = None
        self.currModel = None
        self.models_path = models_path
        self.all_model_results = None

        #TODO: fix logic for path=None or path=absent
        if models_path is not None:
            self.load_all_models(models_path)

    def load_all_models(self,models_path=os.getcwd()+'/../data/outputs/arima_results_m3_fourStep_noFeatures.p'):
        try:
            f = open(self.models_path,"rb")
            self.all_model_results = pickle.load(f)
            print('Loaded models!!')
        except NameError:
            print('Could not load models!!')
            return None

    def getPlayerModel(self,player_name):
        try:
            self.currPlayer = player_name
            return self.all_model_results.loc[player_name,'model']
        except KeyError:
            self.currPlayer = None
            return None
        
    def pred_points(self,player_name:str, num_games: int, print_single_str=False):
        #TODO: add test for self.currModel is None
        if not (self.currPlayer == player_name and (self.currModel is not None)):
            self.currModel = self.getPlayerModel(player_name)
        
        if self.currModel == None:
            return "No such player found"

        prediction, interval = self.currModel.predict(n_periods=num_games, return_conf_int=True)
        assert len(prediction) == num_games, "Issue with model.predict inp/out dims"

        if print_single_str:
            string_resp = f'For the next {num_games} games, the predicted running sum of points for {player_name} are: '
            for i in range(num_games):
                string_resp = string_resp + f'\n    Game {i}: {prediction[i]:.2f} (lower bound: {interval[i,0]:.2f}, upper bound:{interval[i,1]:.2f})'
            return string_resp
        else:
            return [f'Game {i}: {prediction[i]:.2f} (lower bound: {interval[i,0]:.2f}, upper bound:{interval[i,1]:.2f})' for i in range(num_games)]


if __name__ == '__main__':
    fire.Fire(PlayerForecaster)