# SportsCast

SportsCast is a service that helps improve the fantasy sports experience for enthusiasts who take part in that shared activity.

Specifically, it is a forecasting service that helps both new and experienced fantasy hockey enthusiasts make better predictions of hockey player performance over the course of a regular season. Enthusiasts may then use this to improve their player selection and trading strategies and so succeed in their leagues. This encourages them to engage in this social activity in spite of possible inexperience or limited time to invest in it.

SportsCast leverages machine learning technology as well as cloud services to provide game-by-game, up-to-date forecasts. The end-users of this service are targeted to be fantasy sport application developers, who may integrate the forecasting feature to improve user-engagement for their apps.


## Use Case

The typical use case for this app is as described on Slide 5 of this [presentation](https://docs.google.com/presentation/d/1TYmXAC4el1T8N4D6sicpDYRG3scOd_jr8fZxCnGjlI4/edit?usp=sharing). Fantasy sport app developers may consume the service as a REST API endpoint.


## How to Use

When the API has been deployed and is available, GET requests may be sent in any terminal to obtain the forecast for a player. This is as shown below:

```bash
curl -i --header "Content-Type: application/json"  --request POST --data '[<player_name>,<num_games>]' https://ya9k6g79n3.execute-api.us-east-1.amazonaws.com/Prod/predict
```

Here, <player_name> is the name of the player for the forecast and <num_games> is the number of games for which the forecast is to be provided. Specific values should be filled in, e.g. <player_name>="Travis Zajac", <num_games>=10

Although this is the expected usage for software developers, for the less programmatically-inclined, the API may be tested using the following (bare-bones) web application: 


## Re-Training

The full model is deployed on an AWS EC2 instance, and has been set up to continuously retrain using updating data provided by the NHL API. However, a simpler model is located in this Git repository (in the data/models directory) to test re-training.

Because updating data is not always available, the simple model has been trained using data up until May 2019. It may be retrained using the latest data based on the following steps:

1. Clone the current git repository and recreate conda environment using environment.yml

2. In the root directory of the project (SportsCast/), run the command:

```bash
python3 SportsCast_Framework/scripts/PlayerForecaster.py pred_points --player_name=“<player_name>” --num_games=<num_games> --models_dir=$(pwd)/data/models --models_filename=“simple_model”
```

to verify that the model is functional and provides a prediction. <Player_name> and <num_games> should once again be substituted with selected values (e.g. --player_name=“Travis Zajac”, --num_games=10).


3. To retrain, run the command:

```bash
python3 SportsCast_Framework/scripts/PlayerForecaster.py retrain_main --hparams="" --models_fname="simple_model" --use_exog_feats=False
```

This will download the latest data and re-train the model on that data. Note that unlike in the typical use case, where the model will be regularly updated, at least the full 2019-2020 season’s worth of data will be used for retraining and so this process may take some time.


4. Run the prediction command again. You will see that the forecast changes (increases), indicating that the model has been retrained with increased cumulative player points over time.

Note that all commands MUST be run in the root directory of the project.
