# SportsCast

SportsCast is a service that helps improve the fantasy sports experience for enthusiasts who take part in that shared activity.

Specifically, it is a forecasting service that helps both new and experienced fantasy hockey enthusiasts make better predictions of hockey player performance over the course of a regular season. Enthusiasts may then use this to improve their player selection and trading strategies and so succeed in their leagues. This encourages them to engage in this social activity in spite of possible inexperience or limited time to invest in it.

SportsCast leverages machine learning technology as well as cloud services to provide game-by-game, up-to-date forecasts. The end-users of this service are targeted to be fantasy sport application developers, who may integrate the forecasting feature to improve user-engagement for their apps.

1. <a href="#usecase">Use Case</a>
2. <a href="#usage">How to Use</a>
3. <a href="#retraining">Re-Training</a>
4. <a href="#system">System Diagrams</a>
5. <a href="#testing">Unit Testing</a>

<a id="usecase"></a>
## Use Case

The typical use case for this app is as illustrated on Slide 6 of this [presentation](https://docs.google.com/presentation/d/1TYmXAC4el1T8N4D6sicpDYRG3scOd_jr8fZxCnGjlI4/edit?usp=sharing). Fantasy sport app developers may consume the service as a REST API endpoint.

<a id="usage"></a>
## How to Use

One way the API may be tested is using the following (bare-bones) web application: 

http://54.237.18.0:8501

Note that this application is only available for a limited time to conserve AWS resources.

The expected usage for target users (application developers) is as a REST API endpoint. When the API is available, POST requests may be sent in any terminal to obtain the forecast for a player. This is as shown below:

```bash
curl -i --header "Content-Type: application/json"  --request POST --data '[<player_name>,<num_games>]' https://ya9k6g79n3.execute-api.us-east-1.amazonaws.com/Prod/predict
```

Here, <player_name> is the name of the player for the forecast and <num_games> is the number of games for which the forecast is to be provided. Specific values should be filled in, e.g. <player_name>="Travis Zajac", <num_games>=10.

Like the web application, the API is only available for a limited time to conserve AWS resources.

<a id="retraining"></a>
## Re-Training

The full model is deployed on an AWS EC2 instance, and has been set up to continuously retrain using updating data provided by the NHL API. However, a simpler model is located on AWS S3 to test re-training.

Because new NHL data is not always available, the simple model has been trained using data up until May 2019. It may be retrained using the latest data based on the following steps:

1. Clone the current git repository and recreate the conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

2. Download the serialized model weights from the following AWS S3 link and store them in the directory data/models:

  https://insight-prj-bucket.s3.amazonaws.com//home/ubuntu/SportsCast/data/models/simple_model.p

3. In the root directory of the project (SportsCast/), run the command:

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

<a id="system"></a>
## System Diagrams

The below UML diagrams illustrate the components (classes) and interactions (pipelines) in the system. Two of the pipelines - prediction and retraining - can be tested as was described above.

Firstly, the following class diagram primarily shows that the forecasting models for multiple players implement an interface (Model). There are two multi-player  forecasting models - MultiARIMA and DeepAR. This allows for dynamic dispatch of either type of model selected through the ModelCls string. Note that the MultiARIMA model is an aggregation of multiple models of class ARIMA, where the define ARIMA class wraps the Pyramid ARIMA class.

![Alt text](images/SportsCast_Class_Diagram.png?raw=true "Class Diagram")

Secondly, the sequence diagrams show four pipelines in the system:

1. a training pipeline called using the train() method of TrainingEvaluation
2. an evaluation pipeline called using the evaluate() method of TrainingEvaluation afterwards
3. an inference pipeline called using the pred_points() method of PlayerForecaster. (This is served by the deployed model.)
4. a retraining pipeline called using the retrain_main() method of PlayerForecaster. (This is method is scheduled periodically in the real application by Scheduler.)

![Alt text](images/SportsCast_Sequence_Diagram.png?raw=true "Sequence Diagram")

Importantly, the re-/trained model, predictions and re-/training metrics are stored in a pickled DataFrame for persistence. The saving and loading interactions, as well as others, are not shown here for simplicity.

Note that the above UML diagrams are missing many extraneous details and the names of many methods and variables have been slightly altered for clarity.

<a id="testing"></a>
## Unit Testing
To verify that all tests pass, switch to the branch 'exception_handling_testing' and run the command:
```bash
python3 -m tests.test_DataIngestion
```

You should see the output of the unittest module, which is 'OK' when all tests pass. Note that intermediary errors may be ignored as they are internal to the system and required when testing all edge cases.
