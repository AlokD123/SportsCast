"""
Module for working with GluonTS and preprocessing data for features
"""
import itertools
from pathlib import Path
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from sklearn import preprocessing
import pandas as pd
import numpy as np
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import pdb

def clean_duplicates(data, streamlit=False):
    assert len(data)>0, "Missing data"
    assert all([feat in data.columns for feat in ['name','date']]), "Missing feature"
    unique = data.loc[:, 'name'].unique()
    clean = pd.DataFrame()
    for player in unique:
        player_df = data.loc[data.loc[:, 'name'] == player]
        player_df = player_df.drop_duplicates(subset='date')
        clean = pd.concat([clean, player_df])
    return clean

def clean_rookies_retirees(data, split_from='2018-10-03'):
    assert len(data)>0, "Missing data"
    assert all([feat in data.columns for feat in ['name','date']]), "Missing feature"
    unique = data.loc[:, 'name'].unique()
    clean = pd.DataFrame()
    for player in unique:
        player_df = data.loc[data.loc[:, 'name'] == player]
        train_df = player_df.loc[player_df.loc[:, 'date'] < split_from]
        test_df = player_df.loc[player_df.loc[:, 'date'] >= split_from]
        if train_df.shape[0] == 0:
            print(f'{player} is a rookie!') #Rookie since no games before 'split_from'
            continue
        elif test_df.shape[0] == 0:
            print(f'{player} retired!') #Retiree since no games after 'split_from'
            continue
        clean = pd.concat([clean, player_df])
    return clean

#@st.cache
def clean_df(data, split_from='2018-10-03', streamlit=False, boolGenericClean=False):
    assert len(data)>0, "Missing data"
    #TODO: clean duplication
    if boolGenericClean:
        clean = clean_duplicates(data, streamlit=streamlit) #If not splitting, don't consider rookie/retiree
        assert not (len(clean)==0), "All duplicates!?"
    else:
        clean = clean_rookies_retirees(data, split_from=split_from)
        if not (len(clean)==0): #If not rookie or retiree..
            clean = clean_duplicates(clean, streamlit=streamlit)
            assert not (len(clean)==0), "All duplicates!?"
            clean = remove_small_ts(clean, split_from)
    return clean

def sort_dates(data):
    assert len(data)>0, "Missing data"
    assert all([feat in data.columns for feat in ['name','date']]), "Missing feature"
    data_df = pd.DataFrame()
    for player_name in data['name'].unique():
        player_df = data[data['name'] == player_name]
        player_df = player_df.sort_values('date')
        data_df = pd.concat([data_df, player_df])
    return data_df

def remove_small_ts(data, split_from='2018-10-03'):
    assert len(data)>0, "Missing data"
    assert all([feat in data.columns for feat in ['name','date']]), "Missing feature"
    output_df = pd.DataFrame()
    for player_name in data['name'].unique():
        player_df = data[data['name'] == player_name]
        player_test = player_df[player_df['date'] >= split_from]
        player_df = player_df[player_df['date'] < split_from]
        #TODO: replace 82 with NUM_GAMES_SEASON
        if player_df.shape[0] < 82: #If less than one seaon of training data, skip
            continue
        player_df = pd.concat([player_df, player_test])
        output_df = pd.concat([output_df, player_df])
    return output_df

def subset_df(data, column_list=None):
    assert len(data)>0, "Missing data"
    if column_list is None:
        column_list = ['name', 'date', 'cumStatpoints']
    assert all([feat in data.columns for feat in column_list]), "Missing feature"

    data = data.loc[:, column_list]
    return data

def split_train_test(data, split_from='2018-10-03'):
    assert len(data)>0, "Missing data"
    train = data.loc[data.loc[:, 'date'] < split_from]
    test = data.loc[data.loc[:, 'date'] >= split_from]
    return train, test


#TODO: expand tuple after call to prep_df()
def prep_df(data, roster, split_from='2018-10-03', column_list=None, streamlit=False, stand=False, scale=False, boolSplitTrainTest=True):
    '''
    Calls prep_df_generic with each of the split train-test datasets if boolSplitTrainTest, or with just cleaned data if not boolSplitTrainTest
    '''
    try:
        assert len(data)>0, "Missing data"
        assert len(roster)>0, "Missing roster"
        if column_list is None:
            column_list = ['name', 'date', 'gameNumber', 'cumStatpoints']
        assert all([feat in data.columns for feat in column_list]), "Missing feature"

        #data = data.sort_values('date')
        data = sort_dates(data)
        data = clean_df(data, split_from=split_from, streamlit=streamlit,boolGenericClean=(not boolSplitTrainTest))
        if (data is None) or (len(data)==0):
            return None

        if boolSplitTrainTest:
            train, test = split_train_test(data)
            print('Prepping DF Train...')
            retTrain = prep_df_generic(data=train, roster=roster, split_from=split_from, column_list=column_list, streamlit=streamlit, stand=stand, scale=scale)
            print('Prepping DF Test...')
            retTest = prep_df_generic(data=test, roster=roster, split_from=split_from, column_list=column_list, streamlit=streamlit, stand=stand, scale=scale)
            for ret in [retTrain,retTest]:
                if ret is None:
                    return None
            train,targets_trn,targets_meta_trn, targets_raw_trn, targets_raw_meta_trn, stat_cat_features_trn,dyn_cat_features_trn,dyn_real_features_trn,dyn_real_features_meta_trn = retTrain
            test,targets_test,targets_meta_test,targets_raw_test, targets_raw_meta_test, stat_cat_features_test,dyn_cat_features_test,dyn_real_features_test,dyn_real_features_meta_test = retTest
            return train, test, targets_trn, targets_test, targets_raw_trn, targets_raw_test, targets_meta_trn, targets_meta_test, targets_raw_meta_trn, targets_raw_meta_test, stat_cat_features_trn, \
                stat_cat_features_test, dyn_cat_features_trn, dyn_cat_features_test, dyn_real_features_trn, dyn_real_features_test, dyn_real_features_meta_trn, dyn_real_features_meta_test

        else:
            print('Prepping DF Generic...')
            ret = prep_df_generic(data=data, roster=roster, split_from=split_from, column_list=column_list, streamlit=streamlit, stand=stand, scale=scale)
            if ret is None:
                return None
            data, targets, targets_meta, targets_raw, targets_raw_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta = ret #, extra_features[0]
            return data, targets, targets_meta, targets_raw, targets_raw_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta

    except AssertionError as err:
        print(f"Error in preparing data: {err}")
        return None



def prep_df_generic(data, roster, split_from='2018-10-03', column_list=None, streamlit=False, stand=False, scale=False):
    try:
        assert len(data)>2, "Not enough data"
        
        stat_cat_features = assemble_static_cat(data, roster)
        dyn_cat_features = assemble_dynamic_cat(data, roster)
        dyn_real_features, dyn_real_features_meta = days_since_last_game(data,scale=scale)
        #extra_features = assemble_extra_feats(data, roster)
        targets, targets_meta = assemble_target(data, stand=stand, scale=scale)
        targets_raw, targets_raw_meta = assemble_target(data, stand=False, scale=False)
        data = data.loc[:, column_list]
        return data, targets, targets_meta, targets_raw, targets_raw_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta#, extra_features[0]

    except AssertionError as err:
        print(f"Error in preparing data: {err}")
        return None



def encode_roster(feature_df, roster):
    position_map = {'C': 1, 'L': 2, 'R': 3, 'D': 4}
    roster = roster.replace({'position': position_map})
    output_df = pd.DataFrame()
    for player_name in feature_df['name'].unique():
        player_df = feature_df[feature_df['name'] == player_name]
        player_position = roster[roster['name'] == player_name]['position'].iloc[0]
        player_df['position'] = player_position
        output_df = pd.concat([output_df, player_df])
    return output_df


def addPlayerFeatures(data_df=None, player_name=None, roster=None):
    '''
    Input
    ----
    data_df: full game-level data for ALL players
    player_name: name of the player for which augmented features are being construcuted
    roster: ALL players in the league, with associated Position

    Return
    ----
    Data w/ columns for features of other players.
    NOTE: these columns currently INCLUDE a feature for presence of the player 'player_name' (for simplicity)
    #TODO: add/remove others, depending on HOW MANY FEATURES ARE KNOWN AT SELECTION TIME

    '''
    if (data_df is None) or (player_name is None) or (roster is None):
        return None

    player_df = data_df[data_df['name']==player_name]
    playerFeat_df = pd.DataFrame()
    playerFeat_df.columns = roster['name']
    for row, idx in player_df.itertuples():
        team_name = row['teamName'] # mostly unchanging for player between games (rows)
        opponent_name = row['opponentName']
        all_players_date = data_df[data_df['gameLink'] == row['gameLink']][['name','teamName']] #All players for that game
        players_on_team = all_players_date[all_players_date['team']==team_name]['name']
        players_on_opposition = all_players_date[all_players_date['team']==opponent_name]['name']
        for player in roster['name']:
            #playerFeat_df.loc[row,player] = 1 if ((player in players_on_team) or (player in players_on_opposition)) else 0 #Decomposes into players on team OR opposition
            if ((player in players_on_team) or (player in players_on_opposition)):
              playerFeat_df.loc[row,player] = 1  if (player in players_on_team) else -1
            else: 
              playerFeat_df.loc[row,player] = 0

    newFeatures_df = playerFeat_df
    return newFeatures_df

def assemble_extra_feats(data, roster, feature_list=[]):
    assert len(data)>0, "Missing data"
    assert len(roster)>0, "Missing roster"
    if(len(feature_list)==0):
        feature_list = ['isHome','isOT','isWin','statAssists','statBlocked', \
                        'statFaceoffpct','statGoals','statHits','statPenaltyminutes', \
                        'statPlusminus','statShifts','statShotpct','statShots', \
                        'statTimeonice']
    assert all([feat in data.columns for feat in feature_list]), "Missing extra feature"

    features = data.copy()
    output = []
    features = features.loc[:, ['name', 'date'] + feature_list]
    for player_name in features['name'].unique():
        player_df = features[features['name'] == player_name]
        player_df = player_df[feature_list]
        output.append(player_df.values)
        if len(player_df.values.shape)>2:
            raise ValueError(f'Incorrect shape of player_df.values in assemble_extra_feats(). Shape is {player_df.values.shape}')
    output = np.nan_to_num(output[0])
    return output

def assemble_dynamic_cat(data, roster, feature_list=['teamId', 'opponentId'], position=False):
    assert len(data)>0, "Missing data"
    assert len(roster)>0, "Missing roster"
    assert all([feat in data.columns for feat in feature_list]), "Missing feature"

    features = data.copy()
    output = []
    features = features.loc[:, ['name', 'date'] + feature_list]
    for player_name in features['name'].unique():
        player_df = features[features['name'] == player_name]
        player_df = player_df[feature_list]
        output.append(player_df.values)
        if len(player_df.values.shape)>2:
            raise ValueError(f'Incorrect shape of player_df.values in assemble_dynamic_cat(). Shape is {player_df.values.shape}')
    return output

def assemble_dynamic_cat_reshaped(data, roster):
    output = assemble_dynamic_cat(data, roster)
    output = np.array(output[0])
    assert len(output.shape)>1, "Incorrect shape dynamic_cat features"
    assert not (output.shape[0]<output.shape[1]), f"Incorrect orientation dynamic_cat features. Shape: {output.shape}"
    return output

def assemble_static_cat(data, roster):
    assert len(data)>0, "Missing data"
    assert len(roster)>0, "Missing roster"
    try:
        assert all([feat in data.columns for feat in ['name']]), "Missing feature in data"
        assert all([feat in roster.columns for feat in ['name','position']]), "Missing feature in roster"
        output = []
        position_map = {'C': 1, 'L': 2, 'R': 3, 'D': 4}
        roster = roster.replace({'position': position_map})
        for player_name in data['name'].unique():
            position = roster[roster['name'] == player_name]['position'].values.item(0)
            output.append([position])
    except AssertionError as err:
        print(f'Error in assemble_static_cat: {err}. Continuing without feature')
        output =[]
    return output

def assemble_static_cat_reshaped(data, roster):
    output = assemble_static_cat(data, roster)
    output = np.array(output[0])
    return output

def days_since_last_game(data, scale=False):
    assert len(data)>0, "Missing data"
    assert all([feat in data.columns for feat in ['name','date']]), "Missing feature"
    date_df = data[['date', 'name']]
    date_df['date'] = pd.to_datetime(date_df['date'])
    output = []
    output_meta = pd.DataFrame()
    for player_name in date_df['name'].unique():
        player_df = date_df[date_df['name'] == player_name]
        date_diff = player_df['date']
        date_diff = date_diff.diff()
        date_diff.iloc[0] = pd.Timedelta('187 days 00:00:00')
        date_diff = date_diff.dt.days.values.reshape(-1, 1)
        if scale:
            meta_dict = {'name': player_name}
            scaler = preprocessing.MinMaxScaler()
            date_diff = scaler.fit_transform(date_diff)
            meta_dict['min'] = scaler.min_
            meta_dict['scale'] = scaler.scale_
            meta = pd.DataFrame(meta_dict)
            output_meta = pd.concat([output_meta, meta])
            date_diff = date_diff.tolist()
            date_diff = np.array(list(itertools.chain.from_iterable(date_diff))).reshape(1, -1)
        output.append(date_diff)
    return output, output_meta

def assemble_dynamic_real_reshaped(data, scale=False):
    days_since, days_since_meta = days_since_last_game(data, scale=scale)
    output = np.array(days_since[0][0]); output_meta = days_since_meta
    assert output.shape[0]>2, f"Incorrect shape dynamic_real features.\nData={data}\nDays_since={days_since}\nOutput={output}"
    return output, output_meta

def generate_metadata(train_data, test_data, index=None):
    if index is None:
        if 'date' in train_data.columns:
            index = 'date'
        elif 'gameNumber' in train_data.columns:
            index = 'gameNumber'
    prediction_lengths = [test_data.loc[test_data.loc[:, 'name'] == name].shape[0]
                         for name in test_data.loc[:, 'name'].unique()]
    num_unique = len(train_data['name'].unique())
    if index == 'date':
        data_meta = {'num_series': num_unique,
                    'num_steps': [train_data.loc[train_data.loc[:, 'name'] == name] \
                                .shape[0] for name in train_data.loc[:, 'name'].unique()],
                    'prediction_length': prediction_lengths,
                    'freq': '1D',
                    'start': [pd.Timestamp(train_data.loc[train_data.loc[:, 'name'] == name] \
                                            .loc[train_data.loc[train_data.loc[:, 'name'] == name] \
                                            .index[0], 'date'], freq='1D')
                            for name in train_data.loc[:, 'name'].unique()]
                    }
    elif index == 'gameNumber':
        data_meta = {'num_series': num_unique,
                    'num_steps': [train_data.loc[train_data.loc[:, 'name'] == name] \
                                  .shape[0] for name in train_data.loc[:, 'name'].unique()],
                    'prediction_length': prediction_lengths,
                    'freq': '1D',
                    'start': [train_data.loc[train_data.loc[:, 'name'] == name] \
                                        .loc[train_data.loc[train_data.loc[:, 'name'] == name] \
                                        .index[0], 'gameNumber']
                              for name in train_data.loc[:, 'name'].unique()]
                    }
    return data_meta

def generate_minimal_metadata_all(data, index=None):
    if index is None:
        if 'date' in data.columns:
            index = 'date'
        elif 'gameNumber' in data.columns:
            index = 'gameNumber'
    if index == 'date':
        data_meta = {'freq': '1D',
                    'start': [pd.Timestamp(data.loc[data.loc[:, 'name'] == name] \
                                            .loc[data.loc[data.loc[:, 'name'] == name] \
                                            .index[0], 'date'], freq='1D')
                            for name in data.loc[:, 'name'].unique()]
                    }
    elif index == 'gameNumber':
        data_meta = {'freq': '1D',
                    'start': [data.loc[data.loc[:, 'name'] == name] \
                                        .loc[data.loc[data.loc[:, 'name'] == name] \
                                        .index[0], 'gameNumber']
                              for name in data.loc[:, 'name'].unique()]
                    }
    return data_meta


#TODO: move to Preprocessing.py?
#TODO: add 'stand' input option to prep_df()
def assemble_target(data, feature='cumStatpoints', stand=False, scale=False):
    assert len(data)>0, "Missing data"
    assert all([feat in data.columns for feat in ['name',feature]]), "Missing feature"
    targets = []
    targets_meta = pd.DataFrame()
    for player_name in data.loc[:, 'name'].unique():
        meta_dict = {'name':player_name}
        player_df = data.loc[data.loc[:, 'name'] == player_name]
        if not stand and not scale:
            target = player_df.loc[:, feature].values.tolist()
        else:
            target = player_df.loc[:, feature].values.reshape(-1, 1)
            if stand:
                standardizer = preprocessing.StandardScaler()
                target = standardizer.fit_transform(target)
                meta_dict['mean'] = standardizer.mean_
                meta_dict['std'] = standardizer.scale_
            if scale:
                scaler = preprocessing.MinMaxScaler()
                target = scaler.fit_transform(target)
                meta_dict['min'] = scaler.min_
                meta_dict['scale'] = scaler.scale_
            target = target.tolist()
            # print(len(target))
            target = list(itertools.chain.from_iterable(target))
            # print(target)
        targets.append(target)
        if stand or scale:
            meta = pd.DataFrame(meta_dict)
            targets_meta = pd.concat([targets_meta, meta])
    targets_meta = targets_meta.reset_index(drop=True)
    return targets, targets_meta




def run_model(data_train,
              data_meta,
              save_path,
              num_epochs = 50,
              lr=1e-3,
              batch_size=64):
    estimator = DeepAREstimator(freq=data_meta['freq'],
                                prediction_length=data_meta['prediction_length'],
                                trainer=Trainer(batch_size=batch_size,
                                                epochs=num_epochs,
                                                learning_rate=lr,
                                                ctx='cpu',
                                                hybridize=False))
    predictor = estimator.train(data_train)
    predictor.serialize(Path(save_path))
    return predictor


#For train OR test
'''
Inputs
====
targets: vector of targets, OF APPROPRIATE LENGTH (for train or test)

Outputs
====
list_ds: a ListDataset object to hold features + targets

'''
#TODO: determine why STAT_CAT DIFFERENT LENGTH and why not always present
def getListDS(targets,data_meta,stat_cat_features,dyn_cat_features,dyn_real_features, player_names=None):
    if len(stat_cat_features)>0:
        input_list = [{FieldName.TARGET: target, \
                                FieldName.START: start, \
                                FieldName.FEAT_STATIC_CAT: stat_cat, \
                                FieldName.FEAT_DYNAMIC_CAT: dyn_cat, \
                                FieldName.FEAT_DYNAMIC_REAL: dyn_real, \
                                'name': player_name, \
                                'freq': data_meta['freq']} \
                                for target, start, stat_cat, dyn_cat, dyn_real, player_name in zip(targets, \
                                                                                                data_meta['start'], \
                                                                                                stat_cat_features, \
                                                                                                dyn_cat_features, \
                                                                                                dyn_real_features, \
                                                                                                player_names \
                                                                                                )]
    else:
        input_list = [{FieldName.TARGET: target, \
                            FieldName.START: start, \
                            FieldName.FEAT_DYNAMIC_CAT: dyn_cat, \
                            FieldName.FEAT_DYNAMIC_REAL: dyn_real, \
                            'name': player_name, \
                            'freq': data_meta['freq']} \
                            for target, start, dyn_cat, dyn_real, player_name in zip(targets, \
                                                                                            data_meta['start'], \
                                                                                            dyn_cat_features, \
                                                                                            dyn_real_features, \
                                                                                            player_names \
                                                                                            )]
    list_ds = ListDataset(input_list,freq=data_meta['freq'])
    return list_ds