"""
Module for interacting with the NHL's open but undocumented API.
"""
#import streamlit as st
import pandas as pd
from pandas.io.json import json_normalize
import requests as rqsts
import pdb

## data ingestion
def get_seasons(streamlit=False):
    """ returns all seasons on record """
    seasons_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/seasons')
    try:
        seasons_response.raise_for_status()
    except rqsts.exceptions.HTTPError as e:
        if streamlit:
            st.write(e)
        else:
            print(e)
        raise e
    seasons = seasons_response.content
    seasons_df = pd.read_json(seasons)
    seasons_df = json_normalize(seasons_df.seasons)
    seasons_df.set_index('seasonId', inplace=True)
    return seasons_df

def get_current_season():
    season_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/seasons/current')
    season = season_response.content
    season_df = pd.read_json(season)
    season_df = json_normalize(season_df.seasons)
    season_id = season_df.seasonId
    season_start = season_df.regularSeasonStartDate
    season_end = season_df.regularSeasonEndDate
    return season_id, season_start, season_end

def get_teams(streamlit=False):
    """returns all teams FOR THE CURRENT SEASON"""
    teams_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/teams')
    try:
        teams_response.raise_for_status()
    except rqsts.exceptions.HTTPError as e:
        if streamlit:
            st.write(e)
        else:
            print(e)
        raise e
    teams = teams_response.content
    teams_df = pd.read_json(teams)
    teams_df = json_normalize(teams_df.teams)
    return teams_df

def get_schedule(start_date, end_date):
    # teams = get_teams()
    # st.dataframe(teams)
    # output_df = pd.DataFrame()
    schedule_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/schedule?startDate={0}&endDate={1}'.format(start_date, end_date))
    schedule = schedule_response.content
    schedule = pd.read_json(schedule)
    schedule = json_normalize(schedule.dates)
    output_df = pd.DataFrame()
    for game in schedule.games:
        game_df = json_normalize(game)
        output_df = pd.concat([output_df, game_df])
    return output_df

def get_roster(team_id=22, season_id=20182019, streamlit=False):
    """returns roster for given team and season"""
    roster_response = rqsts.get(('https://statsapi.web.nhl.com/api/'
                                 'v1/teams/{0}'
                                 '?expand=team.roster&season={1}') \
                                 .format(team_id, season_id))
    try:
        roster_response.raise_for_status()
    except rqsts.exceptions.HTTPError as e:
        if streamlit:
            st.write(e)
        else:
            print(e)
        raise e
    roster = roster_response.content
    roster_df = pd.read_json(roster)
    roster_df = json_normalize(roster_df.teams)
    roster_list = roster_df['roster.roster'][0]
    roster_df = pd.DataFrame() # generate df to be filled
    for person in roster_list: # populate the df with desired info for each player
        person_info = person['person']
        person_position = person['position']
        player_dict = {'name': person_info['fullName'],
                       'position': person_position['code']}
        player_df = pd.DataFrame(player_dict, index=[person_info['id']])
        # TODO: should change to not extract as int64
        roster_df = roster_df.append(player_df)
    return roster_df

def merge_team_rosters(team_id=22, season_id_list=None, streamlit=False):
    """returns a roster that includes all players
       that played for a team across the seasons provided"""
    if season_id_list is None:
        season_id_list = [20152016, 20162017, 20172018, 20182019]
    merged_roster_df = pd.DataFrame()
    for season in season_id_list:
        try:
            roster_df = get_roster(team_id=team_id, season_id=season)
        except rqsts.exceptions.HTTPError as e:
            if streamlit:
                st.write(e)
            else:
                print(e)
            continue
        merged_roster_df = merged_roster_df.append(roster_df)
    merged_roster_df.drop_duplicates(inplace=True)
    return merged_roster_df

'''
TODO: put exception handling. e.g. return None if stats_list=[]
'''

def get_player_season_game_stats(player_id=8477934, season_id=20182019, streamlit=False):
    """gets the game-by-game stats for a given player and season"""
    stats_response = rqsts.get('https://statsapi.web.nhl.com/api/'
                               'v1/people/{0}/stats?stats=gameLog&season={1}' \
                                .format(player_id, season_id))
    try:
        stats_response.raise_for_status()
    except rqsts.exceptions.HTTPError as e:
        if streamlit:
            st.write(e)
        else:
            print(e)
        raise e
    stats = stats_response.content
    stats_df = pd.read_json(stats)
    stats_df = json_normalize(stats_df.stats)
    stats_df = stats_df.splits
    stats_array = stats_df.array
    stats_list = stats_array[0]
    stats_df = pd.DataFrame() # generate stat df to be filled
    for game in stats_list:
        # import and transpose df from returned game dict (json)
        game_df = pd.DataFrame.from_dict(game).transpose()
        clean_df = pd.DataFrame() # generate game df to be filled
        for stat_type, stat_series in game_df.iterrows():
            if stat_type != 'stat': # the 'stat' stat type contains non-unique but desired values
                try:
                    stat_series.drop_duplicates(inplace=True)
                except SystemError:
                    pass
            stat_series.dropna(inplace=True) # clean out NaN
            # transpose the series so it fits properly into our df
            stat_df = pd.DataFrame(stat_series).transpose()
            # rename columns to prevent collision
            new_columns = []
            if len(stat_df.columns) != 1:
                new_columns = [stat_type + column.capitalize()
                               for column in stat_df.columns]
            if not new_columns:
                new_columns = stat_df.index # use the index if there are no new columns
            stat_df.reset_index(drop=True, inplace=True)
            stat_df.columns = new_columns # rename columns
            clean_df = pd.concat([clean_df, stat_df], axis=1) # add the game to the df
        game_df = clean_df.drop('gameContent', axis=1) # replace the dirty df with our clean one
        # set the indices to be the unique game identifier
        game_df.set_index('gameGamepk', inplace=True)
        stats_df = stats_df.append(game_df)
    return stats_df

def get_combined_player_season_game_stats(player_id=8477934, season_id_list=None):
    """returns player game-by-game stats across multiple seasons"""
    # TODO: add function to set whether each season should be individual cumulative totals or cumulative across all seasons
    if season_id_list is None:
        season_id_list = [20152016, 20162017, 20172018, 20182019]
    full_df = pd.DataFrame()
    for season_id in season_id_list:
        season_df = get_player_season_game_stats(player_id=player_id, season_id=season_id)
        if season_df is None:
            return None
        full_df = pd.concat([full_df, season_df])
    # drop all duplicate entries (resulting from playing for multiple teams)
    full_df.drop_duplicates(subset='date', keep='first', inplace=True)
    return full_df

def augment_combined_player_season_game_stats(player_id=8477934, season_id_list=None):
    #Augments the above with data from other sources. Currently a wrapper to prepend simulated data as a repeat of the past seasons
    
    combined_player_df = get_combined_player_season_game_stats(player_id=player_id)
    if (combined_player_df is None) or (len(combined_player_df)==0):
        return None
    combined_player_df.sort_index(inplace=True)

    if ('season' not in combined_player_df.columns):
        pdb.set_trace()

    #Create simulated data: a copy of seasons 2015-2017 as 2013-2015
    extra_data = combined_player_df[  (combined_player_df['season']=='20152016')|(combined_player_df['season']=='20162017')  ]
    extra_data.loc[extra_data['season']=='20152016','season'] = '20132014'
    extra_data.loc[extra_data['season']=='20162017','season'] = '20142015'
    
    combined_player_df = pd.concat((extra_data,combined_player_df)).reset_index()
    augmented_player_df = combined_player_df
    if ('gameGamepk' in combined_player_df.columns):
        augmented_player_df = augmented_player_df.drop(['gameGamepk'],axis=1)
    return augmented_player_df

def augment_player_dataframe(player_df, cumulative_stat_list=None):
    """generates cumulative totals of stats"""
    #If no player, skip
    if (player_df is None) or (len(player_df)==0):
        return None
    if cumulative_stat_list is None:
        cumulative_stat_list = ['statPoints']
    augmented_df = player_df
    augmented_df.sort_index(inplace=True) # make sure everything is in order
    for stat in cumulative_stat_list:
        try:
            stat_series = augmented_df.loc[:, stat] # grab a stat
        # TODO: verify why there are no points for these players; THINK its because I asked for seasons that didn't exist. still necessary?
        except KeyError:
            stat_series = pd.DataFrame({'cum'+stat.capitalize():
                                        [None for _ in range(len(augmented_df))]})
        stat_series = stat_series.cumsum()
        try:
            stat_series.name = 'cum' + stat_series.name.capitalize() # rename the stat column
        except AttributeError:
            pass
        augmented_df = pd.concat([augmented_df, stat_series], axis=1)
    augmented_df.insert(0, 'gameNumber',
                        [i+1 for i in range(len(augmented_df))]) # add game numbers column
    return augmented_df

def get_player_name_position(player_id=8477934, streamlit=False):
    """returns CURRENT basic player name and position"""
    player_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/people/{0}/'.format(player_id))
    try:
        player_response.raise_for_status()
    except rqsts.exceptions.HTTPError as e:
        if streamlit:
            st.write(e)
        else:
            print(e)
        raise e
    player = player_response.content
    player = pd.read_json(player)
    player = player.people[0]
    player_name = player['fullName']
    player_position = player['primaryPosition']['code']
    return player_name, player_position

def assemble_multiplayer_stat_dataframe(player_id_list=None, season_id_list=None,
                                        stat_list=None, shape='cols', boolAddSimulated=False):
    """returns game-by-game stats for given list of players across given seasons
       (can specify sspecific stats for smaller returns)"""
    if stat_list is None:
        stat_list = ['cumStatpoints']
    if player_id_list is None:
        player_id_list = [8477934, 8476356, 8473468]
    if season_id_list is None:
        season_id_list = [20152016, 20162017, 20172018, 20182019]
    multiplayer_df = pd.DataFrame() # TODO: add a progress bar
    for player_id in player_id_list:
        player_name, player_position = get_player_name_position(player_id)
        if player_position == 'G':
            continue # don't include goalies
        if len(season_id_list) == 1: # handle single or multiple season input lists
            player_df = augment_player_dataframe(
                get_player_season_game_stats(
                    player_id=player_id, season_id=season_id_list[0]))
        elif boolAddSimulated:
            player_df = augment_player_dataframe(
                augment_combined_player_season_game_stats(
                    player_id=player_id, season_id_list=season_id_list))
        else:
            player_df = augment_player_dataframe(
                get_combined_player_season_game_stats(
                    player_id=player_id, season_id_list=season_id_list))
        #IF NO DATA ON PLAYER, skip
        if (player_df is None) or (len(player_df)==0):
            continue
        if stat_list: # if a specific stat is given, only grab that
            player_small_df = player_df \
                                       .loc[:, ['date', 'gameNumber'] \
                                                + stat_list] # keep the useful indices
        else:
            player_small_df = player_df # grab it all otherwise
        player_small_df.reset_index(drop=True, inplace=True) # get rid of messy index
        try:
            player_small_df.insert(0, 'name', [player_name for _ in range(len(player_small_df))]) #Create first column of player names
        except ValueError: # still don't know why this is thrown sometimes
            st.dataframe(player_small_df)
            player_small_df.insert(0, 'errorName',
                                   [player_name for _ in range(len(player_small_df))])
        # player_small_df.set_index('gameNumber', inplace=True)
        # player_small_df.rename(columns={stat: player_name}, inplace=True)
        multiplayer_df = pd.concat([multiplayer_df, player_small_df], axis=0)
        # save temp results
        # multiplayer_df.to_csv('../../data/temp/'
        #                       'assemble_multiplayer_stat_dataframe_TEMP.csv')
    multiplayer_df.reset_index(drop=True, inplace=True)
    if shape == 'rows': # TODO: fix this when required to feed data in a row-per-player fashion
        multiplayer_df = multiplayer_df.transpose()
        # multiplayer_df.set_index(player_id_list, inplace=True)
        try:
            multiplayer_df.insert(0, 'name', multiplayer_df.index)
        except ValueError:
            multiplayer_df.insert(0, 'errorName', multiplayer_df.index)
        multiplayer_df.insert(0, 'playerId', player_id_list)
        multiplayer_df.set_index('playerId', inplace=True)
    return multiplayer_df

def get_all_rosters(season_id_list=None, streamlit=False):
    """returns roster of all players across a given list of seasons"""
    if season_id_list is None:
        season_id_list = [20152016, 20162017, 20172018, 20182019]
    teams_df = get_teams()
    team_ids = teams_df.loc[:, 'id']
    full_roster_df = pd.DataFrame()
    for team_id in team_ids:
        # TODO: save player id's
        team_full_roster = merge_team_rosters(team_id=team_id,
                                              season_id_list=season_id_list,
                                              streamlit=streamlit)
        full_roster_df = pd.concat([full_roster_df, team_full_roster])
    return full_roster_df