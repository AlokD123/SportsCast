import api
import pandas as pd

def ingest_league_data(save_loc=None):
    all_rosters = api.get_all_rosters(streamlit=False)
    full_df = api.assemble_multiplayer_stat_dataframe(player_id_list=list(all_rosters.index), stat_list=[], boolAddSimulated=True)
    if save_loc is None:
        full_df.to_csv('./data/inputs/full_dataset.csv')
    else:
        full_df.to_csv(save_loc)

def ingest_new_league_data():
    pass

def load_some_data(save_loc=None):
    season_id, season_start, season_end = api.get_current_season()
    df = pd.DataFrame({"season_id":season_id,"season_start":season_start,"season_end":season_end})
    if save_loc is None:
        df.to_csv('./data/inputs/some_data.csv')
    else:
        df.to_csv(save_loc)

if __name__=="__main__":
    load_some_data('./data/inputs/some_data.csv')