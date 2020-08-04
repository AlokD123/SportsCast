import streamlit as st
from PlayerForecaster import PlayerForecaster
import os

if __name__=="__main__":
    pf=PlayerForecaster(os.getcwd()+'/data/models/simple_model.p')
    player_name = st.selectbox("NHL Roster",list(pf.all_model_results_df.index),0)
    num_games = st.selectbox("Games ahead to predict",list(range(1,4)),0)
    st.write(pf.pred_points(player_name=player_name, num_games=num_games))
