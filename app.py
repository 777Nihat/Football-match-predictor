import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="centered")

@st.cache_resource
def load_artifacts():
    return joblib.load('football_model_artifacts.joblib')

artifacts = load_artifacts()
model_home = artifacts['model_home']
model_away = artifacts['model_away']
team_stats = artifacts['team_stats']

def predict_match(home_team, away_team, is_neutral):
    h_stat = team_stats[home_team]
    a_stat = team_stats[away_team]
    
    features = pd.DataFrame([{
        'home_elo': h_stat['elo'],
        'away_elo': a_stat['elo'],
        'elo_diff': h_stat['elo'] - a_stat['elo'],
        'home_form_gs': h_stat['form_gs'],
        'home_form_gc': h_stat['form_gc'],
        'away_form_gs': a_stat['form_gs'],
        'away_form_gc': a_stat['form_gc'],
        'neutral': int(is_neutral)
    }])
    
    h_lambda = model_home.predict(features)[0]
    a_lambda = model_away.predict(features)[0]
    
    # Ehtimal matrisi
    max_goals = 8
    t1_probs = [poisson.pmf(i, h_lambda) for i in range(max_goals + 1)]
    t2_probs = [poisson.pmf(i, a_lambda) for i in range(max_goals + 1)]
    matrix = np.outer(t1_probs, t2_probs)
    
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    score = f"{max_idx[0]} - {max_idx[1]}"
    
    h_win = round(np.sum(np.tril(matrix, -1)) * 100, 1)
    draw = round(np.sum(np.diag(matrix)) * 100, 1)
    a_win = round(np.sum(np.triu(matrix, 1)) * 100, 1)
    
    return score, h_win, draw, a_win, h_lambda, a_lambda

# UI
st.title("⚽ International Football Match Predictor")
st.write("LightGBM + Poisson Distribution ilə matç nəticəsi təxmini")

teams = sorted(list(team_stats.keys()))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Ev Sahibi", teams, index=teams.index('Turkey') if 'Turkey' in teams else 0)
with col2:
    away_team = st.selectbox("Səfər Komandası", teams, index=teams.index('Georgia') if 'Georgia' in teams else 1)

is_neutral = st.checkbox("Neytral Meydan (Neutral Venue)")

if st.button("Matçı Simulyasiya Et "):
    if home_team == away_team:
        st.error("Eyni komandanı seçə bilməzsiniz!")
    else:
        score, h_win, draw, a_win, h_l, a_l = predict_match(home_team, away_team, is_neutral)
        
        st.markdown("---")
        st.subheader(f"📊 Təxmini Hesab: **{score}**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_team} Qələbəsi", f"{h_win}%")
        c2.metric("Bərabərlik", f"{draw}%")
        c3.metric(f"{away_team} Qələbəsi", f"{a_win}%")
        
        st.info(f"💡 Gözlənilən Qol Sayıları (xG): {home_team} = {h_l:.2f} | {away_team} = {a_l:.2f}")
