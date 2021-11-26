import streamlit as st
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from joblib import load

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title('T20 International Score Predictor')

def main() :
    teams = ['India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies',
             'Afghanistan', 'Pakistan', 'Australia', 'Sri Lanka']

    cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele', 'Barbados',
              'Sydney', 'Melbourne', 'Durban', 'St Lucia',
              'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham',
              'Southampton', 'Mount Maunganui', 'Chittagong',
              'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
              'Christchurch', 'Trinidad']

    test_columns = ['current_score', 'ball_left', 'wicket_left', 'crr', 'last_five', 'batting_team_Afghanistan', 'batting_team_Australia', 'batting_team_Bangladesh', 'batting_team_England', 'batting_team_India', 'batting_team_New Zealand',
                     'batting_team_Pakistan', 'batting_team_South Africa', 'batting_team_Sri Lanka', 'batting_team_West Indies', 'bowling_team_Afghanistan', 'bowling_team_Australia',
                     'bowling_team_Bangladesh', 'bowling_team_England', 'bowling_team_India', 'bowling_team_New Zealand', 'bowling_team_Pakistan', 'bowling_team_South Africa', 'bowling_team_Sri Lanka',
                     'bowling_team_West Indies', 'city_Abu Dhabi', 'city_Adelaide', 'city_Auckland', 'city_Bangalore', 'city_Barbados', 'city_Cape Town',
                     'city_Cardiff', 'city_Centurion', 'city_Chandigarh', 'city_Chittagong', 'city_Christchurch', 'city_Colombo', 'city_Delhi',
                     'city_Dubai', 'city_Durban', 'city_Hamilton', 'city_Johannesburg', 'city_Kolkata', 'city_Lahore', 'city_Lauderhill',
                     'city_London', 'city_Manchester', 'city_Melbourne', 'city_Mirpur', 'city_Mount Maunganui', 'city_Mumbai',
                     'city_Nagpur', 'city_Nottingham', 'city_Pallekele', 'city_Southampton', 'city_St Kitts', 'city_St Lucia', 'city_Sydney',
                     'city_Trinidad', 'city_Wellington'
                ]

    with st.spinner('Loading Model...') :
        model = keras.models.load_model(r't20_Fmodel.h5', compile = False)

    col1, col2 = st.columns(2)
    with col1 :
        batting_team_ = st.selectbox('Select batting team', sorted(teams))
    with col2 :
        bowling_team_ = st.selectbox('Select bowling team', sorted(teams))

    city_ = st.selectbox('Select city', sorted(cities))

    col3, col4, col5 = st.columns(3)
    with col3 :
        current_score_ = st.number_input('Current score')
    with col4 :
        overs_done_ = st.number_input('Overs done (Must>5)')
    with col5 :
        wickets_ = st.number_input('Wickets')

    last_five_ = st.number_input('Runs scored in last 5 overs')

    if st.button('Predict Score') :
        balls_left = 120 - (overs_done_ * 6)
        wickets_left = 10 - wickets_
        crr = current_score_ / overs_done_

        input_df = pd.DataFrame(np.zeros((1, 60)), columns = test_columns)  # test jesa df create karega
        # dataframe me value dalenge
        input_df['current_score'] = current_score_
        input_df['ball_left'] = balls_left
        input_df['wicket_left'] = wickets_left
        input_df['crr'] = crr
        input_df['last_five'] = last_five_
        bt = 'batting_team_' + batting_team_
        input_df[bt] = 1
        bwt = 'bowling_team_' + bowling_team_
        input_df[bwt] = 1
        ct = 'city_' + city_
        input_df[ct] = 1

        scaler = load('scaler_filename.joblib')
        transformed_data = scaler.transform(input_df)
        # input_df = np.zeros((1, 60))
        result = model.predict(np.asarray(transformed_data).astype(np.float32))
        prediction = 'Predicted score is ' + str(round(result[0][0])) + '.'
        st.text(prediction)

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}

a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>

<div class="footer">
<p align="center"> <a href="https://www.linkedin.com/in/ronylpatil/">Developed with ❤ by ronil</a></p>
</div>
        """

st.markdown(footer, unsafe_allow_html = True)

if __name__ == '__main__' :
    main()