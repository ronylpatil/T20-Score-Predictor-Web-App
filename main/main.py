# -*- coding: utf-8 -*-

"""
Created on Sun Nov 14 23:19:42 2021

@author : Ronil Patil. 

- It is only for T20 Internationals.
- We have 1433 matches data(including females matches), ball by ball. But in this project we will use only matches of males.
- Dataset Source - As usual Kaggle(https://www.kaggle.com/veeralakrishna/cricsheet-a-retrosheet-for-cricket)
- Here data is in form of yaml format.
- 966 T20 matches of males and 466 of females
- At the end got the the Mean Absolute Error of 1.6081.
"""

# Importing required libraries
import numpy as np
import pandas as pd
from yaml import safe_load    # required for converting data from yaml file to pandas dataframe
import os
from tqdm import tqdm

# reading files locations
filename = []
for i in os.listdir(r'E:\DHL Project\Dense Neural Network\T20 Dataset') : 
    filename.append(os.path.join('E:\DHL Project\Dense Neural Network\T20 Dataset', i))

print('Total number of files : ', len(filename))

# converting data from yaml file to pandas dataframe
final_dataframe = pd.DataFrame()
count = 1
for i in tqdm(filename) : 
    with open(i, 'r') as f : 
        # pd.jason_normalize will normalize data into flat table format
        # safe_load will load the file and convert it into user understandable form.
        # and final_dataframe is pandas dataframe so finally we will get our data in dataframe format.
        df = pd.json_normalize(safe_load(f))
        df['match_id'] = count
        final_dataframe = final_dataframe.append(df)
        count += 1

# creating backup file for any emergency
backup = final_dataframe.copy()
# backup.to_csv(r'E:\DHL Project\CNN Projects\Neural Network\backup.csv')

final_dataframe.columns

# droping useless columns 
final_dataframe.drop(columns = [
        'meta.data_version',
        'meta.created',
        'meta.revision',
        'info.outcome.bowl_out',
        'info.bowl_out',
        'info.supersubs.South Africa',
        'info.outcome.eliminator', 
        'info.supersubs.New Zealand',
        'info.neutral_venue', 
        'info.outcome.method', 
        'info.outcome.result',
        'info.outcome.by.runs', 
        'info.match_type_number',
        'info.outcome.by.wickets'
    ], inplace = True)

final_dataframe = final_dataframe[final_dataframe['info.gender'] == 'male']
final_dataframe.drop(columns = 'info.gender', inplace = True)

final_dataframe = final_dataframe[final_dataframe['info.overs'] == 20]
final_dataframe.drop(columns = ['info.match_type', 'info.overs'], inplace = True)

# creating backup file for any emergency
# final_dataframe.to_csv(r'E:\DHL Project\CNN Projects\Neural Network\backup2.csv')

# preprocessing dataset
count = 1
delivery_df = pd.DataFrame()
for index, row in final_dataframe.iterrows():
    if count in [75,108,150,180,268,360,443,458,584,748,982,1052,1111,1226,1345]:
        count+=1
        continue
    count+=1
    ball_of_match = []
    batsman = []
    bowler = []
    runs = []
    player_of_dismissed = []
    teams = []
    batting_team = []
    match_id = []
    city = []
    venue = []
    for ball in row['innings'][0]['1st innings']['deliveries']:
        for key in ball.keys():
            match_id.append(count)
            batting_team.append(row['innings'][0]['1st innings']['team'])
            teams.append(row['info.teams'])
            ball_of_match.append(key)
            batsman.append(ball[key]['batsman'])
            bowler.append(ball[key]['bowler'])
            runs.append(ball[key]['runs']['total'])
            city.append(row['info.city'])
            venue.append(row['info.venue'])
            try :
                player_of_dismissed.append(ball[key]['wicket']['player_out'])
            except :
                player_of_dismissed.append('0')
                
    loop_df = pd.DataFrame({
            'match_id' : match_id,
            'teams' : teams,
            'batting_team' : batting_team,
            'ball' : ball_of_match,
            'batsman' : batsman,
            'bowler' : bowler,
            'runs' : runs,
            'player_dismissed' : player_of_dismissed,
            'city' : city,
            'venue' : venue
        })
    delivery_df = delivery_df.append(loop_df)

# creating backup file for any emergency
# delivery_df.to_csv(r'E:\DHL Project\CNN Projects\Neural Network\backup3.csv')

# extracting bowling team
def bowling_team(row) : 
    for team in row['teams'] :
        if team not in row['batting_team'] : 
            return team

delivery_df['bowling_team'] = delivery_df.apply(bowling_team, axis = 1)
delivery_df.drop(columns = 'teams', inplace = True)

# taking only popular teams into consideration
teams = ['India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies',
         'Afghanistan', 'Pakistan', 'Australia', 'Sri Lanka'] 
delivery_df = delivery_df[delivery_df['batting_team'].isin(teams)]
delivery_df = delivery_df[delivery_df['bowling_team'].isin(teams)]
final = delivery_df[['match_id','batting_team','bowling_team','ball','runs','player_dismissed','city','venue']]

# creating backup file for any emergency
# final.to_csv(r'E:\DHL Project\CNN Projects\Neural Network\backup4.csv')

# final preprocessing part
# Extracting city from venue to fill missing values
cities = np.where(final['city'].isnull(), final['venue'].str.split().apply(lambda x : x[0]), final['city'])
final['city'] = cities
final.drop(columns = 'venue', inplace = True)

# now we will considerd only those cities where more than 4 matches are played(matlab jaha par 5 matches khele gaye he sirf vahi venue lenge hum)
cities = final['city'].value_counts()[final['city'].value_counts() >= 600].index.tolist()
final = final[final['city'].isin(cities)]
# finally we have taken only venues where strictly more than 4 matches were played

# now lets extract runs after each ball, thats why we were taking match_id
final['current_score'] = final.groupby('match_id').cumsum()['runs']     # yaha par humne match id k hisab se run ka cummulative sum nikala jisse ki hume ye pata chale ki har balll k bad kitne run bane

# ab hume ball left and wicket left nikalna he
final['overs'] = final['ball'].apply(lambda x : str(x).split('.')[0])
final['ball_no'] = final['ball'].apply(lambda x : str(x).split('.')[1])
final['ball_bowled'] = (final['overs'].astype('int') * 6 ) + final['ball_no'].astype('int')

'''
yaha par mene extra ball bachi he usko bhi count kiya he matlab extra ball, ball_left me show ho rahi he
'''

# ab hum kitni ball bachi he vo nikalenge  -- (isko change kr k dekhna)
final['ball_left'] = 120 - final['ball_bowled'] 
# final['ball_left'] = 0
final['ball_left'] = final['ball_left'].apply(lambda x : abs(x) if x < 0 else x)

# ab wicket left nikalna he jiiske liye hume player_dismissed ka use karna he
final['player_dismissed'] = final['player_dismissed'].apply(lambda x : 0 if x == '0' else 1)
final['player_dismissed'] = final['player_dismissed'].astype('int')
final['wicket_left'] = 10 - final.groupby('match_id').cumsum()['player_dismissed']

# now ab hume sirf 2 columns or nikalne he, current_run_rate and last_five overs me kitne run bane
final['crr'] = round((final['current_score'] * 6) / final['ball_bowled'], 2)
  
# ab last five overs k run count karne he joki window function se hi hoga
x = final.groupby('match_id')
match_ids = final['match_id'].unique()
last_five = []
for i in match_ids : 
    last_five.extend(x.get_group(i).rolling(window = 30).sum()['runs'].values.tolist())

final['last_five'] = last_five
"""
yaha last 5 hours me kitne run nikle he vo nikala humne, par jese first 5 overs me iski value null rahegi kyuki previous 5 overs ki window me 5 overs ho hi nahi rahe he esliye
"""

# final score nikal lete he ab
# final.drop(columns = 'final_score', inplace = True)
final_df = final.groupby('match_id').sum()['runs'].reset_index().merge(final, on = 'match_id')
final_df = final_df[['batting_team', 'bowling_team', 'city', 'current_score', 'ball_left',
                  'wicket_left', 'crr', 'last_five', 'runs_x']]

final_df.dropna(inplace = True)

# let's shuffle the data to avoide any kind of bias in our dataframe
final_df = final_df.sample(final_df.shape[0])
final_df.reset_index(drop = True, inplace = True)

# Now start doing train test split and create custom neural network architecture using functional API.

# spliting the dataset
X = final_df.drop(columns = 'runs_x')
y = final_df['runs_x']

# create custom neural network architecture using functional API
from sklearn.model_selection import train_test_split
'''
It will work as SEED that we have seen while shuffling the data.
If you don't specify the random_state in the code, then every time you run(execute) your code a new random
value is generated and the train and test datasets would have different values each time.
However, if a fixed value is assigned like random_state = 0 or 1 or 42 or any other integer then no matter 
how many times you execute your code the result would be the same .i.e, same values in train and test datasets.
'''

# spliting the data into training, testing, and validation set
X_, X_test, y_, y_test = train_test_split(X, y, test_size = 0.2, random_state = 65)
X_train, X_valid, y_train, y_valid= train_test_split(X_, y_, test_size = 0.2, random_state = 54)

# now do one hot encoding, and normalize the data.
import pandas as pd

# print('Total no. of column would form : ')
# X_train['batting_team'].unique().shape[0]
# X_train['bowling_team'].unique().shape[0]
# X_train['city'].unique().shape[0]
# total 52 columns will be created + 5 = 57 columns will be there (case of drop first)
# 55 + 5 = 60 columns will be there total (case of drop first -> False)

# one-hot-encoding
X_train = pd.get_dummies(X_train, columns = ['batting_team', 'bowling_team', 'city'])
X_test = pd.get_dummies(X_test, columns = ['batting_team', 'bowling_team', 'city'])
X_valid = pd.get_dummies(X_valid, columns = ['batting_team', 'bowling_team', 'city'])

from sklearn.preprocessing import StandardScaler
from joblib import dump
scaler  = StandardScaler()
scaler.fit(X_train)
# we will save the parameter of scaler
dump(scaler, r'E:\DHL Project\CNN Projects\Neural Network\scaler_filename.joblib')

# use saved weights for transforming another dataset
# from joblib import load
# scaler = load('scaler_filename.joblib')
# transformed_data = scaler.transform(new_data)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_valid_scaled = scaler.transform(X_valid)

# data ka sab kuch ho chuka he, now creating complex neural network using Functional API.
from tensorflow import keras as keras

input_ = keras.layers.Input(shape = X_train_scaled.shape[1:])
hidden1 = keras.layers.Dense(128, activation = 'relu')(input_)
hidden2 = keras.layers.Dense(64, activation = 'relu')(hidden1)
hidden3 = keras.layers.Dense(64, activation = 'relu')(hidden2)
hidden4 = keras.layers.Dense(64, activation = 'relu')(hidden3)
hidden5 = keras.layers.Dense(32, activation = 'relu')(hidden4)
hidden6 = keras.layers.Dense(32, activation = 'relu')(hidden5)
hidden7 = keras.layers.Dense(32, activation = 'relu')(hidden6)
hidden8 = keras.layers.Dense(16, activation = 'relu')(hidden7)
concat = keras.layers.Concatenate()([input_, hidden8])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs = [input_], outputs = [output])
model.compile(loss = 'mae', optimizer = keras.optimizers.Adam(), metrics = ['mae'])
history = model.fit(X_train_scaled, y_train, epochs = 125, validation_data = (X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test) # it will return mae because we are using it in metrics in model.compile
# got MAE : 1.6081

from matplotlib import pyplot as plt
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict(X_test_scaled[:11])
print("Predicted values are : ", predictions)
print("Real values are : ", y_test[:11])

model.save(r'E:\DHL Project\CNN Projects\Neural Network\t20_Fmodel.h5')

from tensorflow.keras.models import load_model
mdl = load_model(r'E:\DHL Project\CNN Projects\Neural Network\t20_model.h5')

prediction = mdl.predict(X_test_scaled[:11])
print("Predicted values are : ", prediction)
print("Real values are : ", y_test[:11])

