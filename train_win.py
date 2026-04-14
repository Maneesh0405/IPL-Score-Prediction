import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

print("Loading data for Win Probability...")
deliveries = pd.read_csv('csv/deliveries.csv')
matches = pd.read_csv('csv/matches.csv')

# Merging data
# matches.csv contains the winner
df = deliveries.merge(matches[['id', 'venue', 'winner']], on='id', how='left')

# Filter for second innings
df = df[df['inning'] == 2]

# Consistency in team names
teams = [
    'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Chennai Super Kings',
    'Kings XI Punjab', 'Rajasthan Royals', 'Delhi Daredevils', 'Mumbai Indians',
    'Sunrisers Hyderabad', 'Deccan Chargers', 'Pune Warriors', 'Gujarat Lions',
    'Rising Pune Supergiant', 'Delhi Capitals', 'Punjab Kings'
]

df = df[df['batting_team'].isin(teams)]
df = df[df['bowling_team'].isin(teams)]

print("Feature Engineering for Win Probability...")
# Target: Did the batting team win?
df['target_win'] = (df['batting_team'] == df['winner']).astype(int)

# Current score in 2nd innings
df['current_score'] = df.groupby('id')['total_runs'].cumsum()

# Handle wickets
df['is_wicket'] = df['is_wicket'].apply(lambda x: 1 if x > 0 else 0)
df['wickets'] = df.groupby('id')['is_wicket'].cumsum()

# Handle overs
df['overs'] = df['over'] + (df['ball'] / 6)

# Get the target run (1st innings score + 1)
# We need to find the total runs in 1st innings for each match
total_runs_1 = deliveries[deliveries['inning'] == 1].groupby('id')['total_runs'].sum().reset_index()
total_runs_1.columns = ['id', 'target_score']
total_runs_1['target_score'] = total_runs_1['target_score'] + 1

df = df.merge(total_runs_1, on='id')

# Calculate runs_left, balls_left, wickets_left
df['runs_left'] = df['target_score'] - df['current_score']
df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])
df['wickets_left'] = 10 - df['wickets']

# Drop rows where balls_left < 0 (can happen in some datasets with mistakes or extra balls)
df = df[df['balls_left'] >= 0]

# Feature selection
features = ['venue', 'batting_team', 'bowling_team', 'runs_left', 'balls_left', 'wickets_left', 'target_score', 'target_win']
df = df[features]

print("One-hot encoding...")
df = pd.get_dummies(data=df, columns=['venue', 'batting_team', 'bowling_team'])

# Splitting data
X = df.drop(columns=['target_win'])
y = df['target_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Win Probability Model (X shape: {X_train.shape})...")
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

# Save Model and Column Info
with open('models/ipl_win_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/win_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Win probability model saved successfully.")
