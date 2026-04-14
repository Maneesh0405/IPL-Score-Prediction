import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("Loading data...")
deliveries = pd.read_csv('csv/deliveries.csv')
matches = pd.read_csv('csv/matches.csv')

# Merging data
# In these datasets, deliveries.csv usually has the team names.
# matches.csv is used for city/venue if needed.
df = deliveries.copy()

# Ensure venue is merged correctly from matches
if 'venue' in matches.columns:
    df = df.merge(matches[['id', 'venue']], on='id', how='left')

# Consistency in team names
teams = [
    'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Chennai Super Kings',
    'Kings XI Punjab', 'Rajasthan Royals', 'Delhi Daredevils', 'Mumbai Indians',
    'Sunrisers Hyderabad', 'Deccan Chargers', 'Pune Warriors', 'Gujarat Lions',
    'Rising Pune Supergiant', 'Delhi Capitals', 'Punjab Kings'
]

df = df[df['batting_team'].isin(teams)]
df = df[df['bowling_team'].isin(teams)]

# Predicting 1st innings score primarily
df = df[df['inning'] == 1]

print("Feature Engineering...")
# Cumulative runs, wickets, and overs
df['current_score'] = df.groupby('id')['total_runs'].cumsum()

# Handle wickets
df['is_wicket'] = df['is_wicket'].apply(lambda x: 1 if x > 0 else 0)
df['wickets'] = df.groupby('id')['is_wicket'].cumsum()

# Handle overs
# Ball-by-ball to decimal overs
df['overs'] = df['over'] + (df['ball'] / 6)

# Get the total score (target variable)
total_score_df = df.groupby('id')['total_runs'].sum().reset_index()
total_score_df.columns = ['id', 'total']
df = df.merge(total_score_df, on='id')

# Advanced Features: Runs and Wickets in last 5 overs
def get_rolling_stats(group):
    group['prev_score'] = group['current_score'].shift(30)
    group['runs_last_5'] = group['current_score'] - group['prev_score'].fillna(0)
    
    group['prev_wickets'] = group['wickets'].shift(30)
    group['wickets_last_5'] = group['wickets'] - group['prev_wickets'].fillna(0)
    return group

df = df.groupby('id').apply(get_rolling_stats)

# Filtering: keeping data only after 5 overs
df = df[df['overs'] >= 5]

# Select relevant features
features = ['venue', 'batting_team', 'bowling_team', 'current_score', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']
df = df[features]

print("One-hot encoding...")
df = pd.get_dummies(data=df, columns=['venue', 'batting_team', 'bowling_team'])

# Splitting data
X = df.drop(columns=['total'])
y = df['total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training RandomForest Model (X shape: {X_train.shape})...")
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) # Reduced estimators for speed
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Save Model and Column Info
with open('models/ipl_score_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/model_columns_v2.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Advanced model saved successfully.")
