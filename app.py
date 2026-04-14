from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load models and columns
MODELS_DIR = 'models'

def load_pkl(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

score_model = load_pkl('ipl_score_model_v2.pkl')
score_columns = load_pkl('model_columns_v2.pkl')
win_model = load_pkl('ipl_win_model.pkl')
win_columns = load_pkl('win_columns.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_score', methods=['POST'])
def predict_score():
    try:
        data = request.json
        
        venue = data['venue']
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        current_score = int(data['current_score'])
        wickets = int(data['wickets'])
        overs = float(data['overs'])
        target_score = data.get('target_score')
        
        # Auto-calculate features for score model
        runs_last_5 = int(round((current_score / overs) * 5)) if overs > 0 else 0
        wickets_last_5 = int(round((wickets / overs) * 5)) if overs > 0 else 0
        
        # Prepare input data for score
        score_df = pd.DataFrame(columns=score_columns)
        score_df.loc[0] = 0
        score_df.at[0, 'current_score'] = current_score
        score_df.at[0, 'wickets'] = wickets
        score_df.at[0, 'overs'] = overs
        score_df.at[0, 'runs_last_5'] = runs_last_5
        score_df.at[0, 'wickets_last_5'] = wickets_last_5
        
        # One-hot encoding for score model
        for prefix, val in [('venue_', venue), ('batting_team_', batting_team), ('bowling_team_', bowling_team)]:
            col = f"{prefix}{val}"
            if col in score_columns:
                score_df.at[0, col] = 1
                
        pred_score = int(np.round(score_model.predict(score_df)[0]))
        
        # Calculate win probability if target is provided
        win_prob = None
        if target_score:
            target_score = int(target_score)
            runs_left = target_score - current_score
            balls_left = 120 - int(overs * 6)
            wickets_left = 10 - wickets
            
            win_df = pd.DataFrame(columns=win_columns)
            win_df.loc[0] = 0
            win_df.at[0, 'runs_left'] = runs_left
            win_df.at[0, 'balls_left'] = balls_left
            win_df.at[0, 'wickets_left'] = wickets_left
            win_df.at[0, 'target_score'] = target_score
            
            # One-hot encoding for win model
            for prefix, val in [('venue_', venue), ('batting_team_', batting_team), ('bowling_team_', bowling_team)]:
                col = f"{prefix}{val}"
                if col in win_columns:
                    win_df.at[0, col] = 1
                    
            win_prob = round(win_model.predict_proba(win_df)[0][1] * 100, 2)
        
        return jsonify({
            'prediction': pred_score,
            'win_probability': win_prob,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/get_metadata', methods=['GET'])
def get_metadata():
    # Use score_columns as base for team and venue list
    venues = sorted(list(set([col.replace('venue_', '') for col in score_columns if col.startswith('venue_')])))
    teams = sorted(list(set([col.replace('batting_team_', '') for col in score_columns if col.startswith('batting_team_')])))
    
    return jsonify({
        'venues': venues,
        'teams': teams
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
