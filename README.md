# IPL Analytics & Prediction Dashboard

An advanced, final-year-project-grade IPL match analysis system. This application uses Machine Learning to provide real-time score predictions and win probability analysis with a premium, interactive dashboard.

## 🚀 Features

- **Advanced Score Prediction**: Uses Random Forest Regression with feature engineering (runs/wickets in last 5 overs) for high-accuracy score forecasting.
- **Win Probability Analysis**: A separate classifier that predicts match outcomes in the second innings based on current match state (runs left, balls left, wickets left).
- **Premium Dashboard**: A modern, dark-mode UI built with glassmorphism principles and responsive design.
- **Interactive Visualizations**: Real-time charts powered by `Chart.js` for better data storytelling.

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Models**: Random Forest Regressor & Classifier
- **Frontend**: Vanilla HTML5, CSS3 (Modern Flex/Grid), Javascript (ES6+)
- **Visualization**: Chart.js

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd IPL-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models** (Optional - pre-trained models included in `models/`):
   ```bash
   python train_v2.py   # For score prediction
   python train_win.py  # For win probability
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```
   Access the dashboard at `http://127.0.0.1:8080`

## 📊 Model Performance

| Model | Technique | Accuracy / Metric |
|-------|-----------|-------------------|
| Score Predictor | Random Forest | MAE ~15 runs |
| Win Probability | Random Forest | Accuracy ~82% |

## 📂 Project Structure
- `app.py`: Flask backend serving the API and frontend files.
- `train_v2.py` & `train_win.py`: Scripts to preprocess data and train the ML models.
- `static/`: Contains CSS styling and Frontend logic.
- `templates/`: Contains the main HTML structure.
- `models/ipl_score_model_v2.pkl` & `models/ipl_win_model.pkl`: The trained machine learning models.
- `models/model_columns_v2.pkl` & `models/win_columns.pkl`: Metadata for model input features.
- `csv/deliveries.csv` & `csv/matches.csv`: Historical IPL data for training.

## 🧠 Methodology
The models are trained using **Random Forest Regression and Classification**. We perform feature engineering on ball-by-ball data to extract:
- Cumulative runs and wickets.
- Current run rate and over progress.
- One-hot encoded categorical features for venues and teams.

The score prediction model achieves a **Mean Absolute Error (MAE)** of approximately **15 runs**, which is quite competitive for T20 score prediction, while the win probability model achieves an accuracy of ~82%.

## 📄 License
This project is for educational purposes. The IPL dataset (`deliveries.csv` and `matches.csv`) used in this project is sourced from **Kaggle**.
