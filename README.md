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
- `train_model.py`: Script to preprocess data and train the ML model.
- `static/`: Contains CSS styling and Frontend logic.
- `templates/`: Contains the main HTML structure.
- `ipl_score_model.pkl`: The trained machine learning model.
- `model_columns.pkl`: Metadata for model input features.
- `deliveries.csv` & `matches.csv`: Historical IPL data for training.

## 🧠 Methodology
The model is trained using **Linear Regression**. We perform feature engineering on ball-by-ball data to extract:
- Cumulative runs and wickets.
- Current run rate and over progress.
- One-hot encoded categorical features for venues and teams.

The model achieves a **Mean Absolute Error (MAE)** of approximately **14 runs**, which is quite competitive for T20 score prediction.

## 📄 License
This project is for educational purposes. All IPL data is sourced from public repositories.
