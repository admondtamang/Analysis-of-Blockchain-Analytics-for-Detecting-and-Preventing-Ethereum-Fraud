from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved model and preprocessing assets
try:
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
    X_train_columns = joblib.load('feature_columns.pkl')
    zero_var_cols = joblib.load('zero_variance_cols.pkl')  # Optional, can be an empty list
    print("Model and preprocessing assets loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessing assets: {e}")
    exit()


@app.route('/', methods=['GET'])
def health_check():
    return send_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json(force=True)
        df = pd.DataFrame([data])  # Assuming data is a single record dictionary

        # Drop irrelevant columns (match training logic)
        cols_to_drop = ['Address', 'ERC20 most sent token type', 'ERC20_most_rec_token_type']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        # Fill missing values
        df.fillna(0, inplace=True)

        # Drop zero variance columns
        df.drop(columns=[col for col in zero_var_cols if col in df.columns], inplace=True)

        # Align columns to match training set
        for col in X_train_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing columns as 0

        # Drop extra columns
        df = df[X_train_columns]

        # Scale numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Make prediction
        pred = int(model.predict(df)[0])
        pred_proba = float(model.predict_proba(df)[0][1])

        return jsonify({
            'prediction': pred,
            'prediction_proba_fraud': round(pred_proba, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run with: python flask_server.py
    # For production: use Gunicorn or similar
    app.run(debug=True, host='0.0.0.0', port=5000)
