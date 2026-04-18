from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load models
anxiety_model = joblib.load('gradient_boosting_anxiety_model.pkl')
suicide_model = joblib.load('gradient_boosting_suicide_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Extract preprocessor components
scaler = preprocessor['scaler']
encoders = preprocessor['encoders']
feature_names = preprocessor['feature_names']  # This should be ['Age_num', 'Hour', 'Sad_enc', 'Irritable_enc', 'Sleep_enc', 'Concentration_enc', 'Appetite_enc', 'Guilt_enc', 'Bonding_enc']
categorical_cols = preprocessor['categorical_cols']


def predict_mental_health(patient_data):
    # Map form field names to model column names
    field_mapping = {
        'Problems concentrating': 'Concentration',
        'Appetite changes': 'Appetite'
    }
    
    # Remap the input data
    mapped_data = {}
    for key, value in patient_data.items():
        mapped_key = field_mapping.get(key, key)
        mapped_data[mapped_key] = value
    
    # ONLY feature columns that were used in training
    # IMPORTANT: Anxious and Suicide are NOT in this list - they are TARGETS!
    feature_cols = ['Age', 'Sad', 'Irritable', 'Sleep', 'Concentration', 
                    'Appetite', 'Guilt', 'Bonding']
    
    # Ensure all required feature columns exist with defaults
    for col in feature_cols:
        if col not in mapped_data:
            mapped_data[col] = 'No'
    
    # Convert to DataFrame (only features!)
    df_pred = pd.DataFrame([mapped_data], columns=feature_cols)
    
    print("="*50)
    print("INPUT FEATURES:")
    for col in feature_cols:
        print(f"  {col}: {df_pred[col].iloc[0]}")
    
    # Extract age as numeric (same as training)
    df_pred['Age_num'] = df_pred['Age'].str.extract(r'(\d+)-(\d+)').apply(
        lambda x: (int(x[0]) + int(x[1]))/2 if pd.notna(x[0]) else np.nan, axis=1
    )
    df_pred['Age_num'].fillna(df_pred['Age_num'].median(), inplace=True)
    
    # Encode categorical features (same as training)
    for col in categorical_cols:
        val = df_pred[col].iloc[0]
        if val in encoders[col].classes_:
            encoded_val = encoders[col].transform([val])[0]
            df_pred[col + '_enc'] = encoded_val
            print(f"  {col}: '{val}' -> encoded as {encoded_val}")
        else:
            df_pred[col + '_enc'] = -1
            print(f"  WARNING: '{val}' not in {col} encoder classes!")
    
    # Set default hour (same as training - all data had hour 12)
    df_pred['Hour'] = 12
    
    # Prepare features in EXACT order as training
    # feature_names should be: ['Age_num', 'Hour', 'Sad_enc', 'Irritable_enc', 'Sleep_enc', 'Concentration_enc', 'Appetite_enc', 'Guilt_enc', 'Bonding_enc']
    X_pred = df_pred[feature_names].copy()
    X_pred[['Age_num', 'Hour']] = scaler.transform(X_pred[['Age_num', 'Hour']])
    
    print("\nFINAL FEATURES (order matches training):")
    for col in feature_names:
        print(f"  {col}: {X_pred[col].iloc[0]:.4f}")
    print("="*50)
    
    # Predict Anxiety (binary classification)
    anxiety_pred = anxiety_model.predict(X_pred)[0]
    anxiety_proba = anxiety_model.predict_proba(X_pred)[0]
    
    # Predict Suicide (3-class classification)
    suicide_pred = suicide_model.predict(X_pred)[0]
    suicide_proba = suicide_model.predict_proba(X_pred)[0]
    
    # Map predictions to labels (based on your training data encoding)
    # From your training: Anxiety_target had 0='No', 1='Yes'
    # Suicide_target had 0='No', 1='Yes', 2='Not interested to say'
    anxiety_labels = {0: 'No', 1: 'Yes'}
    suicide_labels = {0: 'No', 1: 'Yes', 2: 'Not interested to say'}
    
    results = {
        'anxiety_risk': anxiety_labels.get(anxiety_pred, 'Unknown'),
        'anxiety_probability': f"{anxiety_proba[1]:.2%}",  # Probability of "Yes"
        'suicide_risk': suicide_labels.get(suicide_pred, 'Unknown'),
        'suicide_probability': f"{suicide_proba[suicide_pred]:.2%}",
        'anxiety_score': float(anxiety_proba[1]),
        'suicide_score': float(suicide_proba[suicide_pred]),
        'suicide_predicted_class': int(suicide_pred)
    }
    
    # Overall risk calculation
    overall = (anxiety_proba[1] * 0.4 + suicide_proba[suicide_pred] * 0.6)
    
    if overall >= 0.7:
        results['overall_risk'] = 'HIGH RISK'
    elif overall >= 0.4:
        results['overall_risk'] = 'MODERATE RISK'
    else:
        results['overall_risk'] = 'LOW RISK'
    
    results['overall_score'] = float(overall)
    
    return results


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received: {data}")
        prediction = predict_mental_health(data)
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)