# app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import io
import os

app = FastAPI()

# Global variables to store model and scaler
model = None
scaler = None
feature_columns = ['Temperature', 'Run_Time', 'Machine_ID']

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float
    Machine_ID: int

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df.to_csv('data/manufacturing_data.csv', index=False)
        return {"message": "Data uploaded successfully", "rows": len(df)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/train")
async def train_model():
    global model, scaler
    try:
        os.makedirs('models', exist_ok=True)
        
        # Load data
        df = pd.read_csv('data/manufacturing_data.csv')
        
        # Add engineered features
        df['Temp_Risk'] = (df['Temperature'] > 85).astype(int)
        df['Runtime_Risk'] = (df['Run_Time'] > 140).astype(int)
        df['Combined_Risk'] = df['Temp_Risk'] * df['Runtime_Risk']
        
        # Prepare features and target
        feature_columns_extended = feature_columns + ['Temp_Risk', 'Runtime_Risk', 'Combined_Risk']
        X = df[feature_columns_extended]
        y = df['Downtime_Flag']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with adjusted parameters to reduce overconfidence
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Reduced max_depth
            min_samples_split=10,  # Increased min_samples_split
            min_samples_leaf=4,    # Increased min_samples_leaf
            random_state=42,
            class_weight='balanced',
            bootstrap=True,
            max_features='sqrt'  # This helps reduce overfitting
        )
        model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Save model and scaler
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Get feature importance
        feature_importance = dict(zip(feature_columns_extended, 
                                    model.feature_importances_))
        
        return {
            "message": "Model trained successfully",
            "metrics": {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall)
            },
            "feature_importance": feature_importance
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    global model, scaler
    try:
        if model is None:
            with open('models/model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        
        # Create input data with engineered features
        input_dict = input_data.dict()
        input_dict['Temp_Risk'] = 1 if input_dict['Temperature'] > 85 else 0
        input_dict['Runtime_Risk'] = 1 if input_dict['Run_Time'] > 140 else 0
        input_dict['Combined_Risk'] = input_dict['Temp_Risk'] * input_dict['Runtime_Risk']
        
        # Prepare input data
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        
        # Get predictions and probabilities
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Calculate a more nuanced confidence score
        if prediction == 1:
            confidence = float(probabilities[1])  # Probability of downtime
        else:
            confidence = float(probabilities[0])  # Probability of no downtime
            
        # Add some uncertainty based on risk factors
        uncertainty_factor = 0.05  # 5% uncertainty
        if input_dict['Temp_Risk'] or input_dict['Runtime_Risk']:
            confidence = min(confidence * (1 - uncertainty_factor), 0.95)
        
        return {
            "Downtime": "Yes" if prediction == 1 else "No",
            "Confidence": round(confidence, 3),  # Round to 3 decimal places
            "Risk_Factors": {
                "Temperature_Risk": "High" if input_dict['Temp_Risk'] else "Low",
                "Runtime_Risk": "High" if input_dict['Runtime_Risk'] else "Low",
                "Combined_Risk": "High" if input_dict['Combined_Risk'] else "Low"
            },
            "Probability_Distribution": {
                "No_Downtime": round(float(probabilities[0]), 3),
                "Downtime": round(float(probabilities[1]), 3)
            }
        }
    except Exception as e:
        return {"error": str(e)}