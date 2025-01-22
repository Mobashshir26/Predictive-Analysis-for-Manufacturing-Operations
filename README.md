# Predictive Analysis for Manufacturing Operations

This repository contains a FastAPI-based application designed to predict manufacturing downtime using machine learning. The application supports data upload, model training, and real-time predictions based on engineered risk features.

## Features

1. **Upload Data**: Created a synthetic manufacturing data in CSV format.
2. **Train Model**: Train a Random Forest Classifier on the uploaded data, including engineered features for risk assessment.
3. **Make Predictions**: Predict downtime based on input parameters with confidence scores and risk factors.
4. **RESTful API**: Interact with the application through API endpoints.

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Pickle

## Endpoints

### 1. Upload Data
**Endpoint**: `/upload`
- **Method**: `POST`
- **Description**: Upload a CSV file containing manufacturing data.
- **Payload**: File upload (`UploadFile`)
- **Response**:
  ```json
  {
      "message": "Data uploaded successfully",
      "rows": 1000
  }
  ```

### 2. Train Model
**Endpoint**: `/train`
- **Method**: `POST`
- **Description**: Train the machine learning model using the uploaded data.
- **Response**:
  ```json
  {
      "message": "Model trained successfully",
      "metrics": {
          "accuracy": 0.965,
          "f1_score": 0.961,
          "precision": 0.966,
          "recall": 0.905
      },
      "feature_importance": {
          "Temperature": 0.1258,
          "Run_Time": 0.2854,
          "Machine_ID": 0.015,
          "Temp_Risk": 0.055,
          "Runtime_Risk": 0.1741,
          "Combined_Risk": 0.343
      }
  }
  ```

### 3. Predict Downtime
**Endpoint**: `/predict`
- **Method**: `POST`
- **Description**: Predict downtime based on input parameters.
- **Payload**:
  ```json
  {
      "Temperature": 90,
      "Run_Time": 150,
      "Machine_ID": 3
  }
  ```
- **Response**:
  ```json
  {
      "Downtime": "Yes",
      "Confidence": 0.87,
      "Risk_Factors": {
          "Temperature_Risk": "High",
          "Runtime_Risk": "High",
          "Combined_Risk": "High"
      },
      "Probability_Distribution": {
          "No_Downtime": 0.13,
          "Downtime": 0.87
      }
  }
  ```

## Data Requirements

The input CSV file should have the following columns:
- **Temperature**: Float value representing the temperature.
- **Run_Time**: Float value representing the runtime.
- **Machine_ID**: Integer identifier for the machine.
- **Downtime_Flag**: Binary value (0 or 1) indicating downtime.

## Getting Started

### Prerequisites
- Python 3.8+
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Uvicorn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mobashshir26/Predictive-Analysis-for-Manufacturing-Operations.git
   cd Predictive-Analysis-for-Manufacturing-Operations
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:
   ```bash
   uvicorn app:app --reload
   ```

## Project Structure

```
.
├── app.py                # Main application file
├── data/                 # Directory to store uploaded data
├── models/               # Directory to store trained models and scalers
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```





