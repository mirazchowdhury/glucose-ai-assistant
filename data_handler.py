import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Any

class DataHandler:
    def __init__(self):
        self.data_cache = {}
    
    def load_pima_dataset(self):
        """Load and prepare Pima Indians Diabetes Dataset"""
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                  'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
        
        try:
            df = pd.read_csv(url, names=columns)
            return df
        except:
            # Fallback synthetic data
            return self.generate_synthetic_glucose_data(1000)
    
    def generate_synthetic_glucose_data(self, n_samples=500):
        """Generate synthetic glucose data for testing"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Simulate different patient profiles
            patient_type = np.random.choice(['normal', 'prediabetic', 'diabetic'], 
                                          p=[0.6, 0.25, 0.15])
            
            if patient_type == 'normal':
                glucose = np.random.normal(90, 10)
            elif patient_type == 'prediabetic':
                glucose = np.random.normal(115, 15)
            else:  # diabetic
                glucose = np.random.normal(160, 25)
            
            # Ensure realistic ranges
            glucose = max(70, min(400, glucose))
            
            data.append({
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 8760)),
                'glucose': round(glucose, 1),
                'patient_id': f'patient_{i % 50}',  # 50 different patients
                'meal_time': np.random.choice(['fasting', 'post_meal', 'random']),
                'patient_type': patient_type
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Clean and preprocess glucose data"""
        # Remove outliers
        Q1 = df['glucose'].quantile(0.25)
        Q3 = df['glucose'].quantile(0.75)
        IQR = Q3 - Q1
        
        df_clean = df[
            (df['glucose'] >= Q1 - 1.5 * IQR) & 
            (df['glucose'] <= Q3 + 1.5 * IQR)
        ].copy()
        
        # Add derived features
        df_clean['glucose_category'] = pd.cut(
            df_clean['glucose'], 
            bins=[0, 100, 125, 200, float('inf')], 
            labels=['Normal', 'Elevated', 'High', 'Very High']
        )
        
        return df_clean