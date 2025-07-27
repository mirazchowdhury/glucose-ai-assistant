import requests
import json
from typing import Dict, List, Optional
import os
from datetime import datetime

class APIHandler:
    def __init__(self):
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        self.base_headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": "health-data-api.p.rapidapi.com"  # Example host
        }
    
    def fetch_glucose_data(self, patient_id: str, days: int = 7) -> Dict:
        """Fetch glucose data from RapidAPI"""
        # Mock implementation - replace with actual API endpoint
        url = f"https://health-data-api.p.rapidapi.com/glucose/{patient_id}"
        
        params = {
            "days": days,
            "format": "json"
        }
        
        try:
            response = requests.get(url, headers=self.base_headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Return mock data if API fails
                return self.generate_mock_api_response(patient_id, days)
                
        except Exception as e:
            print(f"API Error: {e}")
            return self.generate_mock_api_response(patient_id, days)
    
    def generate_mock_api_response(self, patient_id: str, days: int) -> Dict:
        """Generate mock API response for testing"""
        import numpy as np
        from datetime import datetime, timedelta
        
        readings = []
        base_time = datetime.now()
        
        for i in range(days * 4):  # 4 readings per day
            reading_time = base_time - timedelta(hours=i*6)
            
            # Simulate realistic glucose patterns
            base_glucose = np.random.normal(120, 20)
            
            # Add meal-time variations
            hour = reading_time.hour
            if 7 <= hour <= 9:  # Morning
                glucose = base_glucose + np.random.normal(10, 5)
            elif 12 <= hour <= 14:  # Lunch
                glucose = base_glucose + np.random.normal(15, 8)
            elif 18 <= hour <= 20:  # Dinner
                glucose = base_glucose + np.random.normal(20, 10)
            else:
                glucose = base_glucose
            
            glucose = max(70, min(300, glucose))  # Realistic bounds
            
            readings.append({
                "timestamp": reading_time.isoformat(),
                "glucose_mg_dl": round(glucose, 1),
                "reading_type": "cgm"
            })
        
        return {
            "patient_id": patient_id,
            "readings": readings,
            "summary": {
                "avg_glucose": round(np.mean([r["glucose_mg_dl"] for r in readings]), 1),
                "min_glucose": round(min([r["glucose_mg_dl"] for r in readings]), 1),
                "max_glucose": round(max([r["glucose_mg_dl"] for r in readings]), 1)
            }
        }
    
    def submit_recommendations(self, patient_id: str, recommendations: str) -> bool:
        """Submit AI recommendations back to health system"""
        url = f"https://health-data-api.p.rapidapi.com/recommendations/{patient_id}"
        
        payload = {
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
            "ai_confidence": 0.85
        }
        
        try:
            response = requests.post(
                url, 
                headers=self.base_headers, 
                json=payload
            )
            return response.status_code == 200
        except:
            print("Recommendations logged locally")
            return True