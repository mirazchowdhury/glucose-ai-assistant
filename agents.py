from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataFetcherAgent:
    def __init__(self, api_handler, data_handler):
        self.api_handler = api_handler
        self.data_handler = data_handler
        
    def fetch_patient_data(self, patient_id: str, days: int = 7) -> Dict:
        """Fetch and preprocess patient glucose data"""
        try:
            # Get data from API
            raw_data = self.api_handler.fetch_glucose_data(patient_id, days)
            
            # Convert to DataFrame for processing
            readings_df = pd.DataFrame(raw_data.get('readings', []))
            
            if not readings_df.empty:
                readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'])
                readings_df = readings_df.sort_values('timestamp')
                
                # Add derived features
                readings_df['hour'] = readings_df['timestamp'].dt.hour
                readings_df['day_of_week'] = readings_df['timestamp'].dt.dayofweek
                readings_df['glucose_category'] = pd.cut(
                    readings_df['glucose_mg_dl'], 
                    bins=[0, 70, 100, 140, 200, float('inf')], 
                    labels=['Low', 'Normal', 'Elevated', 'High', 'Very High']
                )
            
            return {
                'patient_id': patient_id,
                'data': readings_df,
                'summary': raw_data.get('summary', {}),
                'fetch_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e), 'patient_id': patient_id}

class PatternAnalyzerAgent:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        
    def analyze_trends(self, data_df: pd.DataFrame) -> Dict:
        """Analyze glucose trends and patterns"""
        if data_df.empty:
            return {'error': 'No data to analyze'}
        
        analysis = {}
        
        # Basic statistics
        analysis['basic_stats'] = {
            'mean_glucose': data_df['glucose_mg_dl'].mean(),
            'std_glucose': data_df['glucose_mg_dl'].std(),
            'min_glucose': data_df['glucose_mg_dl'].min(),
            'max_glucose': data_df['glucose_mg_dl'].max(),
            'readings_count': len(data_df)
        }
        
        # Trend analysis
        glucose_values = data_df['glucose_mg_dl'].values
        if len(glucose_values) > 1:
            trend_slope = np.polyfit(range(len(glucose_values)), glucose_values, 1)[0]
            analysis['trend'] = {
                'slope': trend_slope,
                'direction': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable'
            }
        
        # Time-based patterns
        if 'hour' in data_df.columns:
            hourly_avg = data_df.groupby('hour')['glucose_mg_dl'].mean()
            analysis['time_patterns'] = {
                'dawn_phenomenon': hourly_avg[6:9].mean() > hourly_avg.mean() + 10,
                'peak_hour': hourly_avg.idxmax(),
                'lowest_hour': hourly_avg.idxmin()
            }
        
        # Risk indicators
        analysis['risk_indicators'] = {
            'high_readings_pct': (data_df['glucose_mg_dl'] > 200).mean() * 100,
            'low_readings_pct': (data_df['glucose_mg_dl'] < 70).mean() * 100,
            'variable_glucose': data_df['glucose_mg_dl'].std() > 30
        }
        
        # Get LLM analysis
        readings_summary = f"Average: {analysis['basic_stats']['mean_glucose']:.1f}, Range: {analysis['basic_stats']['min_glucose']:.1f}-{analysis['basic_stats']['max_glucose']:.1f}, Trend: {analysis['trend']['direction'] if 'trend' in analysis else 'unknown'}"
        
        analysis['ai_insights'] = self.llm_handler.analyze_patterns(
            readings_summary, 
            f"Last {len(data_df)} readings over recent period"
        )
        
        return analysis

class HealthAdvisorAgent:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        
    def generate_recommendations(self, patient_data: Dict, analysis: Dict) -> Dict:
        """Generate personalized health recommendations"""
        
        # Prepare context for LLM
        glucose_data = f"""
        Recent glucose summary:
        - Average: {analysis.get('basic_stats', {}).get('mean_glucose', 'N/A')}
        - Range: {analysis.get('basic_stats', {}).get('min_glucose', 'N/A')} - {analysis.get('basic_stats', {}).get('max_glucose', 'N/A')}
        - Trend: {analysis.get('trend', {}).get('direction', 'unknown')}
        - High readings: {analysis.get('risk_indicators', {}).get('high_readings_pct', 0):.1f}%
        - Low readings: {analysis.get('risk_indicators', {}).get('low_readings_pct', 0):.1f}%
        """
        
        patient_context = f"Patient ID: {patient_data.get('patient_id', 'unknown')}"
        
        # Get AI recommendations
        recommendations = self.llm_handler.generate_health_advice(
            glucose_data, 
            patient_context
        )
        
        # Structure the response
        return {
            'recommendations': recommendations,
            'priority_level': self._assess_priority(analysis),
            'generated_at': datetime.now().isoformat(),
            'confidence_score': 0.85
        }
    
    def _assess_priority(self, analysis: Dict) -> str:
        """Assess priority level based on analysis"""
        risk_indicators = analysis.get('risk_indicators', {})
        
        high_pct = risk_indicators.get('high_readings_pct', 0)
        low_pct = risk_indicators.get('low_readings_pct', 0)
        
        if high_pct > 20 or low_pct > 10:
            return 'HIGH'
        elif high_pct > 10 or low_pct > 5:
            return 'MEDIUM'
        else:
            return 'LOW'

class AlertSystemAgent:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        
    def check_critical_alerts(self, current_data: Dict) -> List[Dict]:
        """Check for critical glucose alerts"""
        alerts = []
        
        if 'data' not in current_data or current_data['data'].empty:
            return alerts
        
        df = current_data['data']
        latest_reading = df.iloc[-1]
        
        # Critical low glucose
        if latest_reading['glucose_mg_dl'] < 70:
            alerts.append({
                'type': 'CRITICAL_LOW',
                'message': f"Critical low glucose: {latest_reading['glucose_mg_dl']} mg/dL",
                'severity': 'CRITICAL',
                'timestamp': latest_reading['timestamp'],
                'action_required': 'Immediate glucose treatment needed'
            })
        
        # Critical high glucose
        elif latest_reading['glucose_mg_dl'] > 250:
            alerts.append({
                'type': 'CRITICAL_HIGH',
                'message': f"Critical high glucose: {latest_reading['glucose_mg_dl']} mg/dL",
                'severity': 'CRITICAL',
                'timestamp': latest_reading['timestamp'],
                'action_required': 'Contact healthcare provider immediately'
            })
        
        # Rapid changes
        if len(df) >= 2:
            prev_reading = df.iloc[-2]
            change = latest_reading['glucose_mg_dl'] - prev_reading['glucose_mg_dl']
            time_diff = (latest_reading['timestamp'] - prev_reading['timestamp']).total_seconds() / 3600
            
            if abs(change) > 50 and time_diff < 2:  # >50 mg/dL change in <2 hours
                alerts.append({
                    'type': 'RAPID_CHANGE',
                    'message': f"Rapid glucose change: {change:+.1f} mg/dL in {time_diff:.1f} hours",
                    'severity': 'WARNING',
                    'timestamp': latest_reading['timestamp'],
                    'action_required': 'Monitor closely and check for causes'
                })
        
        return alerts