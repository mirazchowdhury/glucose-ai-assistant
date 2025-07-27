from agents import DataFetcherAgent, PatternAnalyzerAgent, HealthAdvisorAgent, AlertSystemAgent
from llm_handler import LLMHandler
from api_handler import APIHandler
from data_handler import DataHandler
from visualization import GlucoseVisualizer
from typing import Dict, Any
import json

class GlucoseAISystem:
    def __init__(self):
        # Initialize handlers
        self.llm_handler = LLMHandler()
        self.api_handler = APIHandler()
        self.data_handler = DataHandler()
        self.visualizer = GlucoseVisualizer()
        
        # Initialize agents
        self.data_fetcher = DataFetcherAgent(self.api_handler, self.data_handler)
        self.pattern_analyzer = PatternAnalyzerAgent(self.llm_handler)
        self.health_advisor = HealthAdvisorAgent(self.llm_handler)
        self.alert_system = AlertSystemAgent(self.llm_handler)
        
    def analyze_patient(self, patient_id: str, days: int = 7) -> Dict[str, Any]:
        """Complete patient analysis workflow"""
        print(f"Starting analysis for patient {patient_id}...")
        
        # Step 1: Fetch data
        print("📊 Fetching patient data...")
        patient_data = self.data_fetcher.fetch_patient_data(patient_id, days)
        
        if 'error' in patient_data:
            return {'error': patient_data['error']}
        
        # Step 2: Analyze patterns
        print("🔍 Analyzing glucose patterns...")
        analysis = self.pattern_analyzer.analyze_trends(patient_data['data'])
        
        # Step 3: Check for alerts
        print("⚠️ Checking for critical alerts...")
        alerts = self.alert_system.check_critical_alerts(patient_data)
        
        # Step 4: Generate recommendations
        print("💡 Generating health recommendations...")
        recommendations = self.health_advisor.generate_recommendations(patient_data, analysis)
        
        # Step 5: Create visualizations
        print("📈 Creating visualizations...")
        charts = self.create_visualizations(patient_data['data'])
        
        # Compile final report
        report = {
            'patient_id': patient_id,
            'analysis_timestamp': patient_data['fetch_time'],
            'data_summary': patient_data['summary'],
            'pattern_analysis': analysis,
            'alerts': alerts,
            'recommendations': recommendations,
            'visualizations': charts,
            'system_confidence': self._calculate_confidence(patient_data, analysis)
        }
        
        print("✅ Analysis complete!")
        return report
    
    def create_visualizations(self, data_df):
        """Create visualization artifacts"""
        if data_df.empty:
            return {'error': 'No data for visualization'}
        
        # Static plots
        static_fig = self.visualizer.plot_glucose_distribution(data_df)
        
        # Interactive dashboard
        interactive_fig = self.visualizer.create_interactive_dashboard(data_df)
        
        return {
            'static_available': True,
            'interactive_available': True,
            'data_points': len(data_df)
        }
    
    def _calculate_confidence(self, patient_data: Dict, analysis: Dict) -> float:
        """Calculate system confidence score"""
        confidence = 0.5  # Base confidence
        
        # More data points = higher confidence
        data_count = len(patient_data.get('data', []))
        if data_count > 20:
            confidence += 0.3
        elif data_count > 10:
            confidence += 0.2
        
        # Recent data = higher confidence
        if patient_data.get('fetch_time'):
            confidence += 0.1
        
        # Successful analysis = higher confidence
        if 'basic_stats' in analysis:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def get_patient_report(self, patient_id: str) -> str:
        """Generate a formatted patient report"""
        report = self.analyze_patient(patient_id)
        
        if 'error' in report:
            return f"Error generating report: {report['error']}"
        
        # Format report
        formatted_report = f"""
# Glucose Analysis Report
**Patient ID:** {report['patient_id']}
**Analysis Date:** {report['analysis_timestamp']}
**System Confidence:** {report['system_confidence']:.2f}

## Data Summary
- Average Glucose: {report['pattern_analysis']['basic_stats']['mean_glucose']:.1f} mg/dL
- Range: {report['pattern_analysis']['basic_stats']['min_glucose']:.1f} - {report['pattern_analysis']['basic_stats']['max_glucose']:.1f} mg/dL
- Total Readings: {report['pattern_analysis']['basic_stats']['readings_count']}

## Critical Alerts
"""
        
        if report['alerts']:
            for alert in report['alerts']:
                formatted_report += f"- **{alert['severity']}**: {alert['message']}\n"
        else:
            formatted_report += "- No critical alerts\n"
        
        formatted_report += f"""
## AI Recommendations
{report['recommendations']['recommendations']}

## Priority Level: {report['recommendations']['priority_level']}

---
*This analysis is for informational purposes only. Always consult with healthcare professionals for medical decisions.*
        """
        
        return formatted_report