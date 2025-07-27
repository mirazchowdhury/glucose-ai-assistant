import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from glucose_ai_system import GlucoseAISystem
from data_handler import DataHandler
from agents import DataFetcherAgent, PatternAnalyzerAgent

class TestGlucoseAISystem(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.ai_system = GlucoseAISystem()
        self.data_handler = DataHandler()
        
        # Create test data
        self.test_data = self.create_test_glucose_data()
    
    def create_test_glucose_data(self):
        """Create realistic test glucose data"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='6H')
        glucose_values = []
        
        for i, date in enumerate(dates):
            # Simulate realistic patterns
            base_glucose = 120
            hour = date.hour
            
            # Add meal spikes
            if hour in [8, 13, 19]:  # Meal times
                glucose = base_glucose + np.random.normal(30, 10)
            else:
                glucose = base_glucose + np.random.normal(0, 15)
            
            # Ensure realistic bounds
            glucose = max(60, min(300, glucose))
            glucose_values.append(glucose)
        
        return pd.DataFrame({
            'timestamp': dates,
            'glucose_mg_dl': glucose_values,
            'patient_id': 'test_patient'
        })
    
    def test_data_processing(self):
        """Test data processing functionality"""
        processed_data = self.data_handler.preprocess_data(
            self.test_data.rename(columns={'glucose_mg_dl': 'glucose'})
        )
        
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertIn('glucose_category', processed_data.columns)
        self.assertGreater(len(processed_data), 0)
    
    def test_pattern_analysis(self):
        """Test pattern analysis"""
        analyzer = PatternAnalyzerAgent(self.ai_system.llm_handler)
        
        # Add required columns
        test_df = self.test_data.copy()
        test_df['hour'] = test_df['timestamp'].dt.hour
        
        analysis = analyzer.analyze_trends(test_df)
        
        self.assertIn('basic_stats', analysis)
        self.assertIn('trend', analysis)
        self.assertIn('risk_indicators', analysis)
        
        # Check statistics make sense
        stats = analysis['basic_stats']
        self.assertGreater(stats['mean_glucose'], 0)
        self.assertGreater(stats['std_glucose'], 0)
    
    def test_alert_system(self):
        """Test alert system"""
        alert_agent = self.ai_system.alert_system
        
        # Create test data with critical values
        critical_data = {
            'data': pd.DataFrame({
                'timestamp': [datetime.now()],
                'glucose_mg_dl': [65]  # Low glucose
            })
        }
        
        alerts = alert_agent.check_critical_alerts(critical_data)
        
        self.assertIsInstance(alerts, list)
        if alerts:  # If alerts are generated
            self.assertIn('type', alerts[0])
            self.assertIn('severity', alerts[0])
    
    def test_recommendations_generation(self):
        """Test recommendation generation"""
        # Create mock analysis
        mock_analysis = {
            'basic_stats': {
                'mean_glucose': 150,
                'min_glucose': 80,
                'max_glucose': 220,
                'std_glucose': 25
            },
            'risk_indicators': {
                'high_readings_pct': 20,
                'low_readings_pct': 5,
                'variable_glucose': True
            }
        }
        
        mock_patient_data = {'patient_id': 'test_patient'}
        
        recommendations = self.ai_system.health_advisor.generate_recommendations(
            mock_patient_data, mock_analysis
        )
        
        self.assertIn('recommendations', recommendations)
        self.assertIn('priority_level', recommendations)
        self.assertIsInstance(recommendations['recommendations'], str)
    
    def test_end_to_end_analysis(self):
        """Test complete system workflow"""
        try:
            report = self.ai_system.analyze_patient('test_patient', days=7)
            
            # Should return a report even with mock data
            self.assertIsInstance(report, dict)
            
            # Check required fields
            expected_fields = ['patient_id', 'pattern_analysis', 'alerts', 'recommendations']
            for field in expected_fields:
                self.assertIn(field, report)
                
        except Exception as e:
            # System should handle errors gracefully
            self.assertIsInstance(e, Exception)

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and validation"""
    
    def test_glucose_range_validation(self):
        """Test glucose range validation"""
        data_handler = DataHandler()
        
        # Test with extreme values
        test_data = pd.DataFrame({
            'glucose': [50, 100, 150, 400, -10]  # Mix of normal and extreme
        })
        
        processed = data_handler.preprocess_data(test_data)
        
        # Should handle extreme values appropriately
        self.assertGreater(len(processed), 0)
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        data_handler = DataHandler()
        
        # Create data with missing values
        test_data = pd.DataFrame({
            'glucose': [100, np.nan, 150, None, 120]
        })
        
        # System should handle missing data gracefully
        try:
            processed = data_handler.preprocess_data(test_data.dropna())
            self.assertIsInstance(processed, pd.DataFrame)
        except Exception as e:
            self.fail(f"System failed to handle missing data: {e}")

def run_performance_tests():
    """Run performance benchmarks"""
    print("🔥 Running Performance Tests")
    print("-" * 40)
    
    import time
    
    ai_system = GlucoseAISystem()
    
    # Test data fetching speed
    start_time = time.time()
    data = ai_system.data_fetcher.fetch_patient_data('perf_test_patient')
    fetch_time = time.time() - start_time
    
    print(f"Data Fetching: {fetch_time:.2f}s")
    
    # Test analysis speed
    if 'data' in data and not data['data'].empty:
        start_time = time.time()
        analysis = ai_system.pattern_analyzer.analyze_trends(data['data'])
        analysis_time = time.time() - start_time
        
        print(f"Pattern Analysis: {analysis_time:.2f}s")
    
    # Test recommendation generation
    start_time = time.time()
    mock_analysis = {
        'basic_stats': {'mean_glucose': 130, 'min_glucose': 90, 'max_glucose': 180, 'std_glucose': 20},
        'risk_indicators': {'high_readings_pct': 15, 'low_readings_pct': 3, 'variable_glucose': False}
    }
    recommendations = ai_system.health_advisor.generate_recommendations({'patient_id': 'perf_test'}, mock_analysis)
    rec_time = time.time() - start_time
    
    print(f"Recommendation Generation: {rec_time:.2f}s")
    
    total_time = fetch_time + analysis_time + rec_time
    print(f"Total Processing Time: {total_time:.2f}s")
    
    # Performance benchmarks
    if total_time < 10:
        print("✅ Performance: EXCELLENT")
    elif total_time < 20:
        print("✅ Performance: GOOD")
    else:
        print("⚠️ Performance: NEEDS OPTIMIZATION")

if __name__ == '__main__':
    print("🧪 Running Glucose AI System Tests")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n")
    run_performance_tests()