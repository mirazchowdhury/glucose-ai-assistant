#!/usr/bin/env python3
import argparse
import sys
from glucose_ai_system import GlucoseAISystem
import json
from datetime import datetime

class GlucoseAICLI:
    def __init__(self):
        self.ai_system = GlucoseAISystem()
    
    def analyze_patient_cli(self, patient_id: str, days: int = 7, output_format: str = 'text'):
        """Analyze patient via CLI"""
        print(f"Analyzing patient {patient_id} for the last {days} days...\n")
        
        try:
            report = self.ai_system.analyze_patient(patient_id, days)
            
            if 'error' in report:
                print(f"❌ Error: {report['error']}")
                return 1
            
            if output_format == 'json':
                print(json.dumps(report, indent=2, default=str))
            else:
                self.print_formatted_report(report)
            
            return 0
            
        except Exception as e:
            print(f"❌ System error: {str(e)}")
            return 1
    
    def print_formatted_report(self, report):
        """Print human-readable report"""
        print("=" * 60)
        print("🩺 GLUCOSE ANALYSIS REPORT")
        print("=" * 60)
        print(f"Patient ID: {report['patient_id']}")
        print(f"Analysis Time: {report['analysis_timestamp']}")
        print(f"System Confidence: {report['system_confidence']:.1%}")
        print()
        
        # Data Summary
        stats = report['pattern_analysis']['basic_stats']
        print("📊 DATA SUMMARY")
        print("-" * 20)
        print(f"Average Glucose: {stats['mean_glucose']:.1f} mg/dL")
        print(f"Range: {stats['min_glucose']:.1f} - {stats['max_glucose']:.1f} mg/dL")
        print(f"Standard Deviation: {stats['std_glucose']:.1f}")
        print(f"Total Readings: {stats['readings_count']}")
        print()
        
        # Alerts
        print("🚨 ALERTS")
        print("-" * 20)
        if report['alerts']:
            for alert in report['alerts']:
                severity_emoji = "🔴" if alert['severity'] == 'CRITICAL' else "🟡"
                print(f"{severity_emoji} {alert['severity']}: {alert['message']}")
                print(f"   Action: {alert['action_required']}")
        else:
            print("✅ No critical alerts")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 20)
        print(report['recommendations']['recommendations'])
        print(f"\nPriority Level: {report['recommendations']['priority_level']}")
        print()
        
        # Pattern Analysis
        if 'ai_insights' in report['pattern_analysis']:
            print("🔍 PATTERN INSIGHTS")
            print("-" * 20)
            print(report['pattern_analysis']['ai_insights'])
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Glucose AI Analysis System')
    parser.add_argument('patient_id', help='Patient ID to analyze')
    parser.add_argument('--days', '-d', type=int, default=7, 
                       help='Number of days to analyze (default: 7)')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--demo', action='store_true',
                       help='Run with demo data')
    
    args = parser.parse_args()
    
    cli = GlucoseAICLI()
    
    if args.demo:
        print("🎯 Running demo analysis...")
        args.patient_id = 'demo_patient_cli'
    
    exit_code = cli.analyze_patient_cli(args.patient_id, args.days, args.format)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()