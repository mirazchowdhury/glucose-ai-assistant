import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import json

def create_sample_data(patient_type="normal"):
    """Create sample glucose data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                         end=datetime.now(), freq='3H')
    
    glucose_values = []
    for i, date in enumerate(dates):
        hour = date.hour
        
        # Base glucose based on patient type
        if patient_type == "diabetic":
            base_glucose = 150
        elif patient_type == "prediabetic":
            base_glucose = 120
        else:
            base_glucose = 95
        
        # Add meal spikes
        if hour in [7, 8]:  # Breakfast
            glucose = base_glucose + np.random.normal(20, 8)
        elif hour in [12, 13]:  # Lunch
            glucose = base_glucose + np.random.normal(25, 10)
        elif hour in [18, 19]:  # Dinner
            glucose = base_glucose + np.random.normal(30, 12)
        else:
            glucose = base_glucose + np.random.normal(0, 10)
        
        # Ensure realistic bounds
        glucose = max(60, min(300, glucose))
        glucose_values.append(round(glucose, 1))
    
    return pd.DataFrame({
        'timestamp': dates,
        'glucose': glucose_values
    })

def analyze_glucose_data(df):
    """Simple glucose data analysis"""
    if df.empty:
        return {}
    
    analysis = {
        'basic_stats': {
            'mean_glucose': df['glucose'].mean(),
            'std_glucose': df['glucose'].std(),
            'min_glucose': df['glucose'].min(),
            'max_glucose': df['glucose'].max(),
            'readings_count': len(df)
        },
        'risk_indicators': {
            'high_readings_pct': (df['glucose'] > 140).mean() * 100,
            'low_readings_pct': (df['glucose'] < 70).mean() * 100,
            'variable_glucose': df['glucose'].std() > 25
        }
    }
    
    # Determine trend
    glucose_values = df['glucose'].values
    if len(glucose_values) > 1:
        trend_slope = np.polyfit(range(len(glucose_values)), glucose_values, 1)[0]
        analysis['trend'] = {
            'direction': 'increasing' if trend_slope > 1 else 'decreasing' if trend_slope < -1 else 'stable',
            'slope': trend_slope
        }
    
    return analysis

def generate_recommendations(analysis):
    """Generate health recommendations based on analysis"""
    if not analysis:
        return "No data available for recommendations."
    
    stats = analysis.get('basic_stats', {})
    risk = analysis.get('risk_indicators', {})
    trend = analysis.get('trend', {})
    
    recommendations = []
    
    # Check average glucose
    avg_glucose = stats.get('mean_glucose', 0)
    if avg_glucose > 140:
        recommendations.append("• Your average glucose is elevated. Consider reviewing your diet and consulting with your healthcare provider.")
    elif avg_glucose < 70:
        recommendations.append("• Your average glucose is low. Monitor for hypoglycemia symptoms and discuss with your doctor.")
    else:
        recommendations.append("• Your average glucose levels are in a good range. Keep up the good work!")
    
    # Check variability
    if risk.get('variable_glucose', False):
        recommendations.append("• Your glucose levels show high variability. Try to maintain consistent meal timing and portions.")
    
    # Check high readings
    high_pct = risk.get('high_readings_pct', 0)
    if high_pct > 25:
        recommendations.append("• You have frequent high glucose readings. Consider post-meal walks and reviewing carbohydrate intake.")
    
    # Check low readings
    low_pct = risk.get('low_readings_pct', 0)
    if low_pct > 10:
        recommendations.append("• You have some low glucose readings. Ensure regular meals and monitor for hypoglycemia symptoms.")
    
    # Trend analysis
    direction = trend.get('direction', 'stable')
    if direction == 'increasing':
        recommendations.append("• Your glucose trend is increasing. Consider lifestyle modifications and consult your healthcare provider.")
    elif direction == 'decreasing':
        recommendations.append("• Your glucose trend is decreasing. Monitor closely to ensure levels don't go too low.")
    
    # General recommendations
    recommendations.extend([
        "",
        "**General Recommendations:**",
        "• Monitor glucose levels regularly as advised by your healthcare provider",
        "• Maintain a balanced diet with controlled carbohydrate portions",
        "• Stay physically active with regular exercise",
        "• Keep consistent meal and sleep schedules",
        "",
        "**⚠️ Important:** This analysis is for educational purposes only. Always consult with your healthcare provider for medical advice."
    ])
    
    return "\n".join(recommendations)

def create_glucose_chart(df):
    """Create glucose trend chart"""
    if df.empty:
        return None
    
    fig = px.line(df, x='timestamp', y='glucose', 
                 title='Glucose Levels Over Time',
                 labels={'glucose': 'Glucose (mg/dL)', 'timestamp': 'Time'})
    
    # Add reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                 annotation_text="Low (70 mg/dL)")
    fig.add_hline(y=100, line_dash="dash", line_color="green", 
                 annotation_text="Normal (100 mg/dL)")
    fig.add_hline(y=140, line_dash="dash", line_color="orange", 
                 annotation_text="Elevated (140 mg/dL)")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Glucose Level (mg/dL)",
        hovermode='x unified'
    )
    
    return fig

def create_distribution_chart(df):
    """Create glucose distribution chart"""
    if df.empty:
        return None
    
    fig = px.histogram(df, x='glucose', nbins=15,
                      title='Glucose Distribution',
                      labels={'glucose': 'Glucose (mg/dL)', 'count': 'Frequency'})
    
    # Add vertical lines for thresholds
    mean_glucose = df['glucose'].mean()
    fig.add_vline(x=mean_glucose, line_dash="solid", line_color="blue", 
                 annotation_text=f"Mean: {mean_glucose:.1f}")
    fig.add_vline(x=70, line_dash="dash", line_color="red", 
                 annotation_text="Low")
    fig.add_vline(x=140, line_dash="dash", line_color="orange", 
                 annotation_text="High")
    
    return fig

def main():
    st.set_page_config(
        page_title="Glucose AI Assistant - Demo",
        page_icon="🩺",
        layout="wide"
    )
    
    st.title("🩺 Glucose AI Assistant - Demo Version")
    st.markdown("*Educational glucose analysis and recommendations*")
    
    # Sidebar controls
    st.sidebar.header("Patient Simulation")
    patient_type = st.sidebar.selectbox(
        "Select Patient Type",
        ["normal", "prediabetic", "diabetic"],
        help="Simulate different glucose patterns"
    )
    
    patient_id = st.sidebar.text_input("Patient ID", value="demo_patient_001")
    
    # Generate data button
    if st.sidebar.button("📊 Generate Glucose Data", type="primary"):
        with st.spinner("Generating glucose data..."):
            # Create sample data
            glucose_data = create_sample_data(patient_type)
            
            # Analyze data
            analysis = analyze_glucose_data(glucose_data)
            
            # Generate recommendations
            recommendations = generate_recommendations(analysis)
            
            # Store in session state
            st.session_state['glucose_data'] = glucose_data
            st.session_state['analysis'] = analysis
            st.session_state['recommendations'] = recommendations
            st.session_state['patient_id'] = patient_id
            st.session_state['patient_type'] = patient_type
            
            st.success("✅ Glucose data generated and analyzed!")
    
    # Display results if available
    if 'glucose_data' in st.session_state:
        df = st.session_state['glucose_data']
        analysis = st.session_state['analysis']
        recommendations = st.session_state['recommendations']
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        stats = analysis.get('basic_stats', {})
        with col1:
            st.metric("Average Glucose", f"{stats.get('mean_glucose', 0):.1f} mg/dL")
        
        with col2:
            st.metric("Min/Max", f"{stats.get('min_glucose', 0):.1f} / {stats.get('max_glucose', 0):.1f}")
        
        with col3:
            st.metric("Total Readings", f"{stats.get('readings_count', 0)}")
        
        with col4:
            trend_direction = analysis.get('trend', {}).get('direction', 'stable').title()
            st.metric("Trend", trend_direction)
        
        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["📈 Charts", "📊 Analysis", "💡 Recommendations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Glucose trend chart
                fig1 = create_glucose_chart(df)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Distribution chart
                fig2 = create_distribution_chart(df)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Daily pattern chart
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                hourly_avg = df.groupby('hour')['glucose'].mean().reset_index()
                
                fig3 = px.bar(hourly_avg, x='hour', y='glucose',
                             title='Average Glucose by Hour of Day',
                             labels={'hour': 'Hour of Day', 'glucose': 'Average Glucose (mg/dL)'})
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Statistical Summary")
                st.write(f"**Mean:** {stats.get('mean_glucose', 0):.1f} mg/dL")
                st.write(f"**Standard Deviation:** {stats.get('std_glucose', 0):.1f}")
                st.write(f"**Range:** {stats.get('min_glucose', 0):.1f} - {stats.get('max_glucose', 0):.1f} mg/dL")
                st.write(f"**Total Readings:** {stats.get('readings_count', 0)}")
            
            with col2:
                st.subheader("⚠️ Risk Indicators")
                risk = analysis.get('risk_indicators', {})
                st.write(f"**High Readings (>140 mg/dL):** {risk.get('high_readings_pct', 0):.1f}%")
                st.write(f"**Low Readings (<70 mg/dL):** {risk.get('low_readings_pct', 0):.1f}%")
                st.write(f"**High Variability:** {'Yes' if risk.get('variable_glucose', False) else 'No'}")
                
                trend = analysis.get('trend', {})
                st.write(f"**Trend Direction:** {trend.get('direction', 'stable').title()}")
            
            # Data table
            st.subheader("📋 Recent Readings")
            st.dataframe(df.tail(10), use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Personalized Recommendations")
            st.markdown(recommendations)
            
            # Priority assessment
            high_pct = analysis.get('risk_indicators', {}).get('high_readings_pct', 0)
            low_pct = analysis.get('risk_indicators', {}).get('low_readings_pct', 0)
            
            if high_pct > 25 or low_pct > 15:
                priority = "🔴 HIGH"
                st.error("High priority - Consider consulting your healthcare provider soon")
            elif high_pct > 10 or low_pct > 5:
                priority = "🟡 MEDIUM"
                st.warning("Medium priority - Monitor closely and consider lifestyle adjustments")
            else:
                priority = "🟢 LOW"
                st.success("Low priority - Continue current management approach")
            
            st.info(f"**Priority Level:** {priority}")
    
    else:
        # Instructions
        st.info("👆 Use the sidebar to generate sample glucose data and see the analysis")
        
        st.markdown("""
        ### 🎯 How to Use This Demo
        
        1. **Select a patient type** from the sidebar (normal, prediabetic, or diabetic)
        2. **Click "Generate Glucose Data"** to create realistic sample data
        3. **Explore the analysis** in the tabs that appear
        4. **Review recommendations** based on the glucose patterns
        
        ### 📊 What You'll See
        - **Real-time glucose charts** showing trends over time
        - **Statistical analysis** of glucose patterns
        - **Risk indicators** for high/low glucose episodes
        - **Personalized recommendations** based on AI analysis
        
        ### ⚠️ Important Note
        This is a **demonstration system** for educational purposes. In a real medical setting:
        - Data would come from actual glucose monitors or lab results
        - Recommendations would be reviewed by healthcare professionals
        - Patient privacy and data security would be paramount
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*🩺 Glucose AI Assistant - Educational Demo | Not for actual medical use*")

if __name__ == "__main__":
    main()