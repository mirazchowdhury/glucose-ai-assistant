# Fix for the glucose column name error
# Update your visualization.py file with this corrected version

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class GlucoseVisualizer:
    def __init__(self):
        plt.style.use('default')  # Changed from 'seaborn-v0_8' which might not exist
        
    def _get_glucose_column(self, df):
        """Automatically detect the glucose column name"""
        possible_names = ['glucose', 'glucose_mg_dl', 'blood_glucose', 'bg', 'glucose_level']
        
        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
        
        # If no standard name found, look for columns containing 'glucose'
        glucose_cols = [col for col in df.columns if 'glucose' in col.lower()]
        if glucose_cols:
            return glucose_cols[0]
        
        # Last resort - return the first numeric column
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            return numeric_cols[0]
        
        raise ValueError("No glucose column found in the data")
    
    def plot_glucose_distribution(self, df):
        """Plot glucose level distribution with automatic column detection"""
        if df.empty:
            print("Warning: Empty DataFrame provided")
            return None
            
        # Get the correct glucose column name
        glucose_col = self._get_glucose_column(df)
        print(f"Using glucose column: {glucose_col}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution plot
        sns.histplot(data=df, x=glucose_col, bins=30, ax=axes[0,0])
        axes[0,0].set_title('Glucose Level Distribution')
        axes[0,0].set_xlabel('Glucose (mg/dL)')
        
        # Box plot by category (if available)
        if 'glucose_category' in df.columns:
            sns.boxplot(data=df, x='glucose_category', y=glucose_col, ax=axes[0,1])
            axes[0,1].set_title('Glucose by Category')
            axes[0,1].tick_params(axis='x', rotation=45)
        else:
            # Create categories on the fly
            df_temp = df.copy()
            df_temp['glucose_category'] = pd.cut(
                df_temp[glucose_col], 
                bins=[0, 70, 100, 140, 200, float('inf')], 
                labels=['Low', 'Normal', 'Elevated', 'High', 'Very High']
            )
            sns.boxplot(data=df_temp, x='glucose_category', y=glucose_col, ax=axes[0,1])
            axes[0,1].set_title('Glucose by Category')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Time series if timestamp available
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            axes[1,0].plot(df_sorted['timestamp'], df_sorted[glucose_col], marker='o')
            axes[1,0].set_title('Glucose Over Time')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].set_ylabel('Glucose (mg/dL)')
        else:
            # Plot glucose values in order
            axes[1,0].plot(df[glucose_col].values, marker='o')
            axes[1,0].set_title('Glucose Readings Sequence')
            axes[1,0].set_ylabel('Glucose (mg/dL)')
            axes[1,0].set_xlabel('Reading Number')
        
        # Patient comparison (if patient_id available)
        if 'patient_id' in df.columns:
            patient_means = df.groupby('patient_id')[glucose_col].mean().head(10)
            axes[1,1].bar(range(len(patient_means)), patient_means.values)
            axes[1,1].set_title('Average Glucose by Patient (Top 10)')
            axes[1,1].set_ylabel('Average Glucose (mg/dL)')
            axes[1,1].set_xlabel('Patient Index')
        else:
            # Show glucose statistics
            stats_data = {
                'Mean': df[glucose_col].mean(),
                'Median': df[glucose_col].median(),
                'Min': df[glucose_col].min(),
                'Max': df[glucose_col].max()
            }
            axes[1,1].bar(stats_data.keys(), stats_data.values())
            axes[1,1].set_title('Glucose Statistics')
            axes[1,1].set_ylabel('Glucose (mg/dL)')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, df):
        """Create interactive Plotly dashboard with automatic column detection"""
        if df.empty:
            print("Warning: Empty DataFrame provided")
            return None
            
        # Get the correct glucose column name
        glucose_col = self._get_glucose_column(df)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Glucose Trends', 'Distribution', 'Category Analysis', 'Statistics'),
            specs=[[{"secondary_y": True}, {}], [{}, {}]]
        )
        
        # Time series
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            fig.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'], 
                    y=df_sorted[glucose_col], 
                    mode='lines+markers', 
                    name='Glucose',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Add target range lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Low", row=1, col=1)
            fig.add_hline(y=140, line_dash="dash", line_color="orange", 
                         annotation_text="High", row=1, col=1)
        else:
            # Plot sequence if no timestamp
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(df))), 
                    y=df[glucose_col], 
                    mode='lines+markers', 
                    name='Glucose'
                ),
                row=1, col=1
            )
        
        # Distribution
        fig.add_trace(
            go.Histogram(x=df[glucose_col], name='Distribution', nbinsx=20),
            row=1, col=2
        )
        
        # Category analysis
        if 'glucose_category' in df.columns:
            category_counts = df['glucose_category'].value_counts()
        else:
            # Create categories on the fly
            df_temp = df.copy()
            df_temp['glucose_category'] = pd.cut(
                df_temp[glucose_col], 
                bins=[0, 70, 100, 140, 200, float('inf')], 
                labels=['Low', 'Normal', 'Elevated', 'High', 'Very High']
            )
            category_counts = df_temp['glucose_category'].value_counts()
        
        fig.add_trace(
            go.Bar(x=category_counts.index, y=category_counts.values, 
                  name='Categories'),
            row=2, col=1
        )
        
        # Statistics
        stats = {
            'Mean': df[glucose_col].mean(),
            'Median': df[glucose_col].median(),
            'Std Dev': df[glucose_col].std(),
            'Range': df[glucose_col].max() - df[glucose_col].min()
        }
        
        fig.add_trace(
            go.Bar(x=list(stats.keys()), y=list(stats.values()), 
                  name='Statistics'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800, 
            showlegend=True, 
            title_text="Glucose Analysis Dashboard"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
        fig.update_xaxes(title_text="Glucose (mg/dL)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Category", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Statistic", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        return fig


# Also update your data_handler.py to ensure consistent column naming
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
            # Rename to match API format
            df = df.rename(columns={'glucose': 'glucose_mg_dl'})
            return df
        except:
            # Fallback synthetic data
            return self.generate_synthetic_glucose_data(1000)
    
    def generate_synthetic_glucose_data(self, n_samples=500):
        """Generate synthetic glucose data for testing"""
        import numpy as np
        from datetime import datetime, timedelta
        
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
                'glucose_mg_dl': round(glucose, 1),  # Consistent column name
                'patient_id': f'patient_{i % 50}',  # 50 different patients
                'meal_time': np.random.choice(['fasting', 'post_meal', 'random']),
                'patient_type': patient_type
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Clean and preprocess glucose data"""
        # Handle different column names
        glucose_col = None
        if 'glucose_mg_dl' in df.columns:
            glucose_col = 'glucose_mg_dl'
        elif 'glucose' in df.columns:
            glucose_col = 'glucose'
            # Rename for consistency
            df = df.rename(columns={'glucose': 'glucose_mg_dl'})
            glucose_col = 'glucose_mg_dl'
        else:
            raise ValueError("No glucose column found in data")
        
        # Remove outliers
        Q1 = df[glucose_col].quantile(0.25)
        Q3 = df[glucose_col].quantile(0.75)
        IQR = Q3 - Q1
        
        df_clean = df[
            (df[glucose_col] >= Q1 - 1.5 * IQR) & 
            (df[glucose_col] <= Q3 + 1.5 * IQR)
        ].copy()
        
        # Add derived features
        df_clean['glucose_category'] = pd.cut(
            df_clean[glucose_col], 
            bins=[0, 70, 100, 140, 200, float('inf')], 
            labels=['Low', 'Normal', 'Elevated', 'High', 'Very High']
        )
        
        return df_clean


# Update your app.py Streamlit interface to handle the error better
def create_sample_visualization_fixed(report):
    """Create sample visualization with proper error handling"""
    import plotly.express as px
    import pandas as pd
    
    # Create sample data that matches the expected format
    sample_data = []
    
    # Use actual data from report if available
    if 'pattern_analysis' in report and 'basic_stats' in report['pattern_analysis']:
        stats = report['pattern_analysis']['basic_stats']
        mean_glucose = stats.get('mean_glucose', 120)
        std_glucose = stats.get('std_glucose', 20)
        
        # Generate realistic hourly pattern
        import numpy as np
        for hour in range(24):
            # Simulate daily glucose pattern
            base_glucose = mean_glucose
            
            # Add meal spikes
            if hour in [7, 8]:  # Breakfast
                glucose = base_glucose + np.random.normal(15, 5)
            elif hour in [12, 13]:  # Lunch
                glucose = base_glucose + np.random.normal(20, 8)
            elif hour in [18, 19]:  # Dinner
                glucose = base_glucose + np.random.normal(25, 10)
            elif hour in [2, 3, 4]:  # Dawn phenomenon
                glucose = base_glucose + np.random.normal(10, 5)
            else:
                glucose = base_glucose + np.random.normal(0, std_glucose/2)
            
            # Ensure realistic bounds
            glucose = max(60, min(300, glucose))
            
            sample_data.append({
                'Hour': hour, 
                'glucose_mg_dl': round(glucose, 1)  # Use consistent column name
            })
    else:
        # Fallback data
        for hour in range(24):
            glucose = 100 + (hour % 8) * 5 + (20 if hour > 12 else 0)
            sample_data.append({
                'Hour': hour, 
                'glucose_mg_dl': glucose
            })
    
    df_viz = pd.DataFrame(sample_data)
    
    # Create the plot with correct column name
    fig = px.line(df_viz, x='Hour', y='glucose_mg_dl', 
                 title='24-Hour Glucose Pattern',
                 labels={'glucose_mg_dl': 'Glucose (mg/dL)', 'Hour': 'Hour of Day'})
    
    # Add reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                 annotation_text="Low threshold (70 mg/dL)")
    fig.add_hline(y=140, line_dash="dash", line_color="orange", 
                 annotation_text="High threshold (140 mg/dL)")
    
    return fig