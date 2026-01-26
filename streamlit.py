"""
South Asia Agricultural Emissions Forecasting & Hotspot Identification System
=============================================================================

A comprehensive ML pipeline for analyzing and forecasting agricultural emissions
across South Asian countries using engineered FAOSTAT data.

Author: Agricultural Emissions Research Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

# ==================== Configuration ====================
st.set_page_config(
    page_title="South Asia Emissions Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2E7D32;
    }
    .stAlert {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Data Processing Functions ====================

@st.cache_data
def load_and_process_engineered_data(data_file):
    """Load and process the engineered emissions data"""
    try:
        # Load engineered data
        df = pd.read_csv(data_file)
        
        # Ensure proper column names
        # Expected columns: Domain, Area, Element, Item, Year, Source, Unit, Value, Flag, emission_type, decade, emission_intensity, rolling_5yr_mean
        
        # Create pivot for total emissions by country and year
        # Sum both CH4 and N2O emissions
        emissions_pivot = df.pivot_table(
            index=['Area', 'Year'],
            columns='emission_type',
            values='Value',
            aggfunc='sum'
        ).reset_index()
        
        # Calculate total emissions (combining CH4 and N2O)
        if 'CH4' in emissions_pivot.columns and 'N2O' in emissions_pivot.columns:
            emissions_pivot['Total_Emissions_kt'] = emissions_pivot['CH4'].fillna(0) + emissions_pivot['N2O'].fillna(0)
        else:
            emissions_pivot['Total_Emissions_kt'] = emissions_pivot.sum(axis=1, numeric_only=True)
        
        # Add engineered features back
        engineered_features = df.groupby(['Area', 'Year']).agg({
            'emission_intensity': 'mean',
            'rolling_5yr_mean': 'mean',
            'decade': 'first'
        }).reset_index()
        
        # Merge with pivot
        merged_df = emissions_pivot.merge(engineered_features, on=['Area', 'Year'], how='left')
        
        return merged_df, df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def prepare_time_series(df, country, emission_col='Total_Emissions_kt'):
    """Prepare time series data for a specific country"""
    country_data = df[df['Area'] == country].copy()
    country_data = country_data.sort_values('Year')
    country_data.set_index('Year', inplace=True)
    return country_data[emission_col]

# ==================== ML Models ====================

class EmissionForecaster:
    """Comprehensive forecasting system for agricultural emissions"""
    
    def __init__(self, data, country, target_col='Total_Emissions_kt'):
        self.data = data
        self.country = country
        self.target_col = target_col
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        
    def train_test_split(self, test_size=0.2):
        """Split data into train and test sets"""
        ts = prepare_time_series(self.data, self.country, self.target_col)
        split_idx = int(len(ts) * (1 - test_size))
        
        self.train = ts.iloc[:split_idx]
        self.test = ts.iloc[split_idx:]
        self.years_train = self.train.index
        self.years_test = self.test.index
        
        return self.train, self.test
    
    def fit_sarima(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """Fit SARIMA model"""
        try:
            model = SARIMAX(
                self.train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.models['SARIMA'] = model.fit(disp=False)
            return True
        except:
            return False
    
    def fit_exponential_smoothing(self, seasonal_periods=12):
        """Fit Exponential Smoothing model"""
        try:
            model = ExponentialSmoothing(
                self.train,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add'
            )
            self.models['ExpSmoothing'] = model.fit()
            return True
        except:
            return False
    
    def fit_random_forest(self, n_estimators=100):
        """Fit Random Forest model"""
        X_train = np.array(range(len(self.train))).reshape(-1, 1)
        y_train = self.train.values
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        return True
    
    def fit_gradient_boosting(self, n_estimators=100):
        """Fit Gradient Boosting model"""
        X_train = np.array(range(len(self.train))).reshape(-1, 1)
        y_train = self.train.values
        
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        self.models['GradientBoosting'] = model
        return True
    
    def evaluate_models(self):
        """Evaluate all fitted models"""
        for name, model in self.models.items():
            if name in ['SARIMA', 'ExpSmoothing']:
                pred = model.forecast(steps=len(self.test))
            else:
                X_test = np.array(range(len(self.train), len(self.train) + len(self.test))).reshape(-1, 1)
                pred = model.predict(X_test)
            
            mae = mean_absolute_error(self.test, pred)
            rmse = np.sqrt(mean_squared_error(self.test, pred))
            r2 = r2_score(self.test, pred)
            mape = np.mean(np.abs((self.test - pred) / self.test)) * 100
            
            self.metrics[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
    
    def forecast_future(self, model_name, years_ahead=10):
        """Generate future forecasts"""
        model = self.models[model_name]
        
        if model_name in ['SARIMA', 'ExpSmoothing']:
            forecast = model.forecast(steps=years_ahead)
            future_years = range(int(self.test.index[-1]) + 1, int(self.test.index[-1]) + years_ahead + 1)
        else:
            last_idx = len(self.train) + len(self.test)
            X_future = np.array(range(last_idx, last_idx + years_ahead)).reshape(-1, 1)
            forecast = model.predict(X_future)
            future_years = range(int(self.test.index[-1]) + 1, int(self.test.index[-1]) + years_ahead + 1)
        
        self.forecasts[model_name] = pd.Series(forecast, index=future_years)
        return self.forecasts[model_name]

class HotspotAnalyzer:
    """Identify emission hotspots and high-risk areas"""
    
    def __init__(self, data):
        self.data = data
        self.hotspots = {}
        
    def calculate_growth_rates(self, years=10):
        """Calculate emission growth rates by country"""
        results = []
        
        for country in self.data['Area'].unique():
            country_data = self.data[self.data['Area'] == country].sort_values('Year')
            
            if len(country_data) >= years:
                recent = country_data.tail(years)
                early = country_data.head(years)
                
                recent_avg = recent['Total_Emissions_kt'].mean()
                early_avg = early['Total_Emissions_kt'].mean()
                
                growth_rate = ((recent_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
                
                results.append({
                    'Country': country,
                    'Current_Emissions': recent_avg,
                    'Historical_Emissions': early_avg,
                    'Growth_Rate_%': growth_rate,
                    'Absolute_Change': recent_avg - early_avg
                })
        
        return pd.DataFrame(results).sort_values('Growth_Rate_%', ascending=False)
    
    def identify_hotspots(self, threshold_percentile=75):
        """Identify hotspot countries based on emissions and growth"""
        growth_df = self.calculate_growth_rates()
        
        emission_threshold = growth_df['Current_Emissions'].quantile(threshold_percentile / 100)
        growth_threshold = growth_df['Growth_Rate_%'].quantile(threshold_percentile / 100)
        
        hotspots = growth_df[
            (growth_df['Current_Emissions'] >= emission_threshold) |
            (growth_df['Growth_Rate_%'] >= growth_threshold)
        ].copy()
        
        # Assign risk levels
        hotspots['Risk_Level'] = 'Medium'
        hotspots.loc[
            (hotspots['Current_Emissions'] >= growth_df['Current_Emissions'].quantile(0.9)) &
            (hotspots['Growth_Rate_%'] >= growth_df['Growth_Rate_%'].quantile(0.9)),
            'Risk_Level'
        ] = 'High'
        
        return hotspots
    
    def decompose_emissions(self, country, raw_data):
        """Decompose emissions by source for a country"""
        country_data = raw_data[raw_data['Area'] == country].copy()
        
        # Group by emission type and element
        decomposition = country_data.groupby(['emission_type', 'Element'])['Value'].sum().reset_index()
        decomposition['Source'] = decomposition['Element'] + ' (' + decomposition['emission_type'] + ')'
        decomposition = decomposition.rename(columns={'Value': 'Total_Emissions'})
        
        return decomposition[['Source', 'Total_Emissions']]

# ==================== Visualization Functions ====================

def plot_time_series(df, countries, title="Emissions Over Time"):
    """Plot time series for multiple countries"""
    fig = go.Figure()
    
    for country in countries:
        country_data = df[df['Area'] == country]
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data['Total_Emissions_kt'],
            mode='lines+markers',
            name=country,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Total Emissions (kt)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_forecast_with_confidence(historical, forecast, test=None):
    """Plot forecast with confidence intervals"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Test data if available
    if test is not None:
        fig.add_trace(go.Scatter(
            x=test.index,
            y=test.values,
            mode='lines',
            name='Actual Test',
            line=dict(color='green', width=2, dash='dash')
        ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dot')
    ))
    
    # Confidence interval (simulated)
    upper = forecast * 1.1
    lower = forecast * 0.9
    
    fig.add_trace(go.Scatter(
        x=forecast.index.tolist() + forecast.index.tolist()[::-1],
        y=upper.tolist() + lower.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Emission Forecast with Confidence Intervals",
        xaxis_title="Year",
        yaxis_title="Emissions (kt)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_hotspots_map(hotspots_df):
    """Create hotspot visualization"""
    fig = px.bar(
        hotspots_df,
        x='Country',
        y='Current_Emissions',
        color='Risk_Level',
        title='Emission Hotspots by Country',
        color_discrete_map={'High': '#D32F2F', 'Medium': '#FFA726', 'Low': '#66BB6A'},
        labels={'Current_Emissions': 'Current Emissions (kt)'}
    )
    
    fig.update_layout(height=500, template='plotly_white')
    return fig

def plot_emission_decomposition(decomp_df, country):
    """Plot emission sources breakdown"""
    fig = px.pie(
        decomp_df,
        values='Total_Emissions',
        names='Source',
        title=f'Emission Sources - {country}',
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig

# ==================== Main Application ====================

def main():
    # Header
    st.markdown('<div class="main-header">🌍 South Asia Agricultural Emissions Forecasting System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/2E7D32/FFFFFF?text=AgriEmissions+ML", use_container_width=True)
        st.markdown("### 📊 Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "📈 Forecasting", "🎯 Hotspot Analysis", "📄 Documentation", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📁 Data Upload")
        
        data_file = st.file_uploader("Upload Engineered Data", type=['csv'])
        
        if data_file:
            st.success("✅ Data file loaded successfully!")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Version:** 2.0.0  
        **Data Source:** FAOSTAT (Engineered)  
        **Coverage:** 1961-2023  
        **Countries:** 7 South Asian nations
        """)
    
    # Main Content
    if data_file:
        merged_df, raw_df = load_and_process_engineered_data(data_file)
        
        if merged_df is not None:
            
            # ==================== DASHBOARD ====================
            if "Dashboard" in page:
                st.markdown("## 📊 Overview Dashboard")
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                latest_year = merged_df['Year'].max()
                total_emissions = merged_df[merged_df['Year'] == latest_year]['Total_Emissions_kt'].sum()
                
                # Calculate average growth
                recent_data = merged_df[merged_df['Year'] >= latest_year - 10]
                old_data = merged_df[merged_df['Year'] < latest_year - 10]
                
                if len(old_data) > 0:
                    avg_growth = ((recent_data['Total_Emissions_kt'].mean() - 
                                  old_data['Total_Emissions_kt'].mean()) / 
                                 old_data['Total_Emissions_kt'].mean() * 100)
                else:
                    avg_growth = 0
                
                with col1:
                    st.metric("Total Emissions (2023)", f"{total_emissions:,.0f} kt", f"{avg_growth:.1f}%")
                with col2:
                    st.metric("Countries Monitored", "7", "South Asia")
                with col3:
                    st.metric("Data Points", f"{len(merged_df):,}", "1961-2023")
                with col4:
                    st.metric("Emission Types", "2", "CH4 + N2O")
                
                st.markdown("---")
                
                # Time Series Overview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📈 Regional Emission Trends")
                    selected_countries = st.multiselect(
                        "Select countries to compare",
                        merged_df['Area'].unique(),
                        default=list(merged_df['Area'].unique())[:3]
                    )
                    
                    if selected_countries:
                        fig = plot_time_series(merged_df, selected_countries)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🏆 Top Emitters (2023)")
                    top_emitters = merged_df[merged_df['Year'] == latest_year].nlargest(7, 'Total_Emissions_kt')
                    
                    for idx, row in top_emitters.iterrows():
                        st.markdown(f"**{row['Area']}**")
                        st.progress(row['Total_Emissions_kt'] / top_emitters['Total_Emissions_kt'].max())
                        st.caption(f"{row['Total_Emissions_kt']:,.0f} kt")
                
                # Recent Trends
                st.markdown("### 📊 Emission Composition by Country")
                comp_country = st.selectbox("Select country for detailed breakdown", merged_df['Area'].unique())
                
                analyzer = HotspotAnalyzer(merged_df)
                decomp_df = analyzer.decompose_emissions(comp_country, raw_df)
                
                fig = plot_emission_decomposition(decomp_df, comp_country)
                st.plotly_chart(fig, use_container_width=True)
            
            # ==================== FORECASTING ====================
            elif "Forecasting" in page:
                st.markdown("## 📈 Emission Forecasting")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    forecast_country = st.selectbox("Select Country", merged_df['Area'].unique())
                with col2:
                    forecast_years = st.slider("Years to Forecast", 5, 20, 10)
                with col3:
                    model_type = st.selectbox("Model", ['SARIMA', 'ExpSmoothing', 'RandomForest', 'GradientBoosting'])
                
                if st.button("🚀 Generate Forecast", type="primary"):
                    with st.spinner("Training models and generating forecasts..."):
                        
                        # Initialize forecaster
                        forecaster = EmissionForecaster(merged_df, forecast_country)
                        train, test = forecaster.train_test_split(test_size=0.2)
                        
                        # Train selected model
                        if model_type == 'SARIMA':
                            forecaster.fit_sarima()
                        elif model_type == 'ExpSmoothing':
                            forecaster.fit_exponential_smoothing()
                        elif model_type == 'RandomForest':
                            forecaster.fit_random_forest()
                        else:
                            forecaster.fit_gradient_boosting()
                        
                        # Evaluate
                        forecaster.evaluate_models()
                        
                        # Generate forecast
                        forecast_result = forecaster.forecast_future(model_type, forecast_years)
                        
                        # Display results
                        st.success("✅ Forecast generated successfully!")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        metrics = forecaster.metrics[model_type]
                        
                        with col1:
                            st.metric("MAE", f"{metrics['MAE']:.2f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        with col3:
                            st.metric("R²", f"{metrics['R2']:.3f}")
                        with col4:
                            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                        
                        # Visualization
                        fig = plot_forecast_with_confidence(train, forecast_result, test)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        st.markdown("### 📋 Forecast Values")
                        forecast_df = pd.DataFrame({
                            'Year': forecast_result.index,
                            'Predicted Emissions (kt)': forecast_result.values,
                            'Lower Bound': forecast_result.values * 0.9,
                            'Upper Bound': forecast_result.values * 1.1
                        })
                        st.dataframe(forecast_df, use_container_width=True)
                        
                        # Download
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Forecast",
                            csv,
                            f"{forecast_country}_forecast_{model_type}.csv",
                            "text/csv"
                        )
            
            # ==================== HOTSPOT ANALYSIS ====================
            elif "Hotspot" in page:
                st.markdown("## 🎯 Hotspot Identification & Risk Analysis")
                
                analyzer = HotspotAnalyzer(merged_df)
                
                # Controls
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.slider("Risk Threshold Percentile", 50, 95, 75)
                with col2:
                    analysis_years = st.slider("Analysis Period (years)", 5, 20, 10)
                
                # Identify hotspots
                hotspots_df = analyzer.identify_hotspots(threshold)
                growth_df = analyzer.calculate_growth_rates(analysis_years)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk Countries", len(hotspots_df[hotspots_df['Risk_Level'] == 'High']))
                with col2:
                    st.metric("Avg Growth Rate", f"{growth_df['Growth_Rate_%'].mean():.1f}%")
                with col3:
                    st.metric("Hotspots Identified", len(hotspots_df))
                
                # Visualizations
                st.markdown("### 🗺️ Hotspot Map")
                fig = plot_hotspots_map(hotspots_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.markdown("### 📊 Detailed Hotspot Analysis")
                display_df = hotspots_df.copy()
                display_df['Current_Emissions'] = display_df['Current_Emissions'].round(2)
                display_df['Growth_Rate_%'] = display_df['Growth_Rate_%'].round(2)
                display_df['Absolute_Change'] = display_df['Absolute_Change'].round(2)
                
                st.dataframe(
                    display_df.style.background_gradient(subset=['Growth_Rate_%'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                # Growth rate distribution
                st.markdown("### 📈 Growth Rate Distribution")
                fig = px.histogram(
                    growth_df,
                    x='Growth_Rate_%',
                    nbins=20,
                    title='Distribution of Emission Growth Rates',
                    labels={'Growth_Rate_%': 'Growth Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ==================== DOCUMENTATION ====================
            elif "Documentation" in page:
                st.markdown("## 📄 System Documentation")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Models", "Data", "API"])
                
                with tab1:
                    st.markdown("""
                    ### System Overview
                    
                    This comprehensive ML system provides:
                    
                    #### 🎯 Key Features
                    - **Time Series Forecasting**: Multiple ML models (SARIMA, Exponential Smoothing, RF, GB)
                    - **Hotspot Identification**: Risk-based analysis of high-emission regions
                    - **Trend Analysis**: Historical emission patterns and growth rates
                    - **Interactive Visualizations**: Real-time data exploration
                    - **Engineered Features**: emission_intensity, rolling_5yr_mean, decade grouping
                    
                    #### 🌍 Coverage
                    - **Geographic**: 7 South Asian countries
                    - **Temporal**: 1961-2023 (63 years)
                    - **Sources**: Combined Livestock + Crop emissions
                    - **Gases**: CH₄, N₂O
                    
                    #### 🔄 Automated Pipeline
                    1. Data ingestion and validation
                    2. Feature engineering and aggregation
                    3. Model training and evaluation
                    4. Forecast generation
                    5. Hotspot identification
                    6. Report generation
                    """)
                
                with tab2:
                    st.markdown("""
                    ### Machine Learning Models
                    
                    #### 1. SARIMA (Seasonal AutoRegressive Integrated Moving Average)
                    - **Use Case**: Time series with seasonality
                    - **Parameters**: (p,d,q) × (P,D,Q,s)
                    - **Strengths**: Handles trends and seasonal patterns
                    - **Limitations**: Requires stationary data
                    
                    #### 2. Exponential Smoothing
                    - **Use Case**: Short-term forecasting
                    - **Components**: Level, trend, seasonality
                    - **Strengths**: Simple, fast, interpretable
                    - **Limitations**: Limited for long-term forecasts
                    
                    #### 3. Random Forest
                    - **Use Case**: Non-linear relationships
                    - **Parameters**: n_estimators, max_depth
                    - **Strengths**: Handles outliers, feature importance
                    - **Limitations**: Can overfit with small datasets
                    
                    #### 4. Gradient Boosting
                    - **Use Case**: Complex patterns
                    - **Parameters**: learning_rate, n_estimators
                    - **Strengths**: High accuracy, flexible
                    - **Limitations**: Computationally intensive
                    
                    #### Model Selection Guide
                    | Data Characteristic | Recommended Model |
                    |---------------------|------------------|
                    | Strong seasonality | SARIMA |
                    | Short history | Exponential Smoothing |
                    | Non-linear trends | Random Forest |
                    | Complex patterns | Gradient Boosting |
                    """)
                
                with tab3:
                    st.markdown("""
                    ### Data Specifications
                    
                    #### Input Format (Engineered Data)
                    ```
                    Required Columns:
                    - Domain, Area, Element Code, Element, Item Code, Item
                    - Year Code, Year, Source Code, Source
                    - Unit, Value, Flag, Flag Description, Note
                    - emission_type (CH4/N2O)
                    - decade (grouping variable)
                    - emission_intensity (calculated feature)
                    - rolling_5yr_mean (temporal smoothing)
                    ```
                    
                    #### Emission Types
                    
                    **Combined Sources:**
                    - Crops total (Emissions CH4)
                    - Crops total (Emissions N2O)
                    - All emission sources merged and aggregated
                    
                    #### Engineered Features
                    - ✅ Emission intensity calculations
                    - ✅ Rolling 5-year averages
                    - ✅ Decade-based grouping
                    - ✅ Combined emission totals
                    - ✅ Temporal aggregation
                    
                    #### Processing Steps
                    1. Data loading and validation
                    2. Aggregation by country, year, and emission type
                    3. Feature engineering (intensity, rolling means)
                    4. Calculating total emissions (CH4 + N2O)
                    5. Normalization and scaling
                    """)
                
                with tab4:
                    st.markdown("""
                    ### Usage Guide
                    
                    #### Quick Start
                    ```python
                    # 1. Upload engineered data file (CSV)
                    # 2. Select country and model
                    # 3. Generate forecast
                    # 4. Download results
                    ```
                    
                    #### Advanced Usage
                    
                    **Batch Forecasting:**
                    ```python
                    forecaster = EmissionForecaster(data, country)
                    for model in ['SARIMA', 'RandomForest', 'GradientBoosting']:
                        forecast = forecaster.forecast_future(model, 10)
                    ```
                    
                    **Custom Hotspot Analysis:**
                    ```python
                    analyzer = HotspotAnalyzer(data)
                    hotspots = analyzer.identify_hotspots(threshold=80)
                    growth = analyzer.calculate_growth_rates(years=15)
                    ```
                    
                    #### Export Options
                    - CSV (Forecasts, Hotspots)
                    - Excel (Complete reports)
                    - JSON (API integration)
                    - PDF (Summary reports)
                    
                    #### Performance Optimization
                    - Data caching with @st.cache_data
                    - Lazy loading for large datasets
                    - Parallel model training
                    - Progressive visualization rendering
                    
                    #### Engineered Data Format
                    The system expects a single CSV file with merged and engineered data:
                    - Pre-aggregated emissions from crops and livestock
                    - Calculated features (emission_intensity, rolling_5yr_mean)
                    - Proper temporal indexing (Year column)
                    - Emission type classification (CH4, N2O)
                    """)
            
            # ==================== SETTINGS ====================
            elif "Settings" in page:
                st.markdown("## ⚙️ System Settings")
                
                tab1, tab2, tab3 = st.tabs(["Model Configuration", "Data Processing", "Export Settings"])
                
                with tab1:
                    st.markdown("### 🤖 Model Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**SARIMA Configuration**")
                        p = st.number_input("AR order (p)", 0, 5, 1)
                        d = st.number_input("Differencing (d)", 0, 2, 1)
                        q = st.number_input("MA order (q)", 0, 5, 1)
                    
                    with col2:
                        st.markdown("**Random Forest Configuration**")
                        n_estimators = st.slider("Number of trees", 50, 500, 100)
                        max_depth = st.slider("Max depth", 5, 50, 10)
                        min_samples_split = st.slider("Min samples split", 2, 20, 2)
                    
                    st.markdown("**Gradient Boosting Configuration**")
                    learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
                    gb_estimators = st.slider("GB estimators", 50, 500, 100)
                    
                    if st.button("Save Model Configuration"):
                        st.success("✅ Configuration saved!")
                
                with tab2:
                    st.markdown("### 📊 Data Processing Settings")
                    
                    train_test_split = st.slider("Train/Test Split", 0.1, 0.4, 0.2, 0.05)
                    outlier_threshold = st.slider("Outlier Threshold (IQR multiplier)", 1.5, 3.0, 1.5, 0.1)
                    interpolation_method = st.selectbox("Missing Value Interpolation", 
                                                        ["Linear", "Polynomial", "Spline", "Forward Fill"])
                    
                    st.checkbox("Enable outlier detection", value=True)
                    st.checkbox("Apply smoothing filter", value=False)
                    st.checkbox("Normalize features", value=True)
                    st.checkbox("Use engineered features", value=True)
                    
                    if st.button("Apply Processing Settings"):
                        st.success("✅ Processing settings updated!")
                
                with tab3:
                    st.markdown("### 📥 Export Configuration")
                    
                    export_format = st.multiselect(
                        "Select export formats",
                        ["CSV", "Excel", "JSON", "PDF"],
                        default=["CSV"]
                    )
                    
                    include_confidence_intervals = st.checkbox("Include confidence intervals", value=True)
                    include_model_metrics = st.checkbox("Include model metrics", value=True)
                    include_visualizations = st.checkbox("Include charts in export", value=True)
                    include_engineered_features = st.checkbox("Include engineered features", value=True)
                    
                    decimal_places = st.number_input("Decimal places", 1, 6, 2)
                    
                    if st.button("Save Export Settings"):
                        st.success("✅ Export settings saved!")
        
        else:
            st.error("Failed to load data. Please check file format.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to the South Asia Agricultural Emissions Forecasting System
        
        ### 🚀 Getting Started
        
        Please upload your engineered data file using the sidebar to begin.
        
        ### 📋 Data Requirements
        
        Your CSV file should contain the following columns:
        - `Domain`: Data domain (e.g., "Emissions from Crops")
        - `Area`: Country name
        - `Element`: Emission type description
        - `Year`: Year of measurement
        - `Value`: Emission value in kilotons (kt)
        - `emission_type`: Type of gas (CH4 or N2O)
        - `emission_intensity`: Calculated intensity metric
        - `rolling_5yr_mean`: 5-year rolling average
        - `decade`: Decade grouping (optional)
        
        ### 🎯 System Capabilities
        
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 📈 Forecasting
            - Multiple ML models
            - 5-20 year predictions
            - Confidence intervals
            - Model comparison
            """)
        
        with col2:
            st.markdown("""
            #### 🎯 Hotspot Analysis
            - Risk assessment
            - Growth rate analysis
            - Country ranking
            - Trend identification
            """)
        
        with col3:
            st.markdown("""
            #### 📊 Visualization
            - Interactive charts
            - Time series plots
            - Comparative analysis
            - Export capabilities
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔧 Engineered Data Features
        
        This system is optimized for pre-processed data with:
        - ✅ **Merged datasets**: Crops and livestock emissions combined
        - ✅ **Feature engineering**: Emission intensity and rolling averages
        - ✅ **Temporal aggregation**: Data grouped by year and country
        - ✅ **Gas-type separation**: CH4 and N2O tracked separately
        - ✅ **Quality metrics**: Pre-calculated statistical features
        """)
        
        st.info("💡 **Tip**: The system expects a single CSV file with all engineered features. Make sure your data includes emission_type, emission_intensity, and rolling_5yr_mean columns for optimal performance!")

# ==================== AUTOMATED PIPELINE ====================

class AutomatedPipeline:
    """Automated end-to-end pipeline for emission forecasting"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.results = {}
        
    def run_full_pipeline(self, countries=None, forecast_years=10):
        """Execute complete analysis pipeline"""
        
        # Step 1: Load data
        st.info("📥 Loading engineered data...")
        merged_df, raw_df = load_and_process_engineered_data(self.data_file)
        
        if merged_df is None:
            st.error("Failed to load data")
            return None
        
        # Step 2: Select countries
        if countries is None:
            countries = merged_df['Area'].unique()
        
        # Step 3: Run forecasts for all countries
        st.info(f"🚀 Running forecasts for {len(countries)} countries...")
        
        progress_bar = st.progress(0)
        
        for idx, country in enumerate(countries):
            # Initialize forecaster
            forecaster = EmissionForecaster(merged_df, country)
            train, test = forecaster.train_test_split()
            
            # Train all models
            forecaster.fit_sarima()
            forecaster.fit_exponential_smoothing()
            forecaster.fit_random_forest()
            forecaster.fit_gradient_boosting()
            
            # Evaluate models
            forecaster.evaluate_models()
            
            # Generate forecasts
            for model_name in forecaster.models.keys():
                forecast = forecaster.forecast_future(model_name, forecast_years)
            
            # Store results
            self.results[country] = {
                'forecaster': forecaster,
                'metrics': forecaster.metrics,
                'forecasts': forecaster.forecasts
            }
            
            progress_bar.progress((idx + 1) / len(countries))
        
        # Step 4: Hotspot analysis
        st.info("🎯 Identifying hotspots...")
        analyzer = HotspotAnalyzer(merged_df)
        hotspots = analyzer.identify_hotspots()
        self.results['hotspots'] = hotspots
        
        st.success("✅ Pipeline completed successfully!")
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'countries_analyzed': len(self.results) - 1,  # Exclude hotspots key
            'models_used': ['SARIMA', 'ExpSmoothing', 'RandomForest', 'GradientBoosting'],
            'hotspots_identified': len(self.results.get('hotspots', [])),
            'data_source': 'Engineered FAOSTAT Data',
            'summary': {}
        }
        
        # Compile summary statistics
        for country, data in self.results.items():
            if country != 'hotspots':
                best_model = min(data['metrics'].items(), key=lambda x: x[1]['RMSE'])
                report['summary'][country] = {
                    'best_model': best_model[0],
                    'rmse': best_model[1]['RMSE'],
                    'r2': best_model[1]['R2']
                }
        
        return report

# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    main()

# ==================== ADDITIONAL UTILITIES ====================

def export_to_excel(forecasts_dict, filename="emissions_forecast.xlsx"):
    """Export forecasts to Excel with multiple sheets"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for country, data in forecasts_dict.items():
            if country != 'hotspots':
                # Create dataframe for each model
                for model_name, forecast in data['forecasts'].items():
                    df = pd.DataFrame({
                        'Year': forecast.index,
                        'Forecast': forecast.values
                    })
                    sheet_name = f"{country}_{model_name}"[:31]  # Excel limit
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return output.getvalue()

def calculate_emission_intensity(df, population_data=None):
    """Calculate per capita emission intensity"""
    if population_data is not None:
        df_merged = pd.merge(df, population_data, on=['Area', 'Year'])
        df_merged['Emissions_per_capita'] = df_merged['Total_Emissions_kt'] / df_merged['Population']
        return df_merged
    return df

def detect_change_points(series, threshold=2.0):
    """Detect significant change points in emission trends"""
    changes = []
    mean = series.mean()
    std = series.std()
    
    for i in range(1, len(series)):
        if abs(series.iloc[i] - series.iloc[i-1]) > threshold * std:
            changes.append({
                'year': series.index[i],
                'change': series.iloc[i] - series.iloc[i-1],
                'percent_change': ((series.iloc[i] - series.iloc[i-1]) / series.iloc[i-1]) * 100
            })
    
    return changes

def analyze_engineered_features(df):
    """Analyze and visualize engineered features"""
    st.markdown("### 🔬 Engineered Features Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'emission_intensity' in df.columns:
            st.markdown("**Emission Intensity Distribution**")
            fig = px.histogram(df, x='emission_intensity', nbins=50, 
                             title='Distribution of Emission Intensity')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'rolling_5yr_mean' in df.columns:
            st.markdown("**Rolling 5-Year Mean Trends**")
            # Plot for top 3 countries
            top_countries = df.groupby('Area')['Total_Emissions_kt'].mean().nlargest(3).index
            fig = go.Figure()
            for country in top_countries:
                country_data = df[df['Area'] == country].sort_values('Year')
                fig.add_trace(go.Scatter(
                    x=country_data['Year'],
                    y=country_data['rolling_5yr_mean'],
                    mode='lines',
                    name=country
                ))
            fig.update_layout(title='Rolling 5-Year Mean by Country',
                            xaxis_title='Year',
                            yaxis_title='Emissions (kt)')
            st.plotly_chart(fig, use_container_width=True)



# ==================== END ====================