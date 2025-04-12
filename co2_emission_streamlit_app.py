import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Set page config
st.set_page_config(
    page_title="CO2 Emissions Predictor",
    page_icon="🚗",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004D40;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        background-color: #e1f5fe;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🚗 Vehicle CO2 Emissions Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<p class='info-text'>This app analyzes and predicts CO2 emissions from vehicles based on Canada's vehicle emissions dataset. 
You can explore the data, learn about what affects emissions, and predict CO2 emissions for specific vehicle configurations.</p>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('CO2 Emissions_Canada.csv')
    return df

# Load and preprocess data
try:
    df = load_data()
    
    # Create sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Exploration", "Prediction Model", "Compare Vehicles", "About"])
    
    if page == "Data Exploration":
        st.markdown("<h2 class='sub-header'>📊 Data Exploration</h2>", unsafe_allow_html=True)
        
        # Display raw data sample
        with st.expander("View Raw Data Sample"):
            st.dataframe(df.head())
        
        # Summary statistics
        with st.expander("Data Summary"):
            st.write("Dataset Shape:", df.shape)
            st.write("Summary Statistics:")
            st.dataframe(df.describe())
        
        # Data distributions
        st.markdown("<h3 class='sub-header'>CO2 Emissions Distribution</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['CO2 Emissions(g/km)'], kde=True, ax=ax)
            ax.set_title('Distribution of CO2 Emissions')
            ax.set_xlabel('CO2 Emissions (g/km)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            # Box plot of CO2 by vehicle class
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_classes = df.groupby('Vehicle Class')['CO2 Emissions(g/km)'].mean().sort_values().index
            sns.boxplot(x='Vehicle Class', y='CO2 Emissions(g/km)', data=df, 
                        order=sorted_classes, ax=ax)
            ax.set_title('CO2 Emissions by Vehicle Class')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation analysis
        st.markdown("<h3 class='sub-header'>Feature Correlations</h3>", unsafe_allow_html=True)
        
        # Select numerical columns
        numerical_cols = ['Engine Size(L)', 'Cylinders', 
                          'Fuel Consumption City (L/100 km)', 
                          'Fuel Consumption Hwy (L/100 km)', 
                          'Fuel Consumption Comb (L/100 km)',
                          'Fuel Consumption Comb (mpg)',
                          'CO2 Emissions(g/km)']
        
        # Calculate and plot correlation matrix
        corr_matrix = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
        
        # Key relationships
        st.markdown("<h3 class='sub-header'>Key Relationships with CO2 Emissions</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot fuel consumption vs CO2 emissions
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='Fuel Consumption Comb (L/100 km)', 
                            y='CO2 Emissions(g/km)', 
                            hue='Fuel Type', 
                            data=df, ax=ax, alpha=0.6)
            ax.set_title('Combined Fuel Consumption vs CO2 Emissions')
            st.pyplot(fig)
            
            st.markdown("""
            <p class='info-text'>There's a very strong linear relationship between fuel consumption and CO2 emissions. 
            This makes sense as burning more fuel directly produces more CO2.</p>
            """, unsafe_allow_html=True)
        
        with col2:
            # Plot engine size vs CO2 emissions
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='Engine Size(L)', 
                            y='CO2 Emissions(g/km)', 
                            hue='Fuel Type', 
                            data=df, ax=ax, alpha=0.6)
            ax.set_title('Engine Size vs CO2 Emissions')
            st.pyplot(fig)
            
            st.markdown("""
            <p class='info-text'>Engine size is also strongly correlated with CO2 emissions. 
            Larger engines typically consume more fuel, resulting in higher emissions.</p>
            """, unsafe_allow_html=True)
        
        # CO2 by fuel type
        st.markdown("<h3 class='sub-header'>CO2 Emissions by Fuel Type</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot of CO2 by fuel type
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_fuel = df.groupby('Fuel Type')['CO2 Emissions(g/km)'].mean().sort_values().index
            sns.boxplot(x='Fuel Type', y='CO2 Emissions(g/km)', data=df, 
                       order=sorted_fuel, ax=ax)
            ax.set_title('CO2 Emissions by Fuel Type')
            st.pyplot(fig)
            
        with col2:
            # Bar chart of average CO2 by fuel type
            fuel_avg = df.groupby('Fuel Type')['CO2 Emissions(g/km)'].mean().sort_values()
            fuel_count = df.groupby('Fuel Type').size()
            
            fuel_data = pd.DataFrame({
                'Average CO2 (g/km)': fuel_avg,
                'Count': fuel_count
            }).reset_index()
            
            st.dataframe(fuel_data)
            
            st.markdown("""
            <p class='info-text'>
            <b>Fuel Type Key:</b><br>
            Z = Premium Gasoline<br>
            X = Regular Gasoline<br>
            D = Diesel<br>
            E = Ethanol (E85)<br>
            N = Natural Gas
            </p>
            """, unsafe_allow_html=True)
        
        # Top manufacturers by CO2 emissions
        st.markdown("<h3 class='sub-header'>Manufacturers by Average CO2 Emissions</h3>", unsafe_allow_html=True)
        
        # Filter manufacturers with at least 30 models
        make_counts = df['Make'].value_counts()
        makes_to_include = make_counts[make_counts >= 30].index
        
        make_avg = df[df['Make'].isin(makes_to_include)].groupby('Make')['CO2 Emissions(g/km)'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        make_avg.plot(kind='barh', ax=ax)
        ax.set_title('Average CO2 Emissions by Manufacturer')
        ax.set_xlabel('Average CO2 Emissions (g/km)')
        ax.set_ylabel('Manufacturer')
        st.pyplot(fig)
    
    elif page == "Prediction Model":
        st.markdown("<h2 class='sub-header'>🔮 CO2 Emissions Prediction</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <p class='info-text'>This section allows you to predict CO2 emissions for a vehicle based on its characteristics. 
        Select values for each feature, and the model will estimate the CO2 emissions.</p>
        """, unsafe_allow_html=True)
        
        # Create tab for model explanation and prediction interface
        model_tab, performance_tab = st.tabs(["Prediction", "Model Performance"])
        
        with model_tab:
            # Create interface for prediction
            model_types = ["Gradient Boosting", "Random Forest", "Linear Regression", "Simple Formula"]
            selected_model = st.selectbox("Select Model Type", model_types)
            st.markdown("<h3>Enter Vehicle Specifications</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                make = st.selectbox("Manufacturer", sorted(df['Make'].unique()))
                vehicle_class = st.selectbox("Vehicle Class", sorted(df['Vehicle Class'].unique()))
                engine_size = st.slider("Engine Size (L)", 
                                     min_value=float(df['Engine Size(L)'].min()),
                                     max_value=float(df['Engine Size(L)'].max()),
                                     value=2.0,
                                     step=0.1)
                cylinders = st.slider("Number of Cylinders", 
                                    min_value=int(df['Cylinders'].min()),
                                    max_value=int(df['Cylinders'].max()),
                                    value=4)
            
            with col2:
                fuel_type = st.selectbox("Fuel Type", sorted(df['Fuel Type'].unique()),
                                      format_func=lambda x: {
                                          'Z': 'Premium Gasoline',
                                          'X': 'Regular Gasoline',
                                          'D': 'Diesel',
                                          'E': 'Ethanol (E85)',
                                          'N': 'Natural Gas'
                                      }.get(x, x))
                
                transmission = st.selectbox("Transmission", sorted(df['Transmission'].unique()))
                
                fuel_consumption_city = st.slider("Fuel Consumption City (L/100 km)", 
                                             min_value=float(df['Fuel Consumption City (L/100 km)'].min()),
                                             max_value=float(df['Fuel Consumption City (L/100 km)'].max()),
                                             value=10.0,
                                             step=0.1)
                
                fuel_consumption_hwy = st.slider("Fuel Consumption Highway (L/100 km)", 
                                             min_value=float(df['Fuel Consumption Hwy (L/100 km)'].min()),
                                             max_value=float(df['Fuel Consumption Hwy (L/100 km)'].max()),
                                             value=7.0,
                                             step=0.1)
            
            # Calculate combined fuel consumption automatically
            fuel_consumption_comb = (0.55 * fuel_consumption_city) + (0.45 * fuel_consumption_hwy)
            st.info(f"Calculated Combined Fuel Consumption: {fuel_consumption_comb:.1f} L/100 km")
            
            # Train models and predict
            model_data = build_and_train_model(df)
            
            # Create input data for prediction
            input_data = pd.DataFrame({
                'Engine Size(L)': [engine_size],
                'Cylinders': [cylinders],
                'Fuel Consumption Comb (L/100 km)': [fuel_consumption_comb],
                'Fuel Type': [fuel_type],
                'Vehicle Class': [vehicle_class]
            })
            
            # Make prediction based on selected model
            if selected_model == "Simple Formula":
                # Simple formula based on combined fuel consumption
                # CO2 (g/km) ≈ 22.5 * Fuel Consumption (L/100 km) + 15
                prediction = 22.5 * fuel_consumption_comb + 15
                model_note = """
                <p class='info-text'>Using simplified formula: CO2 (g/km) ≈ 22.5 × Fuel Consumption (L/100 km) + 15</p>
                """
            else:
                # Use the selected machine learning model
                model_pipeline = model_data['results'][selected_model]['pipeline']
                prediction = model_pipeline.predict(input_data)[0]
                model_note = f"""
                <p class='info-text'>Using {selected_model} model with 
                R² score: {model_data['results'][selected_model]['r2']:.3f}, 
                RMSE: {model_data['results'][selected_model]['rmse']:.1f}</p>
                """
            
            st.markdown(model_note, unsafe_allow_html=True)
            
            # Display prediction
            st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Predicted CO2 Emissions</h4>
                    <h2>{prediction:.1f} g/km</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Calculate emission rating
                if prediction < 130:
                    rating = "A (Very Low Emissions)"
                    color = "green"
                elif prediction < 170:
                    rating = "B (Low Emissions)"
                    color = "lightgreen"
                elif prediction < 210:
                    rating = "C (Average Emissions)"
                    color = "yellow"
                elif prediction < 250:
                    rating = "D (High Emissions)"
                    color = "orange"
                else:
                    rating = "E (Very High Emissions)"
                    color = "red"
                
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Emission Rating</h4>
                    <h2 style='color: {color};'>{rating}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Show comparable vehicles
            st.markdown("<h3>Similar Vehicles</h3>", unsafe_allow_html=True)
            
            similar_vehicles = df[
                (df['Engine Size(L)'].between(engine_size - 0.5, engine_size + 0.5)) &
                (df['Cylinders'] == cylinders) &
                (df['Fuel Type'] == fuel_type) &
                (df['Vehicle Class'] == vehicle_class)
            ].sort_values(by='CO2 Emissions(g/km)')
            
            if len(similar_vehicles) > 0:
              st.dataframe(similar_vehicles[['Make', 'Model', 'Engine Size(L)', 
                                                'Fuel Consumption Comb (L/100 km)', 
                                                'CO2 Emissions(g/km)']].head(5))
            else:
              st.info("No similar vehicles found. Try adjusting the parameters.")
            
            # Create and train model
            @st.cache_resource
            def build_and_train_model(df):
                # Prepare features and target
                X = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'Fuel Type', 'Vehicle Class']]
                y = df['CO2 Emissions(g/km)']
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Define preprocessing for categorical features
                categorical_features = ['Fuel Type', 'Vehicle Class']
                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                # Define preprocessing for numeric features
                numeric_features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
                numeric_transformer = Pipeline(steps=[
                    ('scaler', StandardScaler())
                ])
                
                # Save test data for model evaluation
                test_data = {
                    'X_test': X_test,
                    'y_test': y_test
                }
                
                # Combine preprocessing steps
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])
                
                # Define the models to evaluate
                models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
                }
                
                # Create pipelines for each model
                pipelines = {}
                for name, model in models.items():
                    pipelines[name] = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                
                # Train and evaluate each model
                results = {}
                for name, pipeline in pipelines.items():
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    results[name] = {
                        'pipeline': pipeline,
                        'mae': mean_absolute_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'r2': r2_score(y_test, y_pred)
                    }
                
                # Find the best model based on R2 score
                best_model_name = max(results, key=lambda k: results[k]['r2'])
                best_pipeline = results[best_model_name]['pipeline']
                
                return {
                    'results': results,
                    'best_model': best_model_name,
                    'best_pipeline': best_pipeline,
                    'test_data': test_data
                    }
except Error:
  print("Error:", Error)
