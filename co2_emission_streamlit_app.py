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
        background-color: #d38434;
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
        text-color: #004D40;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🚗 Vehicle CO2 Emissions Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<p class='info-text'>This app analyzes and predicts CO2 emissions from vehicles based on Canada's vehicle emissions dataset.
You can explore the data, learn about what affects emissions, and predict CO2 emissions for specific vehicle configurations.</p>
""", unsafe_allow_html=True)

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
        'test_data': test_data,
        'feature_names': X.columns.tolist()
    }

# Load data
@st.cache_data
def load_data():
    # df = pd.read_csv('CO2 Emissions_Canada.csv')
    df = pd.read_csv('https://raw.githubusercontent.com/arorashubham/CO2_Emission/refs/heads/main/CO2_Emissions_Canada.csv')
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
        st.markdown("<h2 class='sub-header'>🔮 CO2 Emissions Prediction</h2>", unsafe_allow_html=False)

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

        with performance_tab:
            st.markdown("<h3>Model Performance Comparison</h3>", unsafe_allow_html=True)

            # Compare model performance metrics
            results = model_data['results']
            performance_df = pd.DataFrame({
                'Model': list(results.keys()),
                'MAE': [results[m]['mae'] for m in results],
                'RMSE': [results[m]['rmse'] for m in results],
                'R² Score': [results[m]['r2'] for m in results]
            })

            st.dataframe(performance_df.sort_values('R² Score', ascending=False).reset_index(drop=True))

            # Visualize model performance
            st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)

            # Feature importance for Random Forest model
            if 'Random Forest' in results:
                feature_names = model_data['feature_names']
                rf_model = results['Random Forest']['pipeline'].named_steps['model']
                preprocessor = results['Random Forest']['pipeline'].named_steps['preprocessor']

                # Get the feature names after one-hot encoding
                categorical_features = ['Fuel Type', 'Vehicle Class']
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
                numeric_features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']

                # Combine original numerical feature names with encoded categorical feature names
                all_feature_names = numeric_features + cat_feature_names

                # Get feature importance
                feature_importances = rf_model.feature_importances_

                # Create a DataFrame for visualization
                importance_df = pd.DataFrame({
                    'Feature': all_feature_names,
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=False)

                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
                ax.set_title('Top 10 Features by Importance (Random Forest)')
                st.pyplot(fig)

                st.markdown("""
                <p class='info-text'>Feature importance shows which factors most strongly influence CO2 emissions.
                As expected, fuel consumption is the dominant factor, followed by engine characteristics and vehicle class.</p>
                """, unsafe_allow_html=True)

            # Model error analysis
            st.markdown("<h3>Error Analysis</h3>", unsafe_allow_html=True)

            # Prepare test data for visualization
            X_test = model_data['test_data']['X_test']
            y_test = model_data['test_data']['y_test']

            # Make predictions using all models
            test_results = pd.DataFrame({
                'Actual': y_test
            })

            for model_name, model_info in results.items():
                test_results[f'Predicted ({model_name})'] = model_info['pipeline'].predict(X_test)

            # Combine with fuel consumption for visualization
            test_results['Fuel Consumption'] = X_test['Fuel Consumption Comb (L/100 km)'].values

            # Plot actual vs predicted for the best model
            best_model = model_data['best_model']

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x='Actual',
                y=f'Predicted ({best_model})',
                data=test_results,
                alpha=0.5,
                ax=ax
            )

            # Add perfect prediction line
            max_val = max(test_results['Actual'].max(), test_results[f'Predicted ({best_model})'].max())
            min_val = min(test_results['Actual'].min(), test_results[f'Predicted ({best_model})'].min())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            ax.set_title(f'Actual vs Predicted CO2 Emissions ({best_model})')
            ax.set_xlabel('Actual CO2 Emissions (g/km)')
            ax.set_ylabel('Predicted CO2 Emissions (g/km)')
            st.pyplot(fig)

            # Plot residuals
            test_results[f'Residuals ({best_model})'] = test_results['Actual'] - test_results[f'Predicted ({best_model})']

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x=f'Predicted ({best_model})',
                y=f'Residuals ({best_model})',
                data=test_results,
                alpha=0.5,
                ax=ax
            )
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_title(f'Residuals vs Predicted Values ({best_model})')
            ax.set_xlabel('Predicted CO2 Emissions (g/km)')
            ax.set_ylabel('Residuals (g/km)')
            st.pyplot(fig)

            st.markdown("""
            <p class='info-text'>The residual plot helps identify if there are systematic patterns in prediction errors.
            Ideally, residuals should be randomly distributed around zero with no clear patterns.</p>
            """, unsafe_allow_html=True)

    elif page == "Compare Vehicles":
        st.markdown("<h2 class='sub-header'>🔍 Vehicle Comparison</h2>", unsafe_allow_html=True)

        st.markdown("""
        <p class='info-text'>Compare the CO2 emissions of different vehicle configurations side by side.
        This can help you understand how changes in vehicle specifications affect emissions.</p>
        """, unsafe_allow_html=True)

        # Create columns for vehicle comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3>Vehicle 1</h3>", unsafe_allow_html=True)
            make1 = st.selectbox("Manufacturer", sorted(df['Make'].unique()), key="make1")

            # Filter models by selected make
            models1 = sorted(df[df['Make'] == make1]['Model'].unique())
            model1 = st.selectbox("Model", models1, key="model1")

            # Get available vehicle configurations for the selected make/model
            configs1 = df[(df['Make'] == make1) & (df['Model'] == model1)]

            if len(configs1) > 0:
                # If multiple configurations exist, let user select one
                if len(configs1) > 1:
                    config_desc1 = [f"{row['Vehicle Class']} - {row['Engine Size(L)']}L, {row['Cylinders']} cyl, {row['Transmission']}, {row['Fuel Type']}"
                                   for _, row in configs1.iterrows()]
                    selected_config1 = st.selectbox("Configuration", config_desc1, key="config1")

                    # Get the index of the selected configuration
                    selected_idx1 = config_desc1.index(selected_config1)
                    vehicle1 = configs1.iloc[selected_idx1]
                else:
                    vehicle1 = configs1.iloc[0]

                # Display vehicle details
                st.markdown(f"""
                <div class='metric-card'>
                    <p><b>Engine:</b> {vehicle1['Engine Size(L)']}L, {vehicle1['Cylinders']} cylinders</p>
                    <p><b>Transmission:</b> {vehicle1['Transmission']}</p>
                    <p><b>Fuel Type:</b> {vehicle1['Fuel Type']}</p>
                    <p><b>Fuel Consumption:</b> {vehicle1['Fuel Consumption Comb (L/100 km)']} L/100 km</p>
                    <p><b>CO2 Emissions:</b> <span class='highlight'>{vehicle1['CO2 Emissions(g/km)']} g/km</span></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No data available for this make/model combination.")

        with col2:
            st.markdown("<h3>Vehicle 2</h3>", unsafe_allow_html=True)
            make2 = st.selectbox("Manufacturer", sorted(df['Make'].unique()), key="make2")

            # Filter models by selected make
            models2 = sorted(df[df['Make'] == make2]['Model'].unique())
            model2 = st.selectbox("Model", models2, key="model2")

            # Get available vehicle configurations for the selected make/model
            configs2 = df[(df['Make'] == make2) & (df['Model'] == model2)]

            if len(configs2) > 0:
                # If multiple configurations exist, let user select one
                if len(configs2) > 1:
                    config_desc2 = [f"{row['Vehicle Class']} - {row['Engine Size(L)']}L, {row['Cylinders']} cyl, {row['Transmission']}, {row['Fuel Type']}"
                                   for _, row in configs2.iterrows()]
                    selected_config2 = st.selectbox("Configuration", config_desc2, key="config2")

                    # Get the index of the selected configuration
                    selected_idx2 = config_desc2.index(selected_config2)
                    vehicle2 = configs2.iloc[selected_idx2]
                else:
                    vehicle2 = configs2.iloc[0]

                # Display vehicle details
                st.markdown(f"""
                <div class='metric-card'>
                    <p><b>Engine:</b> {vehicle2['Engine Size(L)']}L, {vehicle2['Cylinders']} cylinders</p>
                    <p><b>Transmission:</b> {vehicle2['Transmission']}</p>
                    <p><b>Fuel Type:</b> {vehicle2['Fuel Type']}</p>
                    <p><b>Fuel Consumption:</b> {vehicle2['Fuel Consumption Comb (L/100 km)']} L/100 km</p>
                    <p><b>CO2 Emissions:</b> <span class='highlight'>{vehicle2['CO2 Emissions(g/km)']} g/km</span></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No data available for this make/model combination.")

        # Comparison section (if both vehicles are selected)
        if 'vehicle1' in locals() and 'vehicle2' in locals():
            st.markdown("<h3>Comparison</h3>", unsafe_allow_html=True)

            # Calculate differences
            co2_diff = vehicle2['CO2 Emissions(g/km)'] - vehicle1['CO2 Emissions(g/km)']
            co2_diff_percent = (co2_diff / vehicle1['CO2 Emissions(g/km)']) * 100

            fuel_diff = vehicle2['Fuel Consumption Comb (L/100 km)'] - vehicle1['Fuel Consumption Comb (L/100 km)']
            fuel_diff_percent = (fuel_diff / vehicle1['Fuel Consumption Comb (L/100 km)']) * 100

            if co2_diff > 0:
                co2_message = f"Vehicle 2 emits {abs(co2_diff):.1f} g/km ({abs(co2_diff_percent):.1f}%) MORE CO2 than Vehicle 1"
                co2_color = "red"
            else:
                co2_message = f"Vehicle 2 emits {abs(co2_diff):.1f} g/km ({abs(co2_diff_percent):.1f}%) LESS CO2 than Vehicle 1"
                co2_color = "green"

            if fuel_diff > 0:
                fuel_message = f"Vehicle 2 uses {abs(fuel_diff):.1f} L/100 km ({abs(fuel_diff_percent):.1f}%) MORE fuel than Vehicle 1"
                fuel_color = "red"
            else:
                fuel_message = f"Vehicle 2 uses {abs(fuel_diff):.1f} L/100 km ({abs(fuel_diff_percent):.1f}%) LESS fuel than Vehicle 1"
                fuel_color = "green"

            st.markdown(f"""
            <div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; margin-bottom: 1rem;'>
                <p style='font-size: 1.1rem; color: {co2_color}; font-weight: bold;'>{co2_message}</p>
                <p style='font-size: 1.1rem; color: {fuel_color}; font-weight: bold;'>{fuel_message}</p>
            </div>
            """, unsafe_allow_html=True)

            # Visualization of the comparison
            fig, ax = plt.subplots(figsize=(10, 6))

            metrics = ['CO2 Emissions(g/km)', 'Fuel Consumption Comb (L/100 km)']
            vehicle1_values = [vehicle1[metric] for metric in metrics]
            vehicle2_values = [vehicle2[metric] for metric in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            ax.bar(x - width/2, vehicle1_values, width, label=f'{make1} {model1}')
            ax.bar(x + width/2, vehicle2_values, width, label=f'{make2} {model2}')

            ax.set_xticks(x)
            ax.set_xticklabels(["CO2 Emissions (g/km)", "Fuel Consumption (L/100 km)"])
            ax.legend()

            plt.tight_layout()
            st.pyplot(fig)

            # Annual impact calculation
            st.markdown("<h3>Annual Environmental Impact</h3>", unsafe_allow_html=True)

            annual_distance = st.slider("Annual Distance Driven (km)", 5000, 50000, 20000, 1000)

            # Calculate annual CO2 emissions difference (in kg)
            annual_co2_diff = (co2_diff * annual_distance) / 1000  # convert g to kg

            # Calculate annual fuel consumption difference (in liters)
            annual_fuel_diff = (fuel_diff * annual_distance) / 100  # convert L/100km to total liters

            if co2_diff > 0:
                impact_message = f"By choosing Vehicle 1 instead of Vehicle 2, you would save approximately {abs(annual_co2_diff):.1f} kg of CO2 emissions per year."
            else:
                impact_message = f"By choosing Vehicle 2 instead of Vehicle 1, you would save approximately {abs(annual_co2_diff):.1f} kg of CO2 emissions per year."

            st.markdown(f"""
            <div style='padding: 1rem; border-radius: 0.5rem; background-color: #d38434; margin-bottom: 1rem;'>
                <p style='font-size: 1.1rem;'>{impact_message}</p>
                <p>This is equivalent to approximately {abs(annual_co2_diff / 21):.1f} trees absorbing CO2 for one year.</p>
                <p>You would also save approximately {abs(annual_fuel_diff):.1f} liters of fuel per year.</p>
            </div>
            """, unsafe_allow_html=True)

    elif page == "About":
        st.markdown("<h2 class='sub-header'>ℹ️ About This App</h2>", unsafe_allow_html=True)

        st.markdown("""
        <p class='info-text'>This app analyzes CO2 emissions from vehicles in Canada and provides predictions based on vehicle characteristics. It was developed as part of a machine learning project to help users understand and reduce their environmental impact.</p>

        <h3 class='sub-header'>Data Source</h3>
        <p>The data used in this application comes from the Government of Canada's open data on vehicle fuel consumption and CO2 emissions. It includes details on various vehicles sold in Canada, including their fuel efficiency and carbon dioxide emissions.</p>

        <h3 class='sub-header'>How It Works</h3>
        <p>The app uses machine learning algorithms to predict CO2 emissions based on vehicle characteristics. The main factors affecting emissions are:</p>
        <ul>
            <li><b>Fuel Consumption:</b> Higher fuel consumption directly leads to higher CO2 emissions</li>
            <li><b>Engine Size:</b> Larger engines typically produce more CO2</li>
            <li><b>Fuel Type:</b> Different fuels have different carbon intensities</li>
            <li><b>Vehicle Class:</b> Larger vehicles generally emit more CO2</li>
        </ul>

        <h3 class='sub-header'>Models Used</h3>
        <p>The app uses several machine learning models to predict CO2 emissions:</p>
        <ul>
            <li><b>Linear Regression:</b> A simple model that assumes a linear relationship between features and CO2 emissions</li>
            <li><b>Random Forest:</b> An ensemble model that builds multiple decision trees and merges their predictions</li>
            <li><b>Gradient Boosting:</b> An advanced ensemble technique that builds trees sequentially, with each tree correcting errors from previous trees</li>
            <li><b>Simple Formula:</b> A straightforward calculation based on fuel consumption, which is the most direct predictor of CO2 emissions</li>
        </ul>

        <h3 class='sub-header'>Key Insights</h3>
        <p>From analyzing the data, we've learned that:</p>
        <ul>
            <li>Combined fuel consumption is the strongest predictor of CO2 emissions, with a near-linear relationship</li>
            <li>Larger engines and more cylinders generally lead to higher emissions</li>
            <li>Vehicle class has a significant impact - larger vehicles typically emit more CO2</li>
            <li>Fuel type affects emissions - ethanol and premium gasoline tend to have higher emissions per kilometer</li>
            <li>Hybrid and highly efficient vehicles can reduce emissions by over 50% compared to similar-sized vehicles with conventional powertrains</li>
        </ul>

        <h3 class='sub-header'>How to Reduce Vehicle CO2 Emissions</h3>
        <p>Based on our analysis, here are some ways to reduce your vehicle's carbon footprint:</p>
        <ul>
            <li>Choose vehicles with better fuel efficiency (lower L/100 km)</li>
            <li>Consider smaller engine sizes when possible</li>
            <li>Choose the right vehicle class for your needs - don't use a larger vehicle than necessary</li>
            <li>Consider alternative fuel types, especially electric or hybrid vehicles</li>
            <li>Maintain your vehicle properly to keep it running efficiently</li>
            <li>Practice eco-driving habits (smooth acceleration, maintaining steady speeds, etc.)</li>
        </ul>

        <h3 class='sub-header'>Project Details</h3>
        <p>This project demonstrates a complete end-to-end machine learning pipeline, including:</p>
        <ul>
            <li>Data cleaning and preprocessing</li>
            <li>Exploratory data analysis</li>
            <li>Feature engineering and selection</li>
            <li>Model training and evaluation</li>
            <li>Model deployment through an interactive web application</li>
        </ul>

        <p class='info-text'>Created by: Data Science Enthusiast</p>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
