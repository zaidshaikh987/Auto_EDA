import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import plotly.express as px
from streamlit_option_menu import option_menu
import data_analysis_functions as function
import data_preprocessing_function as preprocessing_function
import home_page
import base64
import automl_functions as automl



# # page config sets the text and icon that we see on the tab
st.set_page_config(page_icon="✨", page_title="AutoEDA")

# --- JAW-DROPPING CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;900&display=swap');
    html, body, [class*="stApp"]  {
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
        background: linear-gradient(120deg, rgba(99,102,241,0.13) 0%, rgba(16,185,129,0.11) 100%), url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1500&q=80') center/cover fixed no-repeat !important;
        min-height: 100vh;
    }
    .stApp {
        background: transparent !important;
    }
    /* Glassmorphism card effect */
    .block-container {
        background: rgba(255,255,255,0.75) !important;
        box-shadow: 0 8px 32px 0 rgba(31,38,135,0.18);
        backdrop-filter: blur(8px) !important;
        border-radius: 22px !important;
        padding: 2.5em 2em 2em 2em !important;
        margin-top: 2.5em !important;
        margin-bottom: 2.5em !important;
        animation: fadein 1.2s;
    }
    @keyframes fadein {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #10b981 100%) !important;
        color: #fff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.7em 1.5em !important;
        font-size: 1.1em !important;
        font-weight: 600;
        box-shadow: 0 4px 18px rgba(99,102,241,0.13);
        transition: background 0.18s, transform 0.18s, box-shadow 0.18s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #4338ca 0%, #059669 100%) !important;
        color: #fff !important;
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 8px 32px rgba(99,102,241,0.18);
    }
    .stRadio > div { gap: 1.5em !important; }
    .stExpanderHeader { font-size: 1.1em !important; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1em; font-weight: 700; color: #4338ca; }
    .stMetric { background: rgba(99,102,241,0.09); border-radius: 12px; padding: 1em 1.1em; box-shadow: 0 2px 8px rgba(99,102,241,0.06); }
    h1, h2, h3, h4, h5, h6 { color: #4338ca !important; font-weight: 900 !important; letter-spacing: -1px; }
    .stDataFrame { background: #fff !important; border-radius: 12px !important; }
    .stSidebar { background: rgba(255,255,255,0.8) !important; box-shadow: 2px 0 16px 0 rgba(99,102,241,0.09); }
    .stSidebar > div { padding-top: 2em !important; }
    .stMarkdown { font-size: 1.08em; }
    .stAlert { border-radius: 12px !important; }
    .stTextInput > div > input {
        border-radius: 8px !important;
        border: 1.5px solid #6366f1 !important;
        background: #f4f6fc !important;
    }
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1.5px solid #6366f1 !important;
        background: #f4f6fc !important;
    }
    </style>
""", unsafe_allow_html=True)

# Hide Streamlit's default menu and footer
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Create a Streamlit sidebar
st.sidebar.title("AutoEDA: Automated Exploratory Data Analysis and Processing")

# Create the introduction section
st.title("✨ Welcome to AutoEDA")
st.markdown("<div style='font-size:1.2em; color:#6366f1; margin-bottom:1.5em;'>Unleash the Power of Automated Data Science</div>", unsafe_allow_html=True)

# Initialize session state for AutoML if not exists
if 'automl_results' not in st.session_state:
    st.session_state.automl_results = None

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=['Home', 'Data Exploration', 'Data Preprocessing', 'AutoML'],
    icons=['house-heart', 'bar-chart-fill', 'hammer', 'robot'],
    orientation='horizontal'
)

if selected == 'Home':
    home_page.show_home_page()


# Create a button in the sidebar to upload CSV
uploaded_file = st.sidebar.file_uploader("Upload Your CSV File Here", type=["csv","xls"])
use_example_data = st.sidebar.checkbox("Use Example Titanic Dataset", value=False)

# ADDING LINKS TO MY PROFILES 
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")

st.sidebar.markdown("### **Connect with Me**")
# Create columns in the sidebar for LinkedIn and GitHub icons
col1, col2 = st.sidebar.columns(2)

# Define the width and height for the icons (adjust as needed)
icon_width = 80
icon_height = 80

if uploaded_file:
    df = function.load_data(uploaded_file)


    # get a copy of original df from the session state or create a new one. this is for preprocessing purposes
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df.copy()

    

# Create a checkbox in the sidebar to choose between the example dataset and uploaded dataset

elif use_example_data:
    # Load the example dataset
    df = function.load_data(file="example_dataset/titanic.csv")

    # Set st.session_state.new_df to the example dataset for data preprocessing
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df
   


# TODO: Some issue related to session_state. When we upload a new dataset, it does not reflect changes in the data preprocessing tab as we are using session state.
# and the data is defined only once. need to solve this issue.
# Temporary solution is to reload the page and upload a new dataset


# Display the dataset preview or any other content here
if uploaded_file is None and selected!='Home' and not use_example_data:
    # st.subheader("Welcome to DataExplora!")
    st.markdown("#### Use the sidebar to upload a CSV file or use the provided example dataset and explore your data.")
    
else:
    
    if selected=='Data Exploration':

        tab1, tab2, tab3 = st.tabs([
            '📊 Dataset Overview :clipboard', 
            '🔎 Data Exploration and Visualization',
            '🤖 AutoML (Beta)'
        ])
        num_columns, cat_columns = function.categorical_numerical(df)
        
        
        with tab1: # DATASET OVERVIEW TAB
            st.subheader("1. Dataset Preview")
            st.markdown("This section provides an overview of your dataset. You can select the number of rows to display and view the dataset's structure.")
            function.display_dataset_overview(df,cat_columns,num_columns)


            st.subheader("3. Missing Values")
            function.display_missing_values(df)
            
            st.subheader("4. Data Statistics and Visualization")
            function.display_statistics_visualization(df,cat_columns,num_columns)

            st.subheader("5. Data Types")
            function.display_data_types(df)

            st.subheader("Search for a specific column or datatype")
            function.search_column(df)

        with tab2: 

            function.display_individual_feature_distribution(df,num_columns)

            st.subheader("Scatter Plot")
            function.display_scatter_plot_of_two_numeric_features(df,num_columns)


            if len(cat_columns)!=0:
                st.subheader("Categorical Variable Analysis")
                function.categorical_variable_analysis(df,cat_columns)
            else:
                st.info("The dataset does not have any categorical columns")


            st.subheader("Feature Exploration of Numerical Variables")
            if len(num_columns)!=0:
                function.feature_exploration_numerical_variables(df,num_columns)

            else:
                st.warning("The dataset does not contain any numerical variables")

            # Create a bar graph to get relationship between categorical variable and numerical variable
            st.subheader("Categorical and Numerical Variable Analysis")
            if len(num_columns)!=0 and len(cat_columns)!=0:
                function.categorical_numerical_variable_analysis(df,cat_columns,num_columns)
                
            else:
                st.warning("The dataset does not have any numerical variables. Hence Cannot Perform Categorical and Numerical Variable Analysis")
        
    # DATA PREPROCESSING  
    if selected=='Data Preprocessing':
        # st.header("🛠️ Data Preprocessing")
        revert = st.button("Revert to Original Dataset",key="revert_button")

        if revert:
            st.session_state.new_df = df.copy()

        # REMOVING UNWANTED COLUMNS
        st.subheader("Remove Unwanted Columns")
        columns_to_remove = st.multiselect(label='Select Columns to Remove',options=st.session_state.new_df.columns)

        if st.button("Remove Selected Columns"):
            if columns_to_remove:
                st.session_state.new_df = preprocessing_function.remove_selected_columns(st.session_state.new_df,columns_to_remove)
                st.success("Selected Columns Removed Sucessfully")
                
        st.dataframe(st.session_state.new_df)
       

       # Handle missing values in the dataset
        st.subheader("Handle Missing Data")
        missing_count = st.session_state.new_df.isnull().sum()

        if missing_count.any():

            selected_missing_option = st.selectbox(
                "Select how to handle missing data:",
                ["Remove Rows in Selected Columns", "Fill Missing Data in Selected Columns (Numerical Only)"]
            )

            if selected_missing_option == "Remove Rows in Selected Columns":
                columns_to_remove_missing = st.multiselect("Select columns to remove rows with missing data", options=st.session_state.new_df.columns)
                if st.button("Remove Rows with Missing Data"):
                    st.session_state.new_df = preprocessing_function.remove_rows_with_missing_data(st.session_state.new_df, columns_to_remove_missing)
                    st.success("Rows with missing data removed successfully.")

            elif selected_missing_option == "Fill Missing Data in Selected Columns (Numerical Only)":
                numerical_columns_to_fill = st.multiselect("Select numerical columns to fill missing data", options=st.session_state.new_df.select_dtypes(include=['number']).columns)
                fill_method = st.selectbox("Select fill method:", ["mean", "median", "mode"])
                if st.button("Fill Missing Data"):
                    if numerical_columns_to_fill:
                        st.session_state.new_df = preprocessing_function.fill_missing_data(st.session_state.new_df, numerical_columns_to_fill, fill_method)
                        st.success(f"Missing data in numerical columns filled with {fill_method} successfully.")

                    else:
                        st.warning("Please select a column to fill in the missing data")

            function.display_missing_values(st.session_state.new_df)

        else:
            st.info("The dataset does not contain any missing values")

        encoding_tooltip = '''**One-Hot encoding** converts categories into binary values (0 or 1). It's like creating checkboxes for each category. This makes it possible for computers to work with categorical data.
        **Label encoding** assigns unique numbers to categories. It's like giving each category a name (e.g., Red, Green, Blue becomes 1, 2, 3). This helps computers understand and work with categories.
        '''
        st.subheader("Encode Categorical Data")

        new_df_categorical_columns = st.session_state.new_df.select_dtypes(include=['object']).columns

        if not new_df_categorical_columns.empty:
            select_categorical_columns = st.multiselect("Select Columns to perform encoding",new_df_categorical_columns)

            #choose the encoding method
            encoding_method = st.selectbox("Select Encoding Method:",['One Hot Encoding','Label Encoding'],help=encoding_tooltip)
    

            if st.button("Apply Encoding"):
                if encoding_method=="One Hot Encoding":
                    st.session_state.new_df = preprocessing_function.one_hot_encode(st.session_state.new_df,select_categorical_columns)
                    st.success("One-Hot Encoding Applied Sucessfully")

                if encoding_method=="Label Encoding":
                    st.session_state.new_df = preprocessing_function.label_encode(st.session_state.new_df,select_categorical_columns)
                    st.success("Label Encoding Applied Sucessfully")


            st.dataframe(st.session_state.new_df)
        else:
            st.info("The dataset does not contain any categorical columns")

        feature_scaling_tooltip='''**Standardization** scales your data to have a mean of 0 and a standard deviation of 1. It helps in comparing variables with different units. Think of it like making all values fit on the same measurement scale.
        **Min-Max scaling** transforms your data to fall between 0 and 1. It's like squeezing data into a specific range. This makes it easier to compare data points that vary widely.'''


        st.subheader("Feature Scaling")
        new_df_numerical_columns = st.session_state.new_df.select_dtypes(include=['number']).columns
        selected_columns = st.multiselect("Select Numerical Columns to Scale", new_df_numerical_columns)

        scaling_method = st.selectbox("Select Scaling Method:", ['Standardization', 'Min-Max Scaling'],help=feature_scaling_tooltip)

        if st.button("Apply Scaling"):
            if selected_columns:
                if scaling_method == "Standardization":
                    st.session_state.new_df = preprocessing_function.standard_scale(st.session_state.new_df, selected_columns)
                    st.success("Standardization Applied Successfully.")
                elif scaling_method == "Min-Max Scaling":
                    st.session_state.new_df = preprocessing_function.min_max_scale(st.session_state.new_df, selected_columns)
                    st.success("Min-Max Scaling Applied Successfully.")
            else:
                st.warning("Please select numerical columns to scale.")

        st.dataframe(st.session_state.new_df)

        st.subheader("Identify and Handle Outliers")

        
        # Select numeric column for handling outliers
        


        selected_numeric_column = st.selectbox("Select Numeric Column for Outlier Handling:", new_df_numerical_columns)
        st.write(selected_numeric_column)

        
        # Display outliers in a box plot
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=st.session_state.new_df, x=selected_numeric_column)
        st.pyplot(fig)


        outliers = preprocessing_function.detect_outliers_zscore(st.session_state.new_df, selected_numeric_column)
        if outliers:
            st.warning("Detected Outliers:")
            st.write(outliers)
        else:
            st.info("No outliers detected using IQR.")


        # Choose handling method
        outlier_handling_method = st.selectbox("Select Outlier Handling Method:", ["Remove Outliers", "Transform Outliers"])

        # Perform outlier handling based on the method chosen
        if st.button("Apply Outlier Handling"):
            if outlier_handling_method == "Remove Outliers":
               
                st.session_state.new_df = preprocessing_function.remove_outliers(st.session_state.new_df, selected_numeric_column,outliers)
                st.success("Outliers removed successfully.")

            elif outlier_handling_method == "Transform Outliers":
                # Provide options for transforming outliers (e.g., capping, log transformation)
                # Update st.session_state.new_df after transforming outliers
                st.session_state.new_df = preprocessing_function.transform_outliers(st.session_state.new_df, selected_numeric_column,outliers)
                st.success("Outliers transformed successfully.")

        # Show the updated dataset
        st.dataframe(st.session_state.new_df)
        
        if st.session_state.new_df is not None:
            # Convert the DataFrame to CSV
            csv = st.session_state.new_df.to_csv(index=False)
            # Encode as base64
            b64 = base64.b64encode(csv.encode()).decode()
            # Create a download link
            href = f'data:file/csv;base64,{b64}'
            # Display a download button
            st.markdown(f'<a href="{href}" download="preprocessed_data.csv"><button>Download Preprocessed Data</button></a>', unsafe_allow_html=True)
        else:
            st.warning("No preprocessed data available to download.")
    
    # AutoML Tab
    elif selected == 'AutoML':
        st.header("🤖 AutoML - Automated Machine Learning")
        
        # Check if we have data loaded
        if 'new_df' not in st.session_state or st.session_state.new_df.empty:
            st.warning("Please load and preprocess your data in the Data Preprocessing tab first.")
            st.stop()
            
        df = st.session_state.new_df
        
        # Show data summary
        with st.expander("🔍 View Dataset Summary"):
            st.write(f"Dataset Shape: {df.shape}")
            st.write("First 5 rows:")
            st.dataframe(df.head())
        
        # --- Smart Auto-Detection and UI/UX Enhancements ---
        st.markdown("""
            <style>
            .recommended-badge {
                display: inline-block;
                background: linear-gradient(90deg,#10b981 0%,#6366f1 100%);
                color: #fff;
                font-size: 0.95em;
                font-weight: 600;
                border-radius: 8px;
                padding: 0.18em 0.7em;
                margin-left: 0.4em;
                vertical-align: middle;
            }
            .fab-export {
                position: fixed;
                bottom: 32px;
                right: 32px;
                z-index: 9999;
                background: linear-gradient(90deg,#6366f1 0%,#10b981 100%);
                color: #fff;
                font-size: 1.5em;
                border: none;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                box-shadow: 0 4px 18px rgba(99,102,241,0.19);
                transition: background 0.18s, box-shadow 0.18s, transform 0.18s;
            }
            .fab-export:hover {
                background: linear-gradient(90deg,#4338ca 0%,#059669 100%);
                transform: scale(1.08);
                box-shadow: 0 8px 32px rgba(99,102,241,0.23);
            }
            </style>
        """, unsafe_allow_html=True)

        # Smart auto-detection
        # 1. Problem type
        col_types = df.dtypes
        unique_counts = df.nunique()
        likely_targets = [col for col in df.columns if unique_counts[col] <= 10 or col_types[col] == 'object']
        suggested_problem_type = "Classification" if likely_targets else "Regression"
        # 2. Target
        suggested_target = likely_targets[-1] if likely_targets else df.columns[-1]
        # 3. Features
        suggested_features = [col for col in df.columns if col != suggested_target]

        # Problem type selection
        st.subheader("Task Type")
        problem_type = st.radio(
            "Select Problem Type:",
            ["Classification", "Regression"],
            horizontal=True,
            index=0 if suggested_problem_type=="Classification" else 1,
            help="We recommend 'Classification' if your target is categorical or has few unique values; otherwise, use 'Regression'."
        )
        st.markdown(f'<span class="recommended-badge">Recommended: {suggested_problem_type}</span>', unsafe_allow_html=True)

        # Target variable selection
        st.subheader("🔑 Select Target Variable")
        st.markdown("<span style='color:#6366f1;font-size:0.99em;'>We recommend the column with fewest unique values for classification, or the last column for regression.</span>", unsafe_allow_html=True)
        target_col = st.selectbox(
            f"Select Target Variable:",
            options=df.columns,
            index=list(df.columns).index(suggested_target),
            key="target_col",
            help="Target variable is what you want to predict."
        )
        st.markdown(f'<span class="recommended-badge">Recommended: {suggested_target}</span>', unsafe_allow_html=True)

        # Feature selection
        st.subheader("📊 Select Features")
        st.markdown("<span style='color:#6366f1;font-size:0.99em;'>We recommend using all columns except the target as features.</span>", unsafe_allow_html=True)
        feature_cols = st.multiselect(
            "Select Features (leave empty to use all):",
            options=[col for col in df.columns if col != target_col],
            default=suggested_features,
            key="feature_cols",
            help="Features are the columns used to predict the target."
        )
        st.markdown(f'<span class="recommended-badge">Recommended: All except {suggested_target}</span>', unsafe_allow_html=True)

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size (%)", 10, 50, 20, help="Fraction of data used for testing.") / 100.0
                random_state = st.number_input("Random State", 0, 100, 42, help="Seed for reproducibility.")
            with col2:
                time_budget = st.slider("Time Budget (seconds)", 30, 600, 60, 30, help="How long to search for the best model.")

        # Floating export button (appears after results)
        if st.session_state.get('automl_results'):
            st.markdown('<button class="fab-export" title="Export All Results as ZIP">⬇️</button>', unsafe_allow_html=True)
            # (In a real app, hook this up to a download action)

        # Run AutoML button
        if st.button("🚀 Run AutoML", type="primary", use_container_width=True):
            if not feature_cols:
                st.error("❌ Please select at least one feature.")
                st.stop()
                
            with st.spinner("🤖 Training models... This may take a few minutes..."):
                try:
                    # Prepare data
                    data_prep = automl.prepare_data(
                        df, 
                        target_col, 
                        problem_type.lower(),
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    # Train model
                    training_result = automl.train_automl(
                        data_prep['X_train'],
                        data_prep['y_train'],
                        problem_type.lower(),
                        time_budget=time_budget
                    )
                    
                    # Evaluate model
                    evaluation = automl.evaluate_model(
                        training_result['model'],
                        data_prep['X_test'],
                        data_prep['y_test'],
                        problem_type.lower(),
                        data_prep.get('label_mapping')
                    )
                    
                    # Save results to session state
                    st.session_state.automl_results = {
                        'problem_type': problem_type,
                        'training_result': training_result,
                        'evaluation': evaluation,
                        'feature_names': data_prep['feature_names'],
                        'target_col': target_col,
                        'feature_cols': feature_cols
                    }
                    
                    st.success("✅ AutoML completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error in AutoML: {str(e)}")
                    st.exception(e)  # Show full error for debugging
        
        # Display results if available
        if st.session_state.automl_results and 'automl_results' in st.session_state:
            results = st.session_state.automl_results
            st.markdown("---")
            st.subheader("📊 Results")
            
            # Model info
            col1, col2 = st.columns(2)

            # Metrics
            st.subheader("📈 Performance Metrics")
            metrics = results['evaluation']['metrics']
            metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
            st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}), use_container_width=True)

            # --- Download predictions/results as CSV ---
            import io
            import base64
            y_true = results['evaluation']['y_true']
            y_pred = results['evaluation']['y_pred']
            pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            csv = pred_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="automl_predictions.csv"><button>Download Predictions as CSV</button></a>'
            st.markdown(href, unsafe_allow_html=True)

            # --- Model Leaderboard (FLAML history) ---
            st.subheader("🏅 Model Leaderboard")
            leaderboard = None
            try:
                automl_obj = results['training_result']['model']
                if hasattr(automl_obj, 'best_history') and automl_obj.best_history:
                    leaderboard = pd.DataFrame(automl_obj.best_history)
                elif hasattr(automl_obj, 'model_history') and automl_obj.model_history:
                    leaderboard = pd.DataFrame(automl_obj.model_history)
                elif hasattr(automl_obj, 'search_space') and hasattr(automl_obj, 'best_config'):
                    # Fallback: show best config
                    leaderboard = pd.DataFrame([automl_obj.best_config])
                if leaderboard is not None and not leaderboard.empty:
                    st.dataframe(leaderboard)
                else:
                    st.info("Leaderboard not available for this run.")
            except Exception as e:
                st.info(f"Leaderboard not available: {str(e)}")

            # --- SHAP Explainability ---
            st.subheader("🔬 Model Explainability (SHAP)")
            try:
                import shap
                best_model = results['training_result']['model'].model.estimator
                # Use a small sample for speed; convert to DataFrame with feature names for SHAP
                X_test_arr = results['evaluation']['X_test'][:100]
                if isinstance(X_test_arr, pd.DataFrame):
                    X_sample = X_test_arr
                else:
                    X_sample = pd.DataFrame(X_test_arr, columns=results['feature_names'])
                explainer = None
                shap_values = None
                # Try TreeExplainer first
                try:
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_sample)
                except Exception:
                    # Try KernelExplainer as fallback
                    try:
                        explainer = shap.KernelExplainer(best_model.predict, X_sample)
                        shap_values = explainer.shap_values(X_sample)
                    except Exception as e:
                        st.info(f"SHAP not supported for this model: {str(e)}")
                if shap_values is not None and explainer is not None:
                    import matplotlib.pyplot as plt
                    import streamlit as st
                    import io
                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot(fig)
            except Exception as e:
                st.info(f"SHAP explainability not available: {str(e)}")

            # --- Show top errors/misclassifications ---
            st.subheader("🔍 Error Analysis")
            N = 10
            if results['problem_type'].lower() == 'classification':
                misclassified = pred_df[pred_df['y_true'] != pred_df['y_pred']]
                st.write(f"Top {N} Misclassified Samples:")
                st.dataframe(misclassified.head(N))
            else:
                pred_df['abs_error'] = (pred_df['y_true'] - pred_df['y_pred']).abs()
                st.write(f"Top {N} Largest Errors:")
                st.dataframe(pred_df.sort_values('abs_error', ascending=False).head(N))
            
            if results['problem_type'].lower() == 'classification':
                col1, col2, col3 = st.columns(3)

                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                fig, _ = automl.plot_confusion_matrix(
                    results['evaluation']['y_true'],
                    results['evaluation']['predictions'],
                    results['evaluation'].get('class_names')
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Regression
                col1, col2, col3 = st.columns(3)

                
                # Regression plot
                st.subheader("Actual vs Predicted")
                fig, _, _ = automl.plot_regression_results(
                    results['evaluation']['y_true'],
                    results['evaluation']['predictions']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            st.subheader("🔍 Feature Importance")
            fig, _ = automl.plot_feature_importance(
                results['training_result']['model'].model.estimator,
                results['feature_names'],
                top_n=15
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download model button
            import pickle
            model_bytes = pickle.dumps(results['training_result']['model'])
            st.download_button(
                label="💾 Download AutoML Object",
                data=model_bytes,
                file_name="flaml_automl.pkl",
                mime="application/octet-stream"
            )