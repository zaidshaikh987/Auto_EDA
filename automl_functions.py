import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set plot style
sns.set_style('whitegrid')
# plt.style.use('seaborn-v0_8')  # Optional: use seaborn style if available, otherwise rely on seaborn.set_style above

def prepare_data(df, target_col, problem_type, test_size=0.2, random_state=42):
    """
    Prepare data for AutoML training and testing
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        problem_type: 'classification' or 'regression'
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing prepared data and metadata
    """
    try:
        # Create a copy of the data
        data = df.copy()
        
        # Handle target variable
        if problem_type == 'classification':
            # Drop rare classes with only 1 sample
            class_counts = data[target_col].value_counts()
            rare_classes = class_counts[class_counts == 1].index.tolist()
            if rare_classes:
                n_rare = sum(data[target_col].isin(rare_classes))
                st.warning(f"Automatically removed {n_rare} sample(s) from rare classes: {rare_classes}")
                data = data[~data[target_col].isin(rare_classes)]
            # For classification, encode the target if it's categorical
            if data[target_col].dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(data[target_col])
                label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
                inverse_mapping = {v: k for k, v in label_mapping.items()}
            else:
                y = data[target_col].values
                label_mapping = None
                inverse_mapping = None
        else:  # regression
            y = data[target_col].values
            label_mapping = None
            inverse_mapping = None
        
        # Prepare features (one-hot encode categorical variables)
        X = data.drop(columns=[target_col])
        
        # Get numerical and categorical columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Clean column names before one-hot encoding
        def clean_column_name(name):
            if not isinstance(name, str):
                name = str(name)
            # Replace special characters with underscores
            for char in [' ', '<', '>', '[', ']', '{', '}', '(', ')', ',', "'", '\\', '/', ':', ';', '!', '?', '&', '|', '^', '~', '`', '@', '#', '$', '%', '*', '=', '+', '\"']:
                name = name.replace(char, '_')
            # Remove multiple consecutive underscores
            while '__' in name:
                name = name.replace('__', '_')
            # Remove leading/trailing underscores
            name = name.strip('_')
            # Ensure the name is not empty and starts with a letter
            if not name:
                name = 'feature'
            if name[0].isdigit():
                name = 'f_' + name
            return name
        
        # Clean all column names
        X.columns = [clean_column_name(col) for col in X.columns]
        
        # One-hot encode categorical variables
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
            # Clean the new column names from one-hot encoding
            X.columns = [clean_column_name(col) for col in X.columns]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if problem_type == 'classification' else None
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'label_mapping': label_mapping,
            'inverse_mapping': inverse_mapping,
            'num_cols': num_cols,
            'cat_cols': cat_cols
        }
        
    except Exception as e:
        raise Exception(f"Error preparing data: {str(e)}")

def train_automl(X_train, y_train, problem_type, time_budget=60, **kwargs):
    """
    Train an AutoML model using FLAML
    
    Args:
        X_train: Training features
        y_train: Training target
        problem_type: 'classification' or 'regression'
        time_budget: Time budget in seconds
        **kwargs: Additional arguments to pass to FLAML
        
    Returns:
        Trained model and training logs
    """
    try:
        # Initialize AutoML
        automl = AutoML()
        
        # Set up parameters based on problem type
        if problem_type == 'classification':
            metric = 'accuracy'
            task = 'classification'
            # Add class weights if specified
            if 'class_weight' in kwargs:
                automl.class_weight = kwargs.pop('class_weight')
        else:  # regression
            metric = 'r2'
            task = 'regression'
        
        # Set default parameters if not provided
        params = {
            'time_budget': time_budget,
            'metric': metric,
            'task': task,
            'estimator_list': ['lgbm','xgboost','rf','extra_tree','xgb_limitdepth','lrl2','lrl1','catboost','svc','kneighbor','histgb'],
            'n_jobs': -1,
            'eval_method': 'cv',
            'n_splits': 5,
            'verbose': 0,
            **kwargs  # Allow overriding defaults
        }
        
        # Train the model
        start_time = time.time()
        automl.fit(X_train=X_train, y_train=y_train, **params)
        training_time = time.time() - start_time
        
        return {
            'model': automl,
            'training_time': training_time,
            'best_estimator': automl.model.estimator.__class__.__name__,
            'best_params': automl.best_config
        }
        
    except Exception as e:
        raise Exception(f"Error in AutoML training: {str(e)}")

def evaluate_model(model, X_test, y_test, problem_type, label_mapping=None):
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        problem_type: 'classification' or 'regression'
        label_mapping: Mapping of encoded labels to original labels
        
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get predicted probabilities for classification
        y_prob = None
        if hasattr(model, 'predict_proba') and problem_type == 'classification':
            y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {}
        
        if problem_type == 'classification':
            # Robust: Label-encode all labels to ensure numeric metrics work
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_true_raw = np.array(y_test)
            y_pred_raw = np.array(y_pred)
            le.fit(np.concatenate([y_true_raw, y_pred_raw]))
            y_test_enc = le.transform(y_true_raw)
            y_pred_enc = le.transform(y_pred_raw)
            class_names_decoded = le.classes_.astype(str)

            # Classification metrics (use encoded labels)
            metrics['accuracy'] = accuracy_score(y_test_enc, y_pred_enc)
            metrics['f1_weighted'] = f1_score(y_test_enc, y_pred_enc, average='weighted')
            metrics['precision_weighted'] = precision_score(y_test_enc, y_pred_enc, average='weighted')
            metrics['recall_weighted'] = recall_score(y_test_enc, y_pred_enc, average='weighted')

            # Additional metrics for binary classification
            if len(le.classes_) == 2:
                metrics['f1_binary'] = f1_score(y_test_enc, y_pred_enc, average='binary')
                metrics['precision_binary'] = precision_score(y_test_enc, y_pred_enc, average='binary')
                metrics['recall_binary'] = recall_score(y_test_enc, y_pred_enc, average='binary')
                if y_prob is not None and y_prob.ndim == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test_enc, y_prob[:, 1])

            # Classification report with decoded names
            report = classification_report(
                y_test_enc,
                y_pred_enc,
                target_names=class_names_decoded,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()
            
            # Replace encoded predictions with decoded for return
            y_pred = le.inverse_transform(y_pred_enc)
            y_test = le.inverse_transform(y_test_enc)
            
            # Convert report to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            
        else:  # regression
            # Fully robust: Convert to Series, coerce to numeric, drop NaNs, match shape
            import streamlit as st
            y_test_ser = pd.Series(y_test).reset_index(drop=True)
            y_pred_ser = pd.Series(y_pred).reset_index(drop=True)
            y_test_num = pd.to_numeric(y_test_ser, errors='coerce')
            y_pred_num = pd.to_numeric(y_pred_ser, errors='coerce')
            valid_mask = (~pd.isna(y_test_num)) & (~pd.isna(y_pred_num))
            dropped = (~valid_mask).sum()
            if dropped > 0:
                st.warning(f"Dropped {dropped} samples with non-numeric values from regression evaluation. Check your target and predictions for mixed types.")
            y_test_num = y_test_num[valid_mask]
            y_pred_num = y_pred_num[valid_mask]
            # Ensure same length
            min_len = min(len(y_test_num), len(y_pred_num))
            y_test_num = y_test_num.iloc[:min_len]
            y_pred_num = y_pred_num.iloc[:min_len]
            # Final type check
            if not np.issubdtype(y_test_num.dtype, np.number) or not np.issubdtype(y_pred_num.dtype, np.number):
                st.error(f"Regression evaluation failed: Non-numeric types remain after coercion. y_test dtype: {y_test_num.dtype}, y_pred dtype: {y_pred_num.dtype}")
                raise Exception("Non-numeric types after coercion.")
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_test_num, y_pred_num)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test_num, y_pred_num)
            metrics['r2'] = r2_score(y_test_num, y_pred_num)
            report_df = None
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'y_true': y_test,
            'y_pred': y_pred,
            'probabilities': y_prob,
            'report': report_df,
            'X_test': X_test
        }
        
    except Exception as e:
        raise Exception(f"Error evaluating model: {str(e)}")

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance from a trained model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        
    Returns:
        tuple: (Plotly figure object, error_message)
    """
    try:
        # Initialize default empty figure
        fig = go.Figure()
        
        # Handle different model types
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if len(importances.shape) > 1:  # For multi-class
                importances = np.mean(importances, axis=0)
        else:
            # If no importance scores available, create an empty figure with a message
            fig.add_annotation(
                text="Feature importance not available for this model",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=200
            )
            return fig, "Feature importance not available for this model"
        
        # Ensure we have valid importances and feature names
        if len(importances) == 0 or len(feature_names) == 0:
            fig.add_annotation(
                text="No feature importance data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=200
            )
            return fig, "No feature importance data available"
        
        # Make sure we don't try to show more features than we have
        top_n = min(top_n, len(importances))
        
        # Create a DataFrame for plotting
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(importances)],  # Ensure lengths match
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Create the plot
        fig = px.bar(
            feature_importance, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Most Important Features',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
            text_auto='.3f'
        )
        
        # Update layout for better readability
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, 30 * top_n),  # Dynamic height based on number of features
            margin=dict(l=150, r=20, t=50, b=50),
            hovermode='closest'
        )
        
        # Add value labels to bars
        fig.update_traces(
            textposition='outside',
            textfont_size=12,
            textangle=0,
            texttemplate='%{text:.3f}'
        )
        
        return fig, None
        
    except Exception as e:
        # Create a minimal figure with error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error plotting feature importance: {str(e)}",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color="red")
        )
        error_fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=200
        )
        return error_fig, f"Error plotting feature importance: {str(e)}"

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot a confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        
    Returns:
        Plotly figure object
    """
    try:
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class names if not provided
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(y_true)))]
        
        # Create the heatmap
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=class_names,
            y=class_names,
            title="Confusion Matrix",
            color_continuous_scale='Blues'
        )
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig.add_annotation(
                    x=class_names[j],
                    y=class_names[i],
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            coloraxis_colorbar=dict(title="Count")
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error plotting confusion matrix: {str(e)}"

def plot_regression_results(y_true, y_pred):
    """
    Plot regression results (actual vs predicted)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Plotly figure object
    """
    try:
        # Create a DataFrame for plotting
        results = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Error': y_true - y_pred
        })
        
        # Create the scatter plot
        fig = px.scatter(
            results,
            x='Actual',
            y='Predicted',
            title='Actual vs Predicted Values',
            trendline='lowess',
            opacity=0.7
        )
        
        # Add line of perfect prediction
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        
        # Add error distribution
        fig.update_layout(
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            showlegend=True
        )
        
        # Create error distribution plot
        error_fig = px.histogram(
            results,
            x='Error',
            nbins=50,
            title='Prediction Error Distribution',
            labels={'Error': 'Prediction Error (Actual - Predicted)'}
        )
        
        error_fig.update_layout(
            xaxis_title="Prediction Error",
            yaxis_title="Count"
        )
        
        return fig, error_fig, None
        
    except Exception as e:
        return None, None, f"Error plotting regression results: {str(e)}"

def plot_roc_curve(y_true, y_prob, class_names=None):
    """
    Plot ROC curve for binary or multiclass classification
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names (optional)
        
    Returns:
        Plotly figure object
    """
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Convert y_true to one-hot encoding if needed
        if len(y_prob.shape) == 1:
            y_prob = np.column_stack([1-y_prob, y_prob])
        
        n_classes = y_prob.shape[1]
        
        # Get class names if not provided
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            label_binarize(y_true, classes=np.unique(y_true)).ravel(),
            y_prob.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Create the figure
        fig = go.Figure()
        
        # Add ROC curve for each class
        for i in range(n_classes):
            fig.add_trace(go.Scatter(
                x=fpr[i],
                y=tpr[i],
                mode='lines',
                name=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
            ))
        
        # Add micro-average ROC curve
        fig.add_trace(go.Scatter(
            x=fpr["micro"],
            y=tpr["micro"],
            mode='lines',
            line=dict(dash='dash'),
            name=f'Micro-avg (AUC = {roc_auc["micro"]:.2f})'
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend_title='Classes',
            hovermode='closest',
            height=600
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error plotting ROC curve: {str(e)}"

def display_automl_ui(df):
    """
    Main function to display AutoML interface and results in Streamlit
    
    Args:
        df: Input DataFrame
    """
    st.subheader("🔍 AutoML - Automated Machine Learning")
    
    if df is None or df.empty:
        st.warning("Please load a dataset first")
        return
    
    # Determine problem type
    problem_type = st.radio(
        "Select Problem Type",
        ["Classification", "Regression"],
        horizontal=True,
        key="automl_problem_type"
    ).lower()
    
    # Select target variable
    target_col = st.selectbox(
        "Select Target Variable", 
        df.columns,
        key="automl_target"
    )
    
    # Get feature types
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from features
    if target_col in num_cols:
        num_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    # Feature selection
    st.subheader("Feature Selection")
    use_all_features = st.checkbox(
        "Use all features", 
        value=True, 
        key="use_all_features"
    )
    
    selected_features = []
    if not use_all_features:
        col1, col2 = st.columns(2)
        with col1:
            selected_num = st.multiselect(
                "Select numerical features", 
                num_cols, 
                default=num_cols[:min(5, len(num_cols))],
                key="selected_num"
            )
        with col2:
            selected_cat = st.multiselect(
                "Select categorical features", 
                cat_cols, 
                default=cat_cols,
                key="selected_cat"
            )
        selected_features = selected_num + selected_cat
    else:
        selected_features = num_cols + cat_cols
    
    # Add target back to the dataset
    selected_features.append(target_col)
    df_selected = df[selected_features].copy()
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                key="test_size"
            ) / 100.0
            
            random_state = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=1000,
                value=42,
                step=1,
                key="random_state"
            )
        
        with col2:
            time_budget = st.slider(
                "Time Budget (seconds)",
                min_value=10,
                max_value=600,
                value=60,
                step=10,
                key="time_budget",
                help="Maximum time for AutoML to find the best model"
            )
    
    # Run AutoML button
    if st.button("🚀 Run AutoML", key="run_automl"):
        with st.spinner("🧠 Preparing data and training models (this may take a few minutes)..."):
            try:
                # Prepare data
                data_prep = prepare_data(
                    df_selected, 
                    target_col, 
                    problem_type,
                    test_size=test_size,
                    random_state=random_state
                )
                
                # Train model
                training_result = train_automl(
                    data_prep['X_train'],
                    data_prep['y_train'],
                    problem_type,
                    time_budget=time_budget
                )
                
                # Evaluate model
                evaluation = evaluate_model(
                    training_result['model'],
                    data_prep['X_test'],
                    data_prep['y_test'],
                    problem_type,
                    data_prep['label_mapping']
                )
                
                # Display results
                st.success("✅ AutoML completed successfully!")
                
                # Model information
                st.subheader("🏆 Best Model")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Type", training_result['best_estimator'])
                with col2:
                    st.metric("Training Time", f"{training_result['training_time']:.1f} seconds")
                with col3:
                    main_metric = 'Accuracy' if problem_type == 'classification' else 'R² Score'
                    st.metric(f"Test {main_metric}", f"{evaluation['metrics'][main_metric.lower().replace(' ', '_')]:.4f}")
                
                # Display metrics
                st.subheader("📊 Model Performance")
                
                if problem_type == 'classification':
                    # Classification metrics
                    st.write("### Classification Report")
                    st.dataframe(evaluation['report'])
                    
                    # Confusion matrix
                    st.write("### Confusion Matrix")
                    cm_fig, cm_error = plot_confusion_matrix(
                        data_prep['y_test'],
                        evaluation['predictions'],
                        class_names=list(data_prep['inverse_mapping'].values()) if data_prep['inverse_mapping'] else None
                    )
                    if cm_error:
                        st.warning(f"Could not plot confusion matrix: {cm_error}")
                    else:
                        st.plotly_chart(cm_fig, use_container_width=True)
                    
                    # ROC Curve (for binary classification)
                    if evaluation['probabilities'] is not None and len(np.unique(data_prep['y_test'])) == 2:
                        st.write("### ROC Curve")
                        roc_fig, roc_error = plot_roc_curve(
                            data_prep['y_test'],
                            evaluation['probabilities'],
                            class_names=list(data_prep['inverse_mapping'].values()) if data_prep['inverse_mapping'] else None
                        )
                        if roc_error:
                            st.warning(f"Could not plot ROC curve: {roc_error}")
                        else:
                            st.plotly_chart(roc_fig, use_container_width=True)
                
                else:  # Regression
                    # Regression metrics
                    metrics_df = pd.DataFrame(
                        evaluation['metrics'].items(),
                        columns=['Metric', 'Value']
                    )
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Actual vs Predicted plot
                    st.write("### Actual vs Predicted Values")
                    pred_fig, err_fig, plot_error = plot_regression_results(
                        data_prep['y_test'],
                        evaluation['predictions']
                    )
                    
                    if plot_error:
                        st.warning(f"Could not plot regression results: {plot_error}")
                    else:
                        st.plotly_chart(pred_fig, use_container_width=True)
                        st.plotly_chart(err_fig, use_container_width=True)
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                importance_fig, importance_error = plot_feature_importance(
                    training_result['model'].model.estimator,
                    data_prep['feature_names']
                )
                
                if importance_error:
                    st.warning(f"Could not plot feature importance: {importance_error}")
                else:
                    st.plotly_chart(importance_fig, use_container_width=True)
                
                # Model saving
                st.subheader("💾 Save Model")
                model_name = st.text_input(
                    "Model name", 
                    value=f"best_{problem_type}_model",
                    key="model_name"
                )
                
                if st.button("💾 Save Model", key="save_model"):
                    try:
                        os.makedirs("saved_models", exist_ok=True)
                        model_path = os.path.join("saved_models", f"{model_name}.pkl")
                        joblib.dump(training_result['model'], model_path)
                        st.success(f"✅ Model saved successfully to {model_path}")
                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ An error occurred during AutoML: {str(e)}")
                st.exception(e)  # Show full traceback in the UI for debugging
    
    # Add some spacing
    st.markdown("---")
    
    # Show sample of the selected data
    st.subheader("📋 Selected Data Preview")
    st.dataframe(df_selected.head(), use_container_width=True)
    
    # Show data information
    st.subheader("📊 Data Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Shape:**", df_selected.shape)
        st.write("**Numerical Columns:**", len(num_cols))
        st.write("**Categorical Columns:**", len(cat_cols))
    
    with col2:
        st.write("**Missing Values:**", df_selected.isnull().sum().sum())
        st.write("**Duplicate Rows:**", df_selected.duplicated().sum())
        
        if problem_type == 'classification':
            if target_col in df_selected.columns:
                class_dist = df_selected[target_col].value_counts(normalize=True) * 100
                st.write("**Class Distribution (%):**")
                st.dataframe(class_dist, use_container_width=True)

def add_automl_tab(tab, df):
    """
    Add AutoML tab to the main app
    
    Args:
        tab: Streamlit tab object
        df: Input DataFrame
    """
    with tab:
        if df is not None and not df.empty:
            # Add documentation/help section
            with st.expander("ℹ️ About AutoML"):
                st.markdown("""
                ### AutoML (Automated Machine Learning)
                
                This feature automatically:
                - Preprocesses your data (handles missing values, encodes categories, etc.)
                - Splits data into training and testing sets
                - Finds the best machine learning model for your data
                - Tunes hyperparameters for optimal performance
                - Provides detailed evaluation metrics and visualizations
                
                **Note:** For classification tasks, the target variable will be automatically encoded if it's categorical.
                """)
            
            # Display the main AutoML UI
            display_automl_ui(df)
        else:
            st.warning("⚠️ Please load a dataset first using the 'Data Upload' tab")

# For testing
if __name__ == "__main__":
    st.set_page_config(page_title="AutoML Demo", layout="wide")
    st.title("AutoML Demo")
    
    # Example usage with sample data
    from pycaret.datasets import get_data
    data = get_data('titanic')
    display_automl_results(data, 'survived')
