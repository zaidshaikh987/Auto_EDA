"""
Test script for AutoML functionality
"""
import pandas as pd
import streamlit as st
from automl_functions import (
    prepare_data, 
    train_automl, 
    evaluate_model, 
    plot_feature_importance,
    plot_regression_results
)

def test_automl():
    """Test the AutoML functionality with a sample dataset"""
    st.title("🔍 AutoML Integration Test")
    
    # Load sample dataset (Titanic)
    @st.cache_data
    def load_data():
        try:
            # Try to load from example_dataset directory first
            df = pd.read_csv("example_dataset/titanic.csv")
            # Drop columns with too many missing values
            df = df.drop(columns=["Cabin"])
            # Drop rows with missing values for simplicity
            df = df.dropna()
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None
    
    df = load_data()
    
    if df is not None:
        st.success(f"✅ Successfully loaded dataset with shape {df.shape}")
        
        # Test classification
        st.header("🧪 Classification Test")
        st.write("Testing with 'Survived' as target variable")
        
        try:
            # Prepare data
            data_prep = prepare_data(df, "Survived", "classification")
            
            # Train model with shorter time budget for testing
            with st.spinner("Training classification model..."):
                training_result = train_automl(
                    data_prep['X_train'],
                    data_prep['y_train'],
                    "classification",
                    time_budget=30  # Shorter time budget for testing
                )
                
                # Evaluate model
                evaluation = evaluate_model(
                    training_result['model'],
                    data_prep['X_test'],
                    data_prep['y_test'],
                    "classification",
                    data_prep['label_mapping']
                )
                
                # Display results
                st.success("✅ Classification test completed successfully!")
                st.json({
                    "model_type": training_result['best_estimator'],
                    "training_time": f"{training_result['training_time']:.2f} seconds",
                    "accuracy": evaluation['metrics']['accuracy']
                })
                
                # Show feature importance
                st.subheader("Feature Importance")
                fig, error = plot_feature_importance(
                    training_result['model'].model.estimator,
                    data_prep['feature_names'],
                    top_n=10
                )
                if error:
                    st.warning(f"⚠️ {error}")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Classification test failed: {str(e)}")
            st.exception(e)
        
        # Test regression
        st.header("📈 Regression Test")
        st.write("Testing with 'Fare' as target variable")
        
        try:
            # Prepare data
            data_prep = prepare_data(df, "Fare", "regression")
            
            # Train model with shorter time budget for testing
            with st.spinner("Training regression model..."):
                training_result = train_automl(
                    data_prep['X_train'],
                    data_prep['y_train'],
                    "regression",
                    time_budget=30  # Shorter time budget for testing
                )
                
                # Evaluate model
                evaluation = evaluate_model(
                    training_result['model'],
                    data_prep['X_test'],
                    data_prep['y_test'],
                    "regression"
                )
                
                # Display results
                st.success("✅ Regression test completed successfully!")
                st.json({
                    "model_type": training_result['best_estimator'],
                    "training_time": f"{training_result['training_time']:.2f} seconds",
                    "r2_score": evaluation['metrics']['r2']
                })
                
                # Show actual vs predicted
                st.subheader("Actual vs Predicted")
                pred_fig, _, _ = plot_regression_results(
                    data_prep['y_test'],
                    evaluation['predictions']
                )
                st.plotly_chart(pred_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Regression test failed: {str(e)}")
            st.exception(e)
    
    else:
        st.error("❌ Failed to load dataset. Please check the file path and try again.")

if __name__ == "__main__":
    test_automl()
