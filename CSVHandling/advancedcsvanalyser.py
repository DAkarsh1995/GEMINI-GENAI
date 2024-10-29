import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import openpyxl
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Font, PatternFill

# Set default style for seaborn
sns.set(style="whitegrid", palette="muted")

# Streamlit Application
st.title("Advanced CSV Analysis App")
st.write("Upload a CSV, select target and features for analysis, and explore regression, clustering, and advanced EDA.")

# Sidebar for CSV Upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Display and process the CSV file
if uploaded_file is not None:
    st.write("CSV File Preview:")
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())
    
    # Dynamically categorize columns for selection based on data types
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

    # Allow the user to choose the dependent and independent columns
    st.sidebar.subheader("Select Variables")
    
    # Target selection restricted to numeric columns only
    target_column = st.sidebar.selectbox(
        "Select Target Column (Dependent Variable)", 
        options=numeric_columns, 
        help="Only numeric columns are allowed for regression as the target."
    )
    
    if target_column:
        # Exclude the target column from feature selection options
        feature_columns = st.sidebar.multiselect(
            "Select Feature Columns (Independent Variables)", 
            options=[col for col in data.columns if col != target_column],
            help="Choose numeric or categorical columns as features for the model."
        )
        
        # Basic validation for empty feature selection
        if not feature_columns:
            st.warning("Please select at least one feature column to proceed.")
        else:
            # Separate numeric and categorical features for further processing
            numeric_features = [col for col in feature_columns if col in numeric_columns]
            categorical_features = [col for col in feature_columns if col in categorical_columns]

            # Set up preprocessing and modeling pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    ("cat", OneHotEncoder(drop="first"), categorical_features)
                ]
            )

            # Define models to test
            models = {
                "Linear Regression": Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())]),
                "Random Forest Regressor": Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))]),
                "Decision Tree Regressor": Pipeline(steps=[("preprocessor", preprocessor), ("model", DecisionTreeRegressor(random_state=42))]),
                "Gradient Boosting Regressor": Pipeline(steps=[("preprocessor", preprocessor), ("model", GradientBoostingRegressor(random_state=42))]),
                "Support Vector Regressor": Pipeline(steps=[("preprocessor", preprocessor), ("model", SVR())])
            }
            
            X = data[feature_columns]
            y = data[target_column]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train and Evaluate Each Model
            model_results = []
            for name, pipeline in models.items():
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                r2 = r2_score(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)
                model_results.append((name, r2, mse, rmse, mae))
            
            # Best Model
            best_model = max(model_results, key=lambda x: x[1])
            
            st.subheader("Best Model")
            st.write(f"**Model**: {best_model[0]}")
            st.write(f"**R² Score**: {best_model[1]:.3f}")
            st.write(f"**MSE**: {best_model[2]:.3f}")
            st.write(f"**RMSE**: {best_model[3]:.3f}")
            st.write(f"**MAE**: {best_model[4]:.3f}")
            
            # Model Comparison Table
            model_comparison = pd.DataFrame(model_results, columns=["Model", "R² Score", "MSE", "RMSE", "MAE"])
            st.subheader("Model Comparison")
            st.dataframe(model_comparison)
            
            # Optional Clustering Analysis
            clustering_enabled = st.sidebar.checkbox("Enable Clustering Analysis")
            if clustering_enabled:
                num_clusters = st.slider("Number of Clusters", 2, 10, 3)
                clustering_data = preprocessor.fit_transform(data[feature_columns])
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                data["Cluster"] = kmeans.fit_predict(clustering_data)
                st.write("Clustering Analysis Completed")

            # EDA Section
            st.subheader("EDA on Selected Columns")
            
            # Enhanced Correlation Heatmap
            st.write("### Correlation Heatmap")
            plt.figure(figsize=(12, 8))
            selected_data = data[feature_columns + [target_column]]
            sns.heatmap(
                selected_data.corr(),
                annot=True,
                cmap="YlGnBu",
                vmin=-1,
                vmax=1,
                annot_kws={"size": 8},
                linewidths=0.5,
                fmt=".2f",
                square=True,
                cbar_kws={"shrink": .8, "aspect": 30}
            )
            st.pyplot(plt)
            
            # Enhanced Distribution Plot of Target Variable
            st.write("### Distribution of Target Variable")
            plt.figure(figsize=(10, 5))
            sns.histplot(y, kde=True, color="dodgerblue", edgecolor="black")
            plt.title(f"Distribution of {target_column}", fontsize=14)
            plt.xlabel(target_column, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.axvline(y.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {y.mean():.2f}")
            plt.axvline(y.median(), color="green", linestyle="--", linewidth=1.5, label=f"Median: {y.median():.2f}")
            plt.legend()
            st.pyplot(plt)
            
            # Enhanced Pairplot of Numeric Features
            st.write("### Pairplot of Numeric Features")
            pairplot_data = selected_data.select_dtypes(include=[np.number])
            if len(pairplot_data.columns) > 1:  # Only plot if there's more than one numeric column
                sns.pairplot(pairplot_data, diag_kind="kde", corner=True, plot_kws={"s": 20, "alpha": 0.7})
                st.pyplot(plt)
            
            # Prepare Excel File
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                model_comparison.to_excel(writer, sheet_name="Model Comparison", index=False)
                data.to_excel(writer, sheet_name="Clustered Data", index=False)
                
                # Style the sheets
                wb = writer.book
                ws1 = wb["Model Comparison"]
                ws2 = wb["Clustered Data"]

                # Apply styles to Model Comparison sheet
                for col in ws1.columns:
                    ws1.column_dimensions[col[0].column_letter].width = 20
                for cell in ws1["1:1"]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill("solid", fgColor="4F81BD")

                # Add Best Model Heading
                ws1["G1"] = "Best Model Details"
                ws1["G1"].font = Font(bold=True, size=14, color="4F81BD")

            # Download Button
            st.download_button(
                label="Download EDA and Model Results",
                data=output.getvalue(),
                file_name="eda_and_model_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Please upload a CSV file to proceed.")
