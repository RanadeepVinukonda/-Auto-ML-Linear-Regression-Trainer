import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="📈 Auto ML Trainer", layout="wide")
st.title("🤖 Auto ML Linear Regression Trainer")

# Initialize session state
for key in ['original_df', 'df', 'model', 'X_columns', 'target_column']:
    st.session_state.setdefault(key, None)

# Upload Section
st.markdown("### 📤 Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Load into DataFrame
    df = pd.read_csv(uploaded_file)
    st.session_state.original_df = df.copy()
    st.session_state.df = df.copy()
    st.success("✅ Dataset uploaded successfully!")

# Revert Section
if st.session_state.df is not None and st.button("Revert to Original Dataset"):
    st.session_state.df = st.session_state.original_df.copy()
    st.success("✅ Reverted to original dataset.")

# Preview Section
if st.session_state.df is not None:
    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    st.info(f"🧾 Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")

# Missing Values Section
if st.session_state.df is not None and st.session_state.df.isnull().sum().sum() > 0:
    with st.expander("⚠️ Handle Missing Values"):
        st.warning("⚠️ Missing values detected.")
        action = st.selectbox("Choose how to handle missing values", ["Drop Rows", "Fill with Mean", "Fill with 0"])
        if st.button("🧹 Clean Missing Values"):
            if action == "Drop Rows":
                st.session_state.df.dropna(inplace=True)
            elif action == "Fill with Mean":
                st.session_state.df.fillna(st.session_state.df.mean(numeric_only=True), inplace=True)
            elif action == "Fill with 0":
                st.session_state.df.fillna(0, inplace=True)
            st.success("✅ Missing values handled.")
            st.rerun()

# Column Removal Section
if st.session_state.df is not None:
    with st.expander("🗑️ Remove Unwanted Columns"):
        cols_to_remove = st.multiselect("Select columns to remove", st.session_state.df.columns)
        if st.button("🧽 Remove Columns"):
            st.session_state.df.drop(columns=cols_to_remove, inplace=True)
            st.success(f"✅ Removed: {', '.join(cols_to_remove)}")
            st.rerun()

# Training Section
if st.session_state.df is not None:
    st.markdown("### 🏋️ Train Linear Regression Model")
    numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ Not enough numeric columns to train.")
    else:
        target = st.selectbox("🎯 Select Target (Y)", numeric_cols, key="target")
        features = st.multiselect("📊 Select Features (X)", [col for col in numeric_cols if col != target], key="features")

        if st.button("🚀 Train Model"):
            X = st.session_state.df[features]
            y = st.session_state.df[target]

            if X.empty or y.empty:
                st.error("❌ Selected features or target are empty.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.session_state.model = model
            st.session_state.X_columns = features
            st.session_state.target_column = target
            st.session_state.mse = mse
            st.session_state.r2 = r2
            st.session_state.result_df = pd.DataFrame({
                "Actual": y_test.reset_index(drop=True),
                "Predicted": pd.Series(y_pred)
            })

            st.success("✅ Model trained successfully!")

# Evaluation Section
if st.session_state.get('model') is not None:
    st.markdown("### 📈 Model Evaluation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📉 Mean Squared Error (MSE)", f"{st.session_state.mse:.2f}")
    with col2:
        st.metric("📈 R² Score", f"{st.session_state.r2:.2f}")
    with col3:
        st.metric("Accuracy of Model", f"{(st.session_state.r2 * 100):.2f}%")


    with st.expander("📋 Prediction Comparison Table"):
        st.dataframe(st.session_state.result_df, use_container_width=True)
# Prediction Input After Model Training
if st.session_state.get("model") is not None and st.session_state.get("X_columns") is not None:
    with st.sidebar:
        st.markdown("## 🧮 Make a Prediction")
        st.markdown("Adjust the sliders below to predict the target value.")

        input_data = {}
        for col in st.session_state.X_columns:
            if col in st.session_state.df.columns:
                min_val = float(st.session_state.df[col].min())
                max_val = float(st.session_state.df[col].max())
                default_val = float(st.session_state.df[col].mean())
                step_val = 1.0 if st.session_state.df[col].dtype == 'int64' else 0.01

                input_val = st.slider(
                    label=f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=step_val,
                    key=f"slider_{col}"
                )
                input_data[col] = input_val
            else:
                st.sidebar.error(f"❌ Column '{col}' not found. Retrain the model.")

    input_df = pd.DataFrame([input_data])[st.session_state.X_columns]

    try:
        prediction = st.session_state.model.predict(input_df)[0]
        st.markdown("### 🎯 Prediction Result")
        st.success(f"Predicted `{st.session_state.target_column}`: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

else:
    st.warning("⚠️ Please train a model first in the 'Train Model' section.")