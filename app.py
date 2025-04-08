import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title
st.title("📊 Influencer ROI Tracker for Marketing Campaigns")

# Upload CSV
uploaded_file = st.file_uploader("Upload your campaign CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🧹 Raw Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    if 'Acquisition_Cost' in df.columns:
        df['Acquisition_Cost'] = df['Acquisition_Cost'].replace(r'[\$,]', '', regex=True).astype(float)

    if 'Conversion_Rate' in df.columns:
        if df['Conversion_Rate'].dtype == 'object':
            df['Conversion_Rate'] = df['Conversion_Rate'].str.replace('%', '', regex=False).astype(float) / 100

    # Calculate Revenue and ROI
    df['Revenue'] = df['Conversion_Rate'] * df['Acquisition_Cost'] * 10
    df['ROI'] = (df['Revenue'] - df['Acquisition_Cost']) / df['Acquisition_Cost']

    st.subheader("✅ Processed Data with ROI")
    st.dataframe(df[['Acquisition_Cost', 'Conversion_Rate', 'Revenue', 'ROI']].head())

    # ROI Distribution Plot
    st.subheader("📈 ROI Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['ROI'], kde=True, ax=ax)
    ax.set_xlabel("ROI")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Model Training
    st.subheader("🤖 Train Random Forest Model")
    X = df[['Acquisition_Cost', 'Conversion_Rate']]
    y = df['ROI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write(f"✅ Model Trained! Mean Squared Error: **{mse:.5f}**")

    # Prediction
    st.subheader("🔍 Predict ROI")
    acq_cost = st.number_input("Acquisition Cost", min_value=0.0, value=100.0)
    conv_rate = st.slider("Conversion Rate (%)", min_value=0.0, max_value=100.0, value=10.0)

    # Convert slider input to decimal
    conv_rate = conv_rate / 100

    user_input = pd.DataFrame([[acq_cost, conv_rate]], columns=['Acquisition_Cost', 'Conversion_Rate'])
    predicted_roi = model.predict(user_input)[0]

    st.write(f"📈 Predicted ROI: **{predicted_roi:.2f}**")

else:
    st.info("👆 Please upload a CSV file to begin.")
