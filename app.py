import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Load and prepare data ---
xls = pd.ExcelFile("./AcehDisasterDataset.xlsx")
year_sheets = [sheet for sheet in xls.sheet_names if sheet.isdigit()]
all_data = []
for sheet in year_sheets:
    df = xls.parse(sheet)
    df['Year'] = int(sheet)
    all_data.append(df)
df_all = pd.concat(all_data, ignore_index=True)

df_clean = df_all.copy()
df_clean = df_clean[~df_clean['Disaster'].isin([None, 'Disaster', 'Amount'])]
df_clean = df_clean.dropna(subset=['Disaster', 'Amount', 'Year'])
df_clean['Disaster'] = df_clean['Disaster'].str.replace(r'\d+\. ', '', regex=True)
df_clean['Disaster'] = df_clean['Disaster'].str.strip()
df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
df_clean = df_clean.dropna(subset=['Amount'])

df_model = df_clean[['Year', 'Disaster', 'Amount']].copy()
df_model = pd.get_dummies(df_model, columns=['Disaster'])

X = df_model.drop('Amount', axis=1)
y = df_model['Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # R² score

dummy_columns = X.columns.tolist()

# --- Streamlit Interface ---
st.title("Disaster Prediction in Aceh")

st.markdown(f"**Model Accuracy (R² Score):** {score:.2f}")

tahun = st.number_input("Input the year", min_value=2023, max_value=2100, step=1)

def predict_disasters_by_year(year):
    disaster_columns = [col for col in dummy_columns if col != 'Year']
    predictions = {}
    for disaster_col in disaster_columns:
        row = {col: 0 for col in dummy_columns}
        row['Year'] = year
        row[disaster_col] = 1
        input_df = pd.DataFrame([row])
        pred = model.predict(input_df)[0]
        disaster_name = disaster_col.replace("Disaster_", "") if "Disaster_" in disaster_col else disaster_col
        predictions[disaster_name] = max(0, round(pred))
    return predictions

if tahun:
    predictions = predict_disasters_by_year(tahun)
    st.subheader(f"Predicted number of disasters for the year {tahun}")
    df = pd.DataFrame(list(predictions.items()), columns=["Disaster Type", "Occurrences"])
    st.bar_chart(df.set_index("Disaster Type"))