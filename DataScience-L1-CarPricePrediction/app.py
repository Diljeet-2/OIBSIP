from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_PATH = BASE_DIR / "car_data.csv"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_data.csv"

FALLBACK_ROWS = [
    {
        "Car_Name": "Maruti Swift",
        "Year": 2015,
        "Selling_Price": 250000,
        "Present_Price": 300000,
        "Kms_Driven": 60000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 1,
    },
    {
        "Car_Name": "Hyundai i20",
        "Year": 2018,
        "Selling_Price": 420000,
        "Present_Price": 480000,
        "Kms_Driven": 35000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Individual",
        "Transmission": "Manual",
        "Owner": 1,
    },
    {
        "Car_Name": "Toyota Innova",
        "Year": 2012,
        "Selling_Price": 650000,
        "Present_Price": 750000,
        "Kms_Driven": 120000,
        "Fuel_Type": "Diesel",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 2,
    },
    {
        "Car_Name": "Honda City",
        "Year": 2017,
        "Selling_Price": 480000,
        "Present_Price": 520000,
        "Kms_Driven": 55000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Individual",
        "Transmission": "Automatic",
        "Owner": 1,
    },
]

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #0f172a, #1d4ed8); padding: 24px; border-radius: 18px; margin-bottom: 18px;'>
        <h1 style='color: white; margin-bottom: 6px;'>🚗 Car Price Predictor</h1>
        <p style='color: #e2e8f0; margin: 0;'>Estimate the market value of any used car in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.image(str(ASSETS_DIR / "car_header.svg"), use_container_width=True)
st.write("Enter the car details to estimate its selling price.")


def ensure_data_file(path: Path) -> Path:
    if path.exists():
        return path

    fallback_df = pd.DataFrame(FALLBACK_ROWS)
    path.parent.mkdir(parents=True, exist_ok=True)
    fallback_df.to_csv(path, index=False)
    return path


@st.cache_data
def load_data(path: str):
    data_file = ensure_data_file(Path(path))
    df = pd.read_csv(data_file)
    required_columns = {
        "Car_Name",
        "Year",
        "Selling_Price",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
    }

    if not required_columns.issubset(df.columns):
        fallback_df = pd.DataFrame(FALLBACK_ROWS)
        fallback_df.to_csv(data_file, index=False)
        df = fallback_df

    df = df.dropna().drop_duplicates()
    return df


@st.cache_resource
def train_model(path: str):
    df = load_data(path)

    df = df.copy()
    df["Fuel_Type"] = df["Fuel_Type"].astype(str).str.lower().str.strip()
    df["Seller_Type"] = df["Seller_Type"].astype(str).str.lower().str.strip()
    df["Transmission"] = df["Transmission"].astype(str).str.lower().str.strip()
    df["Car_Age"] = 2025 - df["Year"]
    df["Brand"] = df["Car_Name"].astype(str).apply(lambda x: x.split()[0])

    X = df.drop(columns=["Selling_Price", "Car_Name"])
    y = df["Selling_Price"]

    categorical = ["Fuel_Type", "Seller_Type", "Transmission", "Brand"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(X, y)
    return model


def prepare_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["Fuel_Type"] = df["Fuel_Type"].astype(str).str.lower().str.strip()
    df["Seller_Type"] = df["Seller_Type"].astype(str).str.lower().str.strip()
    df["Transmission"] = df["Transmission"].astype(str).str.lower().str.strip()
    df["Car_Age"] = 2025 - df["Year"]
    df["Brand"] = df["Car_Name"].astype(str).apply(lambda x: x.split()[0])
    return df.drop(columns=["Car_Name"])


model = train_model(str(DATA_PATH))

st.subheader("📊 Quick Preview")
st.image(str(ASSETS_DIR / "price_preview.svg"), use_container_width=True)

with st.form("prediction_form"):
    car_name = st.text_input("Car Name", value="Maruti Swift")
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2018, step=1)
    present_price = st.number_input("Present Price", min_value=100000.0, max_value=5000000.0, value=450000.0, step=10000.0)
    kms_driven = st.number_input("Kilometers Driven", min_value=1000, max_value=500000, value=40000, step=1000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.number_input("Owner Count", min_value=0, max_value=5, value=1, step=1)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Car_Name": car_name,
                "Year": int(year),
                "Present_Price": float(present_price),
                "Kms_Driven": int(kms_driven),
                "Fuel_Type": fuel_type,
                "Seller_Type": seller_type,
                "Transmission": transmission,
                "Owner": int(owner),
            }
        ]
    )

    processed = prepare_input(input_df)
    prediction = model.predict(processed)[0]

    st.success(f"Estimated Selling Price: ₹{prediction:,.0f}")
