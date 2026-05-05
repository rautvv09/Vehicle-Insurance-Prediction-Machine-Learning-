import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Insurance Claim Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #1a1a2e;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #1a1a2e;
    border-left: 4px solid #e63946;
    padding-left: 10px;
    margin: 1.5rem 0 1rem 0;
}

.card {
    background: #f8f9fc;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    border: 1px solid #e5e7eb;
}

.result-box-claim {
    background: linear-gradient(135deg, #ff4d4d, #e63946);
    color: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(230,57,70,0.3);
}

.result-box-no-claim {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(34,197,94,0.3);
}

.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label { font-family: 'DM Serif Display', serif; font-size: 1.8rem; }
.result-desc { font-size: 0.95rem; opacity: 0.9; margin-top: 0.5rem; }

.stButton > button {
    background: linear-gradient(135deg, #e63946, #c1121f);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2.5rem;
    font-size: 1.05rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(230,57,70,0.35);
}

div[data-testid="stSidebar"] {
    background: #1a1a2e;
}
div[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

.tag {
    display: inline-block;
    background: #e63946;
    color: white;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model Bundle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    bundle = joblib.load("insurance_bundle_updated.pkl")
    return bundle["model"], bundle["scaler"], bundle["feature_columns"]

try:
    model, scaler, feature_columns = load_bundle()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model bundle: {e}\nMake sure `insurance_bundle.pkl` is in the same directory.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 InsurePredict")
    st.markdown("---")
    st.markdown("""
    **About this App**  
    Predicts whether a vehicle insurance policy will result in a claim, based on:
    - 📄 Policy details
    - 🚘 Vehicle specifications  
    - 🛡️ Safety features
    - 🌍 Area & segment data
    """)
    st.markdown("---")
    st.markdown("**Model Pipeline**")
    st.markdown("""
    - SMOTE oversampling  
    - StandardScaler normalization  
    - Decision Tree Classifier  
    - One-hot encoded features  
    """)
    st.markdown("---")
    st.caption("Built with Streamlit • sklearn")

# ── Main Title ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚗 Vehicle Insurance Claim Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fill in the vehicle and policy details to predict if a claim is likely.</div>', unsafe_allow_html=True)

if not model_loaded:
    st.stop()

# ── Input Form ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:

    # ── Section 1: Policy Info ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Policy Information</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            policy_tenure = st.number_input("Policy Tenure (years)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        with c2:
            age_of_car = st.number_input("Age of Car (years)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        with c3:
            age_of_policyholder = st.number_input("Age of Policyholder", min_value=18, max_value=80, value=35)
        
        population_density = st.number_input("Population Density", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)

    # ── Section 2: Vehicle Specs ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🔧 Vehicle Specifications</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        make = st.number_input("Make (encoded ID)", min_value=0, max_value=100, value=10,
                               help="Numeric code for the car make after label encoding")
        displacement = st.number_input("Engine Displacement (cc)", min_value=600, max_value=6000, value=1500)
        cylinder = st.selectbox("Cylinders", options=[3, 4, 6, 8, 12], index=1)
    with c2:
        gear_box = st.selectbox("Gearbox (no. of gears)", options=[4, 5, 6, 7, 8], index=1)
        turning_radius = st.number_input("Turning Radius (m)", min_value=3.0, max_value=8.0, value=5.2, step=0.1)
        airbags = st.selectbox("Airbags", options=[0, 2, 4, 6, 8], index=1)
    with c3:
        ncap_rating = st.selectbox("NCAP Safety Rating", options=[0, 1, 2, 3, 4, 5], index=3)
        length = st.number_input("Length (mm)", min_value=3000, max_value=6000, value=4200)
        width = st.number_input("Width (mm)", min_value=1400, max_value=2500, value=1750)

    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("Height (mm)", min_value=1300, max_value=2200, value=1540)
    with c2:
        gross_weight = st.number_input("Gross Weight (kg)", min_value=800, max_value=4000, value=1700)

    # ── Section 3: Engine Performance ───────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Engine Performance</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Torque**")
        torque = st.number_input("Torque (Nm)", min_value=50.0, max_value=800.0, value=163.0, step=1.0)
        rpm_T = st.number_input("Torque @ RPM", min_value=500.0, max_value=6000.0, value=2500.0, step=50.0)
    with c2:
        st.markdown("**Power**")
        power = st.number_input("Power (bhp)", min_value=40.0, max_value=700.0, value=115.0, step=1.0)
        rpm_P = st.number_input("Power @ RPM", min_value=1000.0, max_value=8000.0, value=4000.0, step=50.0)

    torque_rpm_ratio = torque / rpm_T if rpm_T != 0 else 0
    power_rpm_ratio = power / rpm_P if rpm_P != 0 else 0
    st.info(f"📐 Torque/RPM Ratio: `{torque_rpm_ratio:.5f}` &nbsp;&nbsp; Power/RPM Ratio: `{power_rpm_ratio:.5f}`")

    # ── Section 4: Safety Features ───────────────────────────────────────────
    st.markdown('<div class="section-header">🛡️ Safety & Comfort Features</div>', unsafe_allow_html=True)

    safety_features = {
        "is_esc": "Electronic Stability Control (ESC)",
        "is_adjustable_steering": "Adjustable Steering",
        "is_tpms": "Tyre Pressure Monitoring (TPMS)",
        "is_parking_sensors": "Parking Sensors",
        "is_parking_camera": "Parking Camera",
        "is_front_fog_lights": "Front Fog Lights",
        "is_rear_window_wiper": "Rear Window Wiper",
        "is_rear_window_washer": "Rear Window Washer",
        "is_rear_window_defogger": "Rear Window Defogger",
        "is_brake_assist": "Brake Assist",
        "is_power_door_locks": "Power Door Locks",
        "is_central_locking": "Central Locking",
        "is_power_steering": "Power Steering",
        "is_driver_seat_height_adjustable": "Driver Seat Height Adjustable",
        "is_day_night_rear_view_mirror": "Day/Night Rear View Mirror",
        "is_ecw": "Engine Check Warning (ECW)",
        "is_speed_alert": "Speed Alert",
    }

    safety_values = {}
    cols = st.columns(3)
    for idx, (key, label) in enumerate(safety_features.items()):
        with cols[idx % 3]:
            safety_values[key] = int(st.checkbox(label, value=True if key in [
                "is_power_steering", "is_central_locking", "is_brake_assist",
                "is_tpms", "is_speed_alert"
            ] else False))

    # ── Section 5: Categorical Features ─────────────────────────────────────
    st.markdown('<div class="section-header">🌍 Segment & Category Details</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        segment = st.selectbox("Segment", options=["A1","B1","C1","D1","E1","F1"],
                               help="Vehicle segment (A1=0, B1=1 ... F1=5)")
        segment_val = ["A1","B1","C1","D1","E1","F1"].index(segment)

        area_cluster = st.selectbox("Area Cluster", options=[f"C{i}" for i in range(1, 23)],
                                    help="Geographic area cluster")
        area_cluster_val = [f"C{i}" for i in range(1, 23)].index(area_cluster)

    with c2:
        car_model = st.selectbox("Model", options=[f"Model {i}" for i in range(1, 12)],
                                 help="Encoded vehicle model (0–10)")
        car_model_val = [f"Model {i}" for i in range(1, 12)].index(car_model)

        fuel_type = st.selectbox("Fuel Type", options=["Type 1 (CNG/EV)", "Type 2 (Diesel)", "Type 3 (Petrol)"])
        fuel_type_val = ["Type 1 (CNG/EV)", "Type 2 (Diesel)", "Type 3 (Petrol)"].index(fuel_type)

    with c3:
        engine_type = st.selectbox("Engine Type", options=[f"Engine {i}" for i in range(1, 12)])
        engine_type_val = [f"Engine {i}" for i in range(1, 12)].index(engine_type)

        rear_brakes = st.selectbox("Rear Brakes", options=["Disc", "Drum"])
        rear_brakes_val = ["Disc", "Drum"].index(rear_brakes)

        transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
        transmission_val = ["Manual", "Automatic"].index(transmission)

        steering = st.selectbox("Steering Type", options=["Electric", "Hydraulic", "Manual"])
        steering_val = ["Electric", "Hydraulic", "Manual"].index(steering)


# ── RIGHT COLUMN: Preview + Predict ──────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-header">📊 Input Summary</div>', unsafe_allow_html=True)
    
    summary_data = {
        "Field": [
            "Policy Tenure", "Car Age", "Policyholder Age",
            "Population Density", "Make", "Displacement",
            "Cylinders", "Airbags", "NCAP Rating",
            "Torque", "Power", "Segment", "Area Cluster",
            "Fuel Type", "Transmission", "Steering"
        ],
        "Value": [
            f"{policy_tenure} yrs", f"{age_of_car} yrs", f"{age_of_policyholder} yrs",
            f"{population_density:,.0f}", f"ID {make}", f"{displacement} cc",
            cylinder, airbags, f"⭐ {ncap_rating}/5",
            f"{torque} Nm @ {int(rpm_T)} rpm",
            f"{power} bhp @ {int(rpm_P)} rpm",
            segment, area_cluster, fuel_type, transmission, steering
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True,
                 height=460)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Claim Likelihood", use_container_width=True)

    if predict_btn:
        # ── Build Feature Vector ──────────────────────────────────────────────
        base = {
            "policy_tenure": policy_tenure,
            "age_of_car": age_of_car,
            "age_of_policyholder": age_of_policyholder,
            "population_density": population_density,
            "make": make,
            "max_torque": 0,        # raw string cols not used (extracted below)
            "max_power": 0,
            "airbags": airbags,
            **safety_values,
            "displacement": displacement,
            "cylinder": cylinder,
            "gear_box": gear_box,
            "turning_radius": turning_radius,
            "Length": length,
            "width": width,
            "height": height,
            "Gross_weight": gross_weight,
            "ncap_rating": ncap_rating,
            "rpm_T": rpm_T,
            "torque": torque,
            "rpm_P": rpm_P,
            "power": power,
            "torque_rpm_ratio": torque_rpm_ratio,
            "power_rpm_ratio": power_rpm_ratio,
        }

        # ── One-Hot: Segment (drop_first → segment_0 dropped) ───────────────
        for i in range(1, 6):
            base[f"segment_{i}"] = 1 if segment_val == i else 0

        # ── One-Hot: Area Cluster ─────────────────────────────────────────────
        for i in range(1, 22):
            base[f"area_cluster_{i}"] = 1 if area_cluster_val == i else 0

        # ── One-Hot: Model ────────────────────────────────────────────────────
        for i in range(1, 11):
            base[f"model_{i}"] = 1 if car_model_val == i else 0

        # ── One-Hot: Fuel Type ────────────────────────────────────────────────
        for i in range(1, 3):
            base[f"fuel_type_{i}"] = 1 if fuel_type_val == i else 0

        # ── One-Hot: Engine Type ──────────────────────────────────────────────
        for i in range(1, 11):
            base[f"engine_type_{i}"] = 1 if engine_type_val == i else 0

        # ── One-Hot: Rear Brakes ──────────────────────────────────────────────
        base["rear_brakes_type_1"] = 1 if rear_brakes_val == 1 else 0

        # ── One-Hot: Transmission ─────────────────────────────────────────────
        base["transmission_type_1"] = 1 if transmission_val == 1 else 0

        # ── One-Hot: Steering ─────────────────────────────────────────────────
        for i in range(1, 3):
            base[f"steering_type_{i}"] = 1 if steering_val == i else 0

        # ── Align to trained feature columns ─────────────────────────────────
        input_df = pd.DataFrame([base])

        # Drop cols not in feature_columns, add missing cols as 0
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]

        # ── Scale ─────────────────────────────────────────────────────────────
        input_scaled = scaler.transform(input_df)

        # ── Predict ───────────────────────────────────────────────────────────
        prediction = model.predict(input_scaled)[0]

        # ── Display Result ────────────────────────────────────────────────────
        st.markdown("---")
        if prediction == 1:
            st.markdown("""
            <div class="result-box-claim">
                <div class="result-emoji">⚠️</div>
                <div class="result-label">Claim Likely</div>
                <div class="result-desc">This policy profile has a higher risk of resulting in an insurance claim.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box-no-claim">
                <div class="result-emoji">✅</div>
                <div class="result-label">No Claim Expected</div>
                <div class="result-desc">This policy profile appears low-risk with no claim predicted.</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Decision Path (if DT) ─────────────────────────────────────────────
        with st.expander("🔍 Raw Input Vector (Debug)"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)