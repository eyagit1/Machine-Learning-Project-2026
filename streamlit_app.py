"""
EcoCrop Tunisia — Interactive Streamlit Application
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="EcoCrop Tunisia — Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #FFFFFF 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #C8E6C9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        background-color: #f0f0f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1B5E20 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data Loading & Model Training (cached)
# ============================================================
@st.cache_resource
def load_and_train_model():
    """Load data, preprocess, and train the model."""
    # Try to load dataset
    data_path = 'ecocrop_cleaned_data (1).csv'
    if not os.path.exists(data_path):
        # Create demo data if file not found
        np.random.seed(42)
        n = 500
        govs = [f'Gov_{i}' for i in range(1, 25)]
        df = pd.DataFrame({
            'precipitation': np.random.exponential(5, n),
            'hc_air_temperature': np.random.normal(18, 5, n),
            'hc_relative_humidity': np.random.normal(60, 15, n),
            'solar_radiation': np.random.normal(250, 50, n),
            'wind_speed_sonic': np.random.normal(3, 1.5, n),
            'Governorate': np.random.choice(govs, n),
            'Year': np.random.randint(2016, 2025, n),
            'Cereales (T)': np.random.exponential(5000, n),
            'Maraichage (T)': np.random.exponential(1000, n),
            'Legumineuses (T)': np.random.exponential(500, n),
            'Fourrages (T)': np.random.exponential(3000, n),
            'Arboriculture (T)': np.random.exponential(2000, n),
            'Olives (T)': np.random.exponential(4000, n),
            'Cultures industrielles (T)': np.random.exponential(800, n),
        })
        st.warning("⚠️ Dataset not found. Using demo data for demonstration.")
    else:
        df = pd.read_csv(data_path)

    # Feature engineering
    def temp_to_season(t):
        if t < 10: return 'Winter'
        elif t < 18: return 'Spring/Autumn'
        else: return 'Summer'

    df['Season'] = df['hc_air_temperature'].apply(temp_to_season)
    df['Rainfall_Bin'] = pd.cut(df['precipitation'],
                                 bins=[-0.01, 0, 2, 10, 30, 200],
                                 labels=['No Rain', 'Trace', 'Light', 'Moderate', 'Heavy'])
    df['Temp_Zone'] = pd.cut(df['hc_air_temperature'],
                              bins=[0, 10, 18, 25, 45],
                              labels=['Cold', 'Mild', 'Warm', 'Hot'])
    df['Heat_Index'] = df['hc_air_temperature'] * (1 + df['hc_relative_humidity'] / 100)

    # Encoding
    le_gov = LabelEncoder()
    df['Governorate_Encoded'] = le_gov.fit_transform(df['Governorate'].astype(str))

    le_season = LabelEncoder()
    df['Season_Encoded'] = le_season.fit_transform(df['Season'].astype(str))

    le_rain = LabelEncoder()
    df['Rainfall_Bin_Encoded'] = le_rain.fit_transform(df['Rainfall_Bin'].astype(str))

    le_temp = LabelEncoder()
    df['Temp_Zone_Encoded'] = le_temp.fit_transform(df['Temp_Zone'].astype(str))

    # Features
    X_COLS = [
        'precipitation', 'hc_air_temperature', 'hc_relative_humidity',
        'solar_radiation', 'wind_speed_sonic', 'Year',
        'Governorate_Encoded', 'Season_Encoded',
        'Rainfall_Bin_Encoded', 'Temp_Zone_Encoded', 'Heat_Index'
    ]

    X = df[X_COLS].fillna(0)
    y = df['Cereales (T)']

    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_COLS)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_COLS)

    # Train model
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=2,
        max_features='sqrt', random_state=42
    )
    model.fit(X_train_s, y_train)

    # Metrics
    preds = model.predict(X_test_s)
    metrics = {
        'r2': r2_score(y_test, preds),
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds),
    }

    return {
        'model': model,
        'scaler': scaler,
        'le_gov': le_gov,
        'le_season': le_season,
        'le_rain': le_rain,
        'le_temp': le_temp,
        'df': df,
        'X_COLS': X_COLS,
        'metrics': metrics,
        'governorates': sorted(df['Governorate'].unique().tolist()),
    }

# Load everything
with st.spinner("🌱 Loading model and data..."):
    artifacts = load_and_train_model()

model = artifacts['model']
scaler = artifacts['scaler']
le_gov = artifacts['le_gov']
le_season = artifacts['le_season']
le_rain = artifacts['le_rain']
le_temp = artifacts['le_temp']
df = artifacts['df']
X_COLS = artifacts['X_COLS']
metrics = artifacts['metrics']
governorates = artifacts['governorates']

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/000000/ear-of-rice-emoji.png", width=80)
    st.title("🌾 EcoCrop Tunisia")
    st.caption("Crop Yield Prediction")
    st.divider()

    st.subheader("Navigation")
    st.info("Use the tabs above to navigate:\n\n"
            "🏠 **Home** — Overview\n"
            "🎯 **Predict** — Make predictions\n"
            "📊 **Insights** — Model analysis\n"
            "📋 **Data** — Explore dataset")

    st.divider()
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{metrics['r2']:.4f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:,.0f} T")
    st.metric("MAE", f"{metrics['mae']:,.0f} Tonnes")

    st.divider()
    st.caption("Built by Ferdaws Saidi & Aya Gharsalli\n© 2024")

# ============================================================
# Main Content
# ============================================================
st.markdown('<p class="main-header">🌾 EcoCrop Tunisia</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning-Powered Crop Yield Prediction for Tunisian Agriculture</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🎯 Predict", "📊 Model Insights", "📋 Data Explorer"])

# ============================================================
# TAB 1: Home
# ============================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem;">📐</div>
            <div style="font-size:1.5rem;font-weight:bold;color:#1B5E20;">{:,}</div>
            <div style="color:#666;">Observations</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem;">🗺️</div>
            <div style="font-size:1.5rem;font-weight:bold;color:#1B5E20;">{}</div>
            <div style="color:#666;">Governorates</div>
        </div>
        """.format(df['Governorate'].nunique()), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem;">📅</div>
            <div style="font-size:1.5rem;font-weight:bold;color:#1B5E20;">{}–{}</div>
            <div style="color:#666;">Years Covered</div>
        </div>
        """.format(df['Year'].min(), df['Year'].max()), unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem;">🎯</div>
            <div style="font-size:1.5rem;font-weight:bold;color:#1B5E20;">{:.1%}</div>
            <div style="color:#666;">Model Accuracy</div>
        </div>
        """.format(metrics['r2']), unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🌱 About This Project")
        st.write("""
        This application predicts **cereal crop yield** (in Tonnes) across Tunisia's 24 governorates
        using weather and regional data from 2016 to 2024.

        **How it works:**
        1. Weather data (temperature, precipitation, humidity, solar radiation, wind) is combined
           with regional and temporal information
        2. Features are engineered (Heat Index, Season, Rainfall Categories)
        3. A tuned **Random Forest** model makes the prediction
        4. The model achieves **R² = {:.4f}** on the test set
        """.format(metrics['r2']))

    with col_b:
        st.subheader("🌡️ Input Features")
        st.write("""
        The model uses **11 features** to predict yield:

        | Category | Features |
        |---|---|
        | **Weather** | Precipitation, Temperature, Humidity, Solar Radiation, Wind Speed |
        | **Engineered** | Heat Index, Season, Rainfall Bin, Temperature Zone |
        | **Context** | Governorate, Year |

        **Top 3 predictors:**
        1. 🗺️ Governorate (regional identity)
        2. 🌡️ Heat Index (combined stress)
        3. ☀️ Solar Radiation (photosynthesis)
        """)

    st.markdown("---")
    st.subheader("📊 Production by Governorate")
    crop_cols = ['Cereales (T)', 'Maraichage (T)', 'Legumineuses (T)',
                 'Fourrages (T)', 'Arboriculture (T)', 'Olives (T)', 'Cultures industrielles (T)']
    available_crops = [c for c in crop_cols if c in df.columns]
    if available_crops:
        df['Total_Production'] = df[available_crops].sum(axis=1)
        gov_prod = df.groupby('Governorate')['Total_Production'].sum().sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.YlGn(np.linspace(0.3, 0.9, len(gov_prod)))
        ax.barh(gov_prod.index, gov_prod.values / 1e6, color=colors, edgecolor='grey', linewidth=0.3)
        ax.set_xlabel('Total Production (Million Tonnes)', fontsize=11)
        ax.set_title('Top 15 Governorates by Total Agricultural Production', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================
# TAB 2: Predict
# ============================================================
with tab2:
    st.subheader("🎯 Predict Cereal Yield")
    st.write("Adjust the sliders below to simulate different weather conditions and get a yield prediction.")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        precipitation = st.slider("🌧️ Precipitation (mm)", 0.0, 150.0, 5.0, 0.5)
        temperature = st.slider("🌡️ Air Temperature (°C)", -5.0, 45.0, 18.0, 0.5)
        humidity = st.slider("💧 Relative Humidity (%)", 10.0, 100.0, 60.0, 1.0)

    with col_in2:
        solar = st.slider("☀️ Solar Radiation (W/m²)", 50.0, 500.0, 250.0, 5.0)
        wind = st.slider("💨 Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.5)
        year = st.slider("📅 Year", 2016, 2024, 2023)

    with col_in3:
        selected_gov = st.selectbox("🗺️ Governorate", governorates)

    # Compute derived features
    heat_index = temperature * (1 + humidity / 100)
    if temperature < 10:
        season = 'Winter'
    elif temperature < 18:
        season = 'Spring/Autumn'
    else:
        season = 'Summer'

    if precipitation <= 0:
        rain_bin = 'No Rain'
    elif precipitation <= 2:
        rain_bin = 'Trace'
    elif precipitation <= 10:
        rain_bin = 'Light'
    elif precipitation <= 30:
        rain_bin = 'Moderate'
    else:
        rain_bin = 'Heavy'

    if temperature < 10:
        temp_zone = 'Cold'
    elif temperature < 18:
        temp_zone = 'Mild'
    elif temperature < 25:
        temp_zone = 'Warm'
    else:
        temp_zone = 'Hot'

    # Encode
    try:
        gov_encoded = le_gov.transform([selected_gov])[0]
    except ValueError:
        gov_encoded = 0

    try:
        season_encoded = le_season.transform([season])[0]
    except ValueError:
        season_encoded = 1

    try:
        rain_encoded = le_rain.transform([rain_bin])[0]
    except ValueError:
        rain_encoded = 0

    try:
        temp_encoded = le_temp.transform([temp_zone])[0]
    except ValueError:
        temp_encoded = 1

    # Create input DataFrame
    input_df = pd.DataFrame([{
        'precipitation': precipitation,
        'hc_air_temperature': temperature,
        'hc_relative_humidity': humidity,
        'solar_radiation': solar,
        'wind_speed_sonic': wind,
        'Year': year,
        'Governorate_Encoded': gov_encoded,
        'Season_Encoded': season_encoded,
        'Rainfall_Bin_Encoded': rain_encoded,
        'Temp_Zone_Encoded': temp_encoded,
        'Heat_Index': heat_index,
    }])

    # Scale and predict
    input_scaled = pd.DataFrame(scaler.transform(input_df[X_COLS]), columns=X_COLS)
    prediction = model.predict(input_scaled)[0]
    pred_lower = prediction - metrics['mae']
    pred_upper = prediction + metrics['mae']

    # Display prediction
    st.markdown("---")
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:1rem;color:#666;">Predicted Yield</div>
            <div style="font-size:2.5rem;font-weight:bold;color:#1B5E20;">{:,.0f}</div>
            <div style="color:#666;">Tonnes of Cereals</div>
        </div>
        """.format(prediction), unsafe_allow_html=True)

    with col_res2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:1rem;color:#666;">Confidence Range</div>
            <div style="font-size:1.5rem;font-weight:bold;color:#1565C0;">{:,.0f} – {:,.0f}</div>
            <div style="color:#666;">Tonnes (±1 MAE)</div>
        </div>
        """.format(pred_lower, pred_upper), unsafe_allow_html=True)

    with col_res3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:1rem;color:#666;">Derived Features</div>
            <div style="font-size:0.9rem;margin-top:8px;">
            <b>Season:</b> {}<br>
            <b>Rain Category:</b> {}<br>
            <b>Temp Zone:</b> {}<br>
            <b>Heat Index:</b> {:.1f}
            </div>
        </div>
        """.format(season, rain_bin, temp_zone, heat_index), unsafe_allow_html=True)

    st.info("💡 **Note:** This prediction is based on weather and regional patterns. Actual yield depends on additional factors like soil quality, irrigation, seed variety, and pest management.")

# ============================================================
# TAB 3: Model Insights
# ============================================================
with tab3:
    st.subheader("📊 Model Insights & Analysis")

    # Feature Importance
    col_fi1, col_fi2 = st.columns(2)

    with col_fi1:
        st.markdown("#### 🌟 Feature Importance")
        importances = pd.Series(model.feature_importances_, index=X_COLS).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
        importances.plot(kind='barh', ax=ax, color=colors, edgecolor='grey', linewidth=0.3)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title('Random Forest Feature Importances', fontsize=12, fontweight='bold')
        ax.axvline(importances.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {importances.mean():.3f}')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_fi2:
        st.markdown("#### 🥧 Top 5 Features Share")
        top5 = importances.sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors_pie = ['#1B5E20', '#2E7D32', '#4CAF50', '#81C784', '#C8E6C9']
        wedges, texts, autotexts = ax.pie(
            top5.values, labels=top5.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, pctdistance=0.75
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight('bold')
        ax.set_title('Top 5 Features — Relative Importance', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Correlation heatmap
    st.markdown("#### 🔗 Feature Correlation Matrix")
    weather_cols = ['precipitation', 'hc_air_temperature', 'hc_relative_humidity',
                    'solar_radiation', 'wind_speed_sonic', 'Heat_Index']
    available_weather = [c for c in weather_cols if c in df.columns]
    if len(available_weather) >= 2:
        corr = df[available_weather].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    linewidths=0.5, ax=ax, square=True, annot_kws={'size': 10})
        ax.set_title('Correlation Heatmap — Weather Features', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Yield by governorate boxplot
    st.markdown("#### 🗺️ Cereal Yield Distribution by Governorate")
    fig, ax = plt.subplots(figsize=(14, 6))
    top_govs = df['Governorate'].value_counts().head(12).index.tolist()
    df_top = df[df['Governorate'].isin(top_govs)]
    sns.boxplot(data=df_top, x='Governorate', y='Cereales (T)', palette='Greens', ax=ax)
    ax.set_title('Cereal Yield Distribution (Top 12 Governorates)', fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Cereals (Tonnes)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================
# TAB 4: Data Explorer
# ============================================================
with tab4:
    st.subheader("📋 Dataset Explorer")

    # Filters
    col_filt1, col_filt2, col_filt3 = st.columns(3)
    with col_filt1:
        filter_gov = st.selectbox("Filter by Governorate", ["All"] + governorates)
    with col_filt2:
        filter_year = st.selectbox("Filter by Year", ["All"] + sorted(df['Year'].unique().tolist()))
    with col_filt3:
        show_rows = st.number_input("Rows to display", min_value=10, max_value=len(df), value=50)

    # Apply filters
    df_filtered = df.copy()
    if filter_gov != "All":
        df_filtered = df_filtered[df_filtered['Governorate'] == filter_gov]
    if filter_year != "All":
        df_filtered = df_filtered[df_filtered['Year'] == int(filter_year)]

    st.write(f"Showing **{min(show_rows, len(df_filtered))}** of **{len(df_filtered)}** filtered records")

    # Display data
    display_cols = ['Governorate', 'Year', 'precipitation', 'hc_air_temperature',
                    'hc_relative_humidity', 'solar_radiation', 'wind_speed_sonic', 'Cereales (T)']
    available_display = [c for c in display_cols if c in df_filtered.columns]
    st.dataframe(df_filtered[available_display].head(show_rows).reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    # Statistics
    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        st.markdown("#### 📊 Descriptive Statistics")
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df_filtered[numeric_cols].describe().round(2), use_container_width=True)

    with col_stat2:
        st.markdown("#### 🌾 Yield Statistics by Crop")
        crop_cols_all = ['Cereales (T)', 'Maraichage (T)', 'Legumineuses (T)',
                         'Fourrages (T)', 'Arboriculture (T)', 'Olives (T)', 'Cultures industrielles (T)']
        available_crops = [c for c in crop_cols_all if c in df_filtered.columns]
        if available_crops:
            crop_stats = df_filtered[available_crops].agg(['mean', 'std', 'min', 'max']).T
            crop_stats.columns = ['Mean', 'Std', 'Min', 'Max']
            st.dataframe(crop_stats.round(0), use_container_width=True)

    # Download button
    st.markdown("---")
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"ecocrop_filtered_{filter_gov}_{filter_year}.csv",
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.caption("🌾 EcoCrop Tunisia — Ferdaws Saidi & Aya Gharsalli — 2024 | Built with Streamlit & Scikit-learn")