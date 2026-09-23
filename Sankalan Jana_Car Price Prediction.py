"""
Car Price Prediction & Analytics — Streamlit Web Application
Dataset: car price cleaned.csv (117 850 Polish used-car listings)
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "car price cleaned.csv"
MODEL_CACHE = "model_cache.pkl"

FUEL_LABELS = {
    "gasoline": "Gasoline",
    "diesel": "Diesel",
    "lpg": "LPG",
    "hybrid": "Hybrid",
    "electric": "Electric",
    "cng": "CNG",
}

PLN_TO_EUR = 0.23  # approximate

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["price", "year", "mileage", "vol_engine"])
    df = df[df["price"].between(1_000, 2_000_000)]
    df = df[df["year"].between(1990, 2023)]
    df = df[df["mileage"] >= 0]
    df = df[df["vol_engine"] > 0]
    df["age"] = 2024 - df["year"]
    df["price_eur"] = (df["price"] * PLN_TO_EUR).round(0)
    df["fuel_label"] = df["fuel"].map(FUEL_LABELS).fillna(df["fuel"].str.title())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING & MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = ["year", "mileage", "vol_engine", "age",
            "mark_enc", "fuel_enc", "province_enc"]
TARGET = "price"

MODELS = {
    "XGBoost": XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=18, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=42,
    ),
    "Ridge Regression": Ridge(alpha=10.0),
}


@st.cache_resource(show_spinner="Training models … (first run only)",
                   hash_funcs={pd.DataFrame: lambda df: df.shape})
def train_models(df: pd.DataFrame):
    # Encode categoricals
    encoders = {}
    df2 = df.copy()
    for col, enc_col in [("mark", "mark_enc"), ("fuel", "fuel_enc"), ("province", "province_enc")]:
        le = LabelEncoder()
        df2[enc_col] = le.fit_transform(df2[col].astype(str))
        encoders[col] = le

    X = df2[FEATURES].values
    y = df2[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale for Ridge
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = {}
    trained = {}

    for name, model in MODELS.items():
        if name == "Ridge Regression":
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        preds = np.clip(preds, 1_000, 2_000_000)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2, "predictions": preds, "actuals": y_test}
        trained[name] = model

    # Feature importance from XGBoost
    fi = dict(zip(FEATURES, trained["XGBoost"].feature_importances_))

    return trained, encoders, scaler, results, fi, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ──────────────────────────────────────────────────────────────────────────────
def predict_price(model, encoders, scaler, model_name,
                  year, mileage, vol_engine, mark, fuel, province):
    age = 2024 - year
    mark_enc = encoders["mark"].transform([mark])[0] if mark in encoders["mark"].classes_ else 0
    fuel_enc = encoders["fuel"].transform([fuel])[0] if fuel in encoders["fuel"].classes_ else 0
    prov_enc = encoders["province"].transform([province])[0] if province in encoders["province"].classes_ else 0

    X = np.array([[year, mileage, vol_engine, age, mark_enc, fuel_enc, prov_enc]], dtype=float)

    if model_name == "Ridge Regression":
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    return float(np.clip(pred, 1_000, 2_000_000))


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a1d23; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #64748b !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #1e293b !important; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        margin: 18px 0 10px 0;
    }

    /* Prediction result box */
    .pred-box {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-radius: 14px;
        padding: 24px 32px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .pred-box h2 { margin: 0; font-size: 2.2rem; font-weight: 800; }
    .pred-box p  { margin: 4px 0 0; font-size: 1rem; opacity: 0.85; }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        st.markdown("## 🚗 Car Price App")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🏠 Overview", "🔍 Price Predictor", "📊 Data Explorer",
             "🤖 Model Performance", "📈 Market Insights"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Dataset stats**")
        return page


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
def page_overview(df: pd.DataFrame):
    st.title("🚗 Car Price Prediction & Analytics")
    st.markdown("Explore **117 850** Polish used-car listings and predict market prices with ML models.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", f"{len(df):,}")
    c2.metric("Car Brands", f"{df['mark'].nunique()}")
    c3.metric("Avg Price (PLN)", f"{df['price'].mean():,.0f}")
    c4.metric("Avg Price (EUR)", f"{df['price_eur'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df[df["price"] < 300_000], x="price", nbins=80,
            labels={"price": "Price (PLN)"}, color_discrete_sequence=["#3b82f6"],
        )
        fig.update_layout(margin=dict(t=10, b=10), height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Listings by Fuel Type</div>', unsafe_allow_html=True)
        fuel_counts = df["fuel_label"].value_counts().reset_index()
        fuel_counts.columns = ["Fuel", "Count"]
        fig2 = px.pie(fuel_counts, names="Fuel", values="Count",
                      color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig2.update_layout(margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Average Price by Brand</div>', unsafe_allow_html=True)
        brand_avg = df.groupby("mark")["price"].mean().sort_values(ascending=False).reset_index()
        brand_avg.columns = ["Brand", "Avg Price (PLN)"]
        fig3 = px.bar(brand_avg, x="Avg Price (PLN)", y="Brand", orientation="h",
                      color="Avg Price (PLN)", color_continuous_scale="Blues")
        fig3.update_layout(margin=dict(t=10, b=10), height=450, showlegend=False,
                           coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Price vs Year (median by year)</div>', unsafe_allow_html=True)
        yr_med = df.groupby("year")["price"].median().reset_index()
        yr_med.columns = ["Year", "Median Price (PLN)"]
        fig4 = px.line(yr_med, x="Year", y="Median Price (PLN)", markers=True,
                       color_discrete_sequence=["#3b82f6"])
        fig4.update_layout(margin=dict(t=10, b=10), height=450)
        st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PRICE PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
def page_predictor(df, trained, encoders, scaler, results):
    st.title("🔍 Car Price Predictor")
    st.markdown("Fill in the car details below and get an instant market price estimate.")

    marks = sorted(df["mark"].unique().tolist())
    fuels = sorted(df["fuel"].unique().tolist())
    provinces = sorted(df["province"].unique().tolist())

    with st.form("predict_form"):
        st.markdown('<div class="section-header">Car Specifications</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            mark = st.selectbox("Brand", marks, index=marks.index("bmw") if "bmw" in marks else 0)
            year = st.slider("Year", 1995, 2022, 2018)
        with c2:
            fuel = st.selectbox("Fuel Type", fuels, index=fuels.index("gasoline") if "gasoline" in fuels else 0)
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=500_000,
                                      value=80_000, step=1_000)
        with c3:
            province = st.selectbox("Province", provinces)
            vol_engine = st.number_input("Engine Volume (cc)", min_value=500, max_value=8_000,
                                         value=2000, step=50)

        st.markdown('<div class="section-header">Select Models to Compare</div>', unsafe_allow_html=True)
        model_choices = st.multiselect(
            "Models", list(MODELS.keys()), default=["XGBoost", "Random Forest"]
        )

        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    if submitted:
        if not model_choices:
            st.warning("Please select at least one model.")
            return

        predictions = {}
        for mname in model_choices:
            p = predict_price(
                trained[mname], encoders, scaler, mname,
                year, mileage, vol_engine, mark, fuel, province,
            )
            predictions[mname] = p

        avg_pred = np.mean(list(predictions.values()))

        st.markdown(f"""
        <div class="pred-box">
          <p>Ensemble Estimate</p>
          <h2>{avg_pred:,.0f} PLN</h2>
          <p>≈ {avg_pred * PLN_TO_EUR:,.0f} EUR</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Model Breakdown</div>', unsafe_allow_html=True)
        cols = st.columns(len(predictions))
        for idx, (mname, pred) in enumerate(predictions.items()):
            r2 = results[mname]["R²"]
            cols[idx].metric(
                label=mname,
                value=f"{pred:,.0f} PLN",
                delta=f"R²={r2:.3f}",
            )

        # Bar chart comparison
        fig = go.Figure(go.Bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            marker_color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"][: len(predictions)],
            text=[f"{v:,.0f}" for v in predictions.values()],
            textposition="outside",
        ))
        fig.update_layout(
            title="Predicted Prices by Model",
            yaxis_title="Price (PLN)",
            height=380,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Similar cars from dataset
        st.markdown('<div class="section-header">Similar Listings in Dataset</div>', unsafe_allow_html=True)
        sim = df[
            (df["mark"] == mark) &
            (df["fuel"] == fuel) &
            (df["year"].between(year - 2, year + 2)) &
            (df["mileage"].between(max(0, mileage - 30_000), mileage + 30_000))
        ][["mark", "model", "year", "mileage", "vol_engine", "fuel", "province", "price"]].head(10)

        if len(sim):
            st.dataframe(sim.style.format({"price": "{:,.0f}", "mileage": "{:,.0f}"}),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No closely matching listings found for the selected criteria.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3 — DATA EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
def page_explorer(df: pd.DataFrame):
    st.title("📊 Data Explorer")
    st.markdown("Filter and explore the raw dataset interactively.")

    with st.expander("🔧 Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_marks = st.multiselect("Brand", sorted(df["mark"].unique()), default=[])
        with c2:
            sel_fuels = st.multiselect("Fuel", sorted(df["fuel"].unique()), default=[])
        with c3:
            sel_provinces = st.multiselect("Province", sorted(df["province"].unique()), default=[])

        c4, c5 = st.columns(2)
        with c4:
            yr_range = st.slider("Year Range", 1990, 2022, (2010, 2022))
        with c5:
            price_range = st.slider("Price Range (PLN)", 1_000, 500_000, (5_000, 300_000), step=1_000)

    mask = (
        df["year"].between(*yr_range) &
        df["price"].between(*price_range)
    )
    if sel_marks:
        mask &= df["mark"].isin(sel_marks)
    if sel_fuels:
        mask &= df["fuel"].isin(sel_fuels)
    if sel_provinces:
        mask &= df["province"].isin(sel_provinces)

    filtered = df[mask]
    st.markdown(f"**{len(filtered):,} listings** match your filters.")

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings", f"{len(filtered):,}")
    c2.metric("Median Price", f"{filtered['price'].median():,.0f} PLN")
    c3.metric("Min Price", f"{filtered['price'].min():,.0f} PLN")
    c4.metric("Max Price", f"{filtered['price'].max():,.0f} PLN")

    tab1, tab2, tab3 = st.tabs(["📋 Table", "📦 Box Plots", "🔥 Heatmap"])

    with tab1:
        show_cols = ["mark", "model", "year", "mileage", "vol_engine",
                     "fuel", "province", "price"]
        st.dataframe(
            filtered[show_cols]
            .sort_values("price", ascending=False)
            .head(500)
            .style.format({"price": "{:,.0f}", "mileage": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
            height=450,
        )
        st.caption("Showing top 500 by price.")

    with tab2:
        if len(filtered) > 0:
            fig = px.box(
                filtered[filtered["price"] < 300_000],
                x="mark", y="price",
                color="mark",
                labels={"price": "Price (PLN)", "mark": "Brand"},
                title="Price Distribution by Brand",
            )
            fig.update_layout(height=450, showlegend=False,
                              xaxis=dict(tickangle=-30))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if len(filtered) > 20:
            hm_data = (
                filtered.groupby(["mark", "fuel"])["price"]
                .median()
                .reset_index()
                .pivot(index="mark", columns="fuel", values="price")
            )
            fig = px.imshow(
                hm_data,
                color_continuous_scale="Blues",
                labels=dict(color="Median Price (PLN)"),
                title="Median Price Heatmap: Brand × Fuel",
                aspect="auto",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
def page_model_perf(results, fi):
    st.title("🤖 Model Performance")
    st.markdown("Evaluation metrics on a held-out **20 % test set**.")

    # Metrics table
    rows = []
    for mname, r in results.items():
        rows.append({
            "Model": mname,
            "MAE (PLN)": f"{r['MAE']:,.0f}",
            "RMSE (PLN)": f"{r['RMSE']:,.0f}",
            "R²": f"{r['R²']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Feature Importance"])

    with tab1:
        sel = st.selectbox("Select Model", list(results.keys()), key="avp")
        preds = results[sel]["predictions"]
        acts = results[sel]["actuals"]
        sample = min(3000, len(preds))
        idx = np.random.choice(len(preds), sample, replace=False)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=acts[idx], y=preds[idx], mode="markers",
            marker=dict(color="#3b82f6", opacity=0.35, size=4),
            name="Predictions",
        ))
        mn, mx = int(acts.min()), int(acts.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="#ef4444", dash="dash"), name="Perfect Fit",
        ))
        fig.update_layout(
            title=f"{sel} — Actual vs Predicted",
            xaxis_title="Actual Price (PLN)",
            yaxis_title="Predicted Price (PLN)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sel2 = st.selectbox("Select Model", list(results.keys()), key="res")
        preds2 = results[sel2]["predictions"]
        acts2 = results[sel2]["actuals"]
        residuals = acts2 - preds2
        sample2 = min(3000, len(preds2))
        idx2 = np.random.choice(len(preds2), sample2, replace=False)

        fig2 = px.scatter(
            x=preds2[idx2], y=residuals[idx2],
            labels={"x": "Predicted Price (PLN)", "y": "Residual (PLN)"},
            title=f"{sel2} — Residual Plot",
            color_discrete_sequence=["#10b981"],
            opacity=0.35,
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(height=430)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fi_df = pd.DataFrame(
            {"Feature": list(fi.keys()), "Importance": list(fi.values())}
        ).sort_values("Importance", ascending=True)
        label_map = {
            "year": "Year", "mileage": "Mileage", "vol_engine": "Engine Volume",
            "age": "Car Age", "mark_enc": "Brand", "fuel_enc": "Fuel Type",
            "province_enc": "Province",
        }
        fi_df["Feature"] = fi_df["Feature"].map(label_map)
        fig3 = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            title="XGBoost Feature Importances",
        )
        fig3.update_layout(height=380, coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # R² comparison bar
    st.markdown('<div class="section-header">R² Score Comparison</div>', unsafe_allow_html=True)
    r2_df = pd.DataFrame({
        "Model": list(results.keys()),
        "R²": [results[m]["R²"] for m in results],
    })
    fig4 = px.bar(r2_df, x="Model", y="R²", color="R²",
                  color_continuous_scale="Blues", text_auto=".4f")
    fig4.update_layout(height=320, coloraxis_showscale=False, showlegend=False,
                       yaxis_range=[0, 1])
    st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 5 — MARKET INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────
def page_insights(df: pd.DataFrame):
    st.title("📈 Market Insights")
    st.markdown("Deep-dive analytics into the Polish used-car market.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Price by Province", "Depreciation", "Fuel Trends", "Brand Deep Dive"
    ])

    with tab1:
        prov_med = df.groupby("province")["price"].median().sort_values(ascending=False).reset_index()
        prov_med.columns = ["Province", "Median Price (PLN)"]
        fig = px.bar(prov_med, x="Median Price (PLN)", y="Province", orientation="h",
                     color="Median Price (PLN)", color_continuous_scale="Blues",
                     title="Median Car Price by Province")
        fig.update_layout(height=520, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sel_brand = st.selectbox("Pick a Brand", sorted(df["mark"].unique()), key="dep_brand",
                                 index=list(sorted(df["mark"].unique())).index("bmw")
                                 if "bmw" in df["mark"].unique() else 0)
        dep = (
            df[df["mark"] == sel_brand]
            .groupby("age")["price"]
            .median()
            .reset_index()
        )
        dep.columns = ["Car Age (years)", "Median Price (PLN)"]
        dep = dep[dep["Car Age (years)"] <= 25]
        fig2 = px.line(dep, x="Car Age (years)", y="Median Price (PLN)", markers=True,
                       title=f"{sel_brand.title()} — Price Depreciation Over Age",
                       color_discrete_sequence=["#3b82f6"])
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fuel_yr = (
            df.groupby(["year", "fuel_label"])["price"]
            .median()
            .reset_index()
        )
        fuel_yr.columns = ["Year", "Fuel", "Median Price (PLN)"]
        fuel_yr = fuel_yr[fuel_yr["Year"] >= 2010]
        fig3 = px.line(fuel_yr, x="Year", y="Median Price (PLN)", color="Fuel",
                       markers=True, title="Median Price by Fuel Type Over the Years")
        fig3.update_layout(height=430)
        st.plotly_chart(fig3, use_container_width=True)

        # Volume
        fuel_vol = df.groupby(["year", "fuel_label"]).size().reset_index(name="Listings")
        fuel_vol = fuel_vol[fuel_vol["year"] >= 2010]
        fig4 = px.bar(fuel_vol, x="year", y="Listings", color="Fuel",
                      title="Listing Volume by Fuel Type Per Year", barmode="stack")
        fig4.update_layout(height=380)
        st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        brand_b = st.selectbox("Select Brand", sorted(df["mark"].unique()), key="brand_dd")
        bdf = df[df["mark"] == brand_b]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Listings", f"{len(bdf):,}")
        c2.metric("Median Price", f"{bdf['price'].median():,.0f} PLN")
        c3.metric("Models", f"{bdf['model'].nunique()}")
        c4.metric("Avg Age (yrs)", f"{bdf['age'].mean():.1f}")

        col1, col2 = st.columns(2)
        with col1:
            top_models = bdf.groupby("model")["price"].median().sort_values(ascending=False).head(10)
            fig5 = px.bar(x=top_models.values, y=top_models.index,
                          orientation="h", labels={"x": "Median Price (PLN)", "y": "Model"},
                          title=f"Top {brand_b.title()} Models by Price",
                          color=top_models.values, color_continuous_scale="Blues")
            fig5.update_layout(height=380, coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            fuel_dist = bdf["fuel_label"].value_counts().reset_index()
            fuel_dist.columns = ["Fuel", "Count"]
            fig6 = px.pie(fuel_dist, names="Fuel", values="Count",
                          title=f"{brand_b.title()} — Fuel Mix",
                          hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
            fig6.update_layout(height=380)
            st.plotly_chart(fig6, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    df = load_data()
    trained, encoders, scaler, results, fi, X_test, y_test = train_models(df)

    # Navigation first so it appears at the top of the sidebar
    page = sidebar_nav()

    # Sidebar stats appear below the nav
    with st.sidebar:
        best_model = max(results, key=lambda k: results[k]["R²"])
        st.metric("Best Model", best_model)
        st.metric("Best R²", f"{results[best_model]['R²']:.4f}")
        st.metric("Best MAE", f"{results[best_model]['MAE']:,.0f} PLN")
        st.markdown("---")
        st.caption("Data: Polish used-car listings (OLX/Otomoto)")

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "🔍 Price Predictor":
        page_predictor(df, trained, encoders, scaler, results)
    elif page == "📊 Data Explorer":
        page_explorer(df)
    elif page == "🤖 Model Performance":
        page_model_perf(results, fi)
    elif page == "📈 Market Insights":
        page_insights(df)


if __name__ == "__main__":
    main()
