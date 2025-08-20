# app.py — DHS-aware Streamlit app for Malaria Prediction
# Handles wide-to-long for:
#   - Target: columns like pfir_YYYY  (e.g., pfir_2017)
#   - Rainfall: columns like pYYYY_MM (e.g., p2017_1 ... p2017_12)
#   - Temperature: columns like tYYYY_MM
# Merges by location keys (e.g., dhsid) + year, supports Random 80/20 and By-Year (train ≤2016, test=2017).

import os
import re
import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

st.set_page_config(page_title="Malaria (DHS) — Transformers", page_icon="🧬", layout="wide")

# ---------------- Pattern helpers ----------------

PFIR_PAT = re.compile(r"^pfir_(\d{4})$")
RAIN_PAT = re.compile(r"^p(\d{4})_(\d{1,2})$")
TEMP_PAT = re.compile(r"^t(\d{4})_(\d{1,2})$")

def melt_pfir(df):
    cols = [c for c in df.columns if PFIR_PAT.match(str(c))]
    if not cols:
        return None
    df_m = df.copy()
    long = df_m.melt(
        id_vars=[c for c in df.columns if c not in cols],
        value_vars=cols,
        var_name="pfir_col",
        value_name="malaria_rate"
    )
    long["year"] = long["pfir_col"].str.extract(r"pfir_(\d{4})").astype(int)
    long = long.drop(columns=["pfir_col"])
    return long

def melt_indicator(df, pat, value_name):
    cols = [c for c in df.columns if pat.match(str(c))]
    if not cols:
        return None
    df_m = df.copy()
    long = df_m.melt(
        id_vars=[c for c in df.columns if c not in cols],
        value_vars=cols,
        var_name="col",
        value_name=value_name
    )
    # Extract year, month
    yrmo = long["col"].str.extract(r"([pt])(\d{4})_(\d{1,2})")  # pYYYY_M or tYYYY_M
    # Some columns might be missing the leading letter depending on export; fallback:
    # Try to extract any (\d{4})_(\d{1,2})
    fallback = long["col"].str.extract(r"(\d{4})_(\d{1,2})")
    year = pd.to_numeric(yrmo[1].fillna(fallback[0]), errors="coerce")
    month = pd.to_numeric(yrmo[2].fillna(fallback[1]), errors="coerce")
    long["year"] = year.astype("Int64")
    long["month"] = month.astype("Int64")
    long = long.drop(columns=["col"])
    long = long.dropna(subset=["year","month"])
    long["year"] = long["year"].astype(int)
    long["month"] = long["month"].astype(int)
    return long

def annualize(df, value_name, how="mean"):
    if df is None or value_name not in df.columns or "year" not in df.columns:
        return None
    keys = [c for c in df.columns if c not in [value_name, "year", "month"]]
    agg = {value_name: ("mean" if how=="mean" else "sum")}
    if how == "mean":
        grp = df.groupby(keys + ["year"], dropna=False)[value_name].mean().reset_index()
    else:
        grp = df.groupby(keys + ["year"], dropna=False)[value_name].sum().reset_index()
    return grp

def common_key_candidates(df):
    prefs = ["dhsid","country","dhsregna","dhsregco","latnum","longnum","URBAN_RURA"]
    return [c for c in prefs if c in df.columns]

# ---------------- Models ----------------

class FTTransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
        self.col_emb = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        toks = [self.proj[j](x[:, j:j+1]) + self.col_emb[j:j+1] for j in range(self.n_features)]
        seq = torch.stack(toks, dim=1)
        h = self.encoder(seq).mean(dim=1)
        return self.head(h).squeeze(-1)

class TabTransformerLiteRegressor(nn.Module):
    def __init__(self, n_features: int, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
        self.col_emb = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        toks = [self.proj[j](x[:, j:j+1]) + self.col_emb[j:j+1] for j in range(self.n_features)]
        tok = torch.stack(toks, dim=1)
        cls = self.cls.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        h = self.encoder(seq)
        return self.head(h[:,0,:]).squeeze(-1)

def fit_torch(model, X_train, y_train, X_val, y_val, epochs=150, lr=1e-3, batch=128, use_cuda=True):
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    tr = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32)),
                    batch_size=batch, shuffle=True)
    va = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                  torch.tensor(y_val, dtype=torch.float32)),
                    batch_size=batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_v, patience, since = None, float("inf"), 15, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval(); v = 0.0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                v += loss_fn(model(xb), yb).item() * xb.size(0)
        v /= len(va.dataset)
        if v < best_v - 1e-6:
            best_v, best = v, {k:v.cpu().clone() for k,v in model.state_dict().items()}; since = 0
        else:
            since += 1
            if since >= patience: break
    if best is not None: model.load_state_dict(best)
    return model

def predict_torch(model, X):
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()

# ---------------- UI ----------------

st.title("🧬 Malaria Prediction (DHS-style data)")
st.caption("Understands pfir_YYYY (target), pYYYY_M (rainfall) and tYYYY_M (temperature). Converts to long-by-year, aggregates monthly indicators, and evaluates Transformer models.")

st.sidebar.header("1) Data files")
malaria_path = st.sidebar.text_input("Malaria CSV (pfir_YYYY)", value="data/MalariaRate.csv")
rain_path = st.sidebar.text_input("Rainfall CSV (pYYYY_M)", value="data/Rainfall.csv")
temp_path = st.sidebar.text_input("Temperature CSV (tYYYY_M)", value="data/temperature.csv")

agg_method = st.sidebar.selectbox("Annualize monthly indicators by", options=["mean","sum"], index=0)

@st.cache_data(show_spinner=True)
def load_csv(path):
    return pd.read_csv(path)

df_mal, df_rain, df_temp = None, None, None
try:
    df_mal = load_csv(malaria_path); st.sidebar.success("Loaded malaria file")
except Exception as e:
    st.sidebar.error(f"Malaria CSV error: {e}")
try:
    df_rain = load_csv(rain_path); st.sidebar.success("Loaded rainfall file")
except Exception as e:
    st.sidebar.warning(f"Rainfall CSV not loaded: {e}")
try:
    df_temp = load_csv(temp_path); st.sidebar.success("Loaded temperature file")
except Exception as e:
    st.sidebar.warning(f"Temperature CSV not loaded: {e}")

with st.expander("File previews", expanded=False):
    if df_mal is not None: st.write("**Malaria**", df_mal.head())
    if df_rain is not None: st.write("**Rainfall**", df_rain.head())
    if df_temp is not None: st.write("**Temperature**", df_temp.head())

# Melt to long format
if df_mal is None: st.stop()
mal_long = melt_pfir(df_mal)
if mal_long is None:
    st.error("Could not find pfir_YYYY columns in malaria CSV."); st.stop()

rain_ann = annualize(melt_indicator(df_rain, RAIN_PAT, "rainfall"), "rainfall", agg_method) if df_rain is not None else None
temp_ann = annualize(melt_indicator(df_temp, TEMP_PAT, "temperature"), "temperature", agg_method) if df_temp is not None else None

# Pick merge keys
st.sidebar.header("2) Merge keys")
key_cands = common_key_candidates(mal_long)
if rain_ann is not None: key_cands = [k for k in key_cands if k in rain_ann.columns]
if temp_ann is not None: key_cands = [k for k in key_cands if k in temp_ann.columns]

default_keys = ["dhsid"] + [k for k in ["country","dhsregna"] if k in mal_long.columns]
keys = st.sidebar.multiselect("Common keys (plus 'year' is always used)", options=sorted(list(set(key_cands))), default=default_keys)
keys = list(dict.fromkeys(keys))  # dedupe

# Merge
left = mal_long
if rain_ann is not None:
    left = pd.merge(left, rain_ann, on=keys+["year"], how="left")
if temp_ann is not None:
    left = pd.merge(left, temp_ann, on=keys+["year"], how="left")

if left.empty:
    st.error("Merged dataset is empty. Check keys."); st.stop()

with st.expander("Merged sample (long by year)", expanded=False):
    st.dataframe(left.head(30), use_container_width=True)

# Feature selection
st.sidebar.header("3) Indicators")
available = [c for c in ["rainfall","temperature"] if c in left.columns]
indicators = st.sidebar.multiselect("Choose 1–3 indicators", options=available, default=available[:2])
if len(indicators) == 0: st.error("Pick at least one indicator."); st.stop()

use_log = st.sidebar.checkbox("Apply log1p() to malaria_rate (invert for reporting)", value=False)

# Split mode
st.sidebar.header("4) Validation")
split_mode = st.sidebar.radio("Split mode", ["Random 80/20","By Year"], index=1)
train_end_year = st.sidebar.number_input("Train ≤ year", value=2016, step=1)
test_year = st.sidebar.number_input("Test year", value=2017, step=1)

# Prepare matrices
df = left.dropna(subset=["malaria_rate"]).copy()
X = df[indicators].values.astype(np.float32)
y_raw = df["malaria_rate"].values.astype(np.float32)
if use_log:
    y = np.log1p(np.clip(y_raw, a_min=0, a_max=None))
    invert = np.expm1
else:
    y = y_raw
    invert = None

scaler = StandardScaler(); imputer = SimpleImputer(strategy="median")
X_proc = scaler.fit_transform(imputer.fit_transform(X))

if split_mode == "Random 80/20":
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(X_proc, y, y_raw, test_size=0.2, random_state=42)
else:
    train_mask = df["year"] <= int(train_end_year)
    test_mask  = df["year"] == int(test_year)
    train_df = df[train_mask].copy(); test_df = df[test_mask].copy()
    if train_df.empty or test_df.empty:
        st.error("Train or test set is empty for the selected years."); st.stop()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[indicators].values.astype(np.float32)))
    y_train = np.log1p(np.clip(train_df["malaria_rate"].values.astype(np.float32), 0, None)) if use_log else train_df["malaria_rate"].values.astype(np.float32)
    X_test  = scaler.transform(imputer.transform(test_df[indicators].values.astype(np.float32)))
    y_test  = np.log1p(np.clip(test_df["malaria_rate"].values.astype(np.float32), 0, None)) if use_log else test_df["malaria_rate"].values.astype(np.float32)
    y_raw_test = test_df["malaria_rate"].values.astype(np.float32)

# Model selection
st.sidebar.header("5) Model")
model_name = st.sidebar.selectbox("Transformer / Baseline", ["FT-Transformer","TabTransformer-Lite","Ridge (Baseline)"], index=0)
epochs = st.sidebar.slider("Epochs", 20, 500, 150, 10)
d_model = st.sidebar.selectbox("d_model", [32,64,128], index=1)
n_heads = st.sidebar.selectbox("n_heads", [2,4,8], index=1)
n_layers = st.sidebar.selectbox("layers", [1,2,3,4], index=1)
dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
lr = st.sidebar.selectbox("Learning rate", [1e-4,3e-4,1e-3,3e-3], index=2)
batch = st.sidebar.selectbox("Batch size", [32,64,128,256], index=2)

st.subheader("Results")
start = time.time()
if model_name == "Ridge (Baseline)":
    ridge = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
else:
    n_features = X_train.shape[1]
    if model_name == "FT-Transformer":
        model = FTTransformerRegressor(n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    else:
        model = TabTransformerLiteRegressor(n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers+1, dropout=dropout)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = fit_torch(model, X_tr, y_tr, X_val, y_val, epochs=epochs, lr=lr, batch=batch, use_cuda=True)
    y_pred = predict_torch(model, X_test)
elapsed = time.time() - start

def eval_r2_mae(y_true, y_pred, y_true_raw, invert):
    if invert is not None:
        y_pred_raw = invert(y_pred); y_eval = y_true_raw
    else:
        y_pred_raw = y_pred; y_eval = y_true
    return r2_score(y_eval, y_pred_raw), mean_absolute_error(y_eval, y_pred_raw), y_pred_raw

r2, mae, y_pred_raw = eval_r2_mae(y_test, y_pred, y_raw_test, invert)

c1, c2, c3 = st.columns(3)
c1.metric("R² (test)", f"{r2:.4f}")
c2.metric("MAE (test)", f"{mae:.4f}")
c3.metric("Time", f"{elapsed:.1f} s")

import matplotlib.pyplot as plt
fig = plt.figure()
plt.scatter(y_raw_test, y_pred_raw, alpha=0.65)
mn, mx = float(np.min(y_raw_test)), float(np.max(y_raw_test))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("True malaria_rate")
plt.ylabel("Predicted malaria_rate")
plt.title("Predicted vs True (test)")
st.pyplot(fig, use_container_width=True)

with st.expander("Selections & Debug"):
    st.write("**Keys:**", keys)
    st.write("**Indicators:**", indicators, f"(annual={agg_method})")
    st.write("**Split:**", split_mode, f"train≤{train_end_year}, test={test_year}")
    st.write("**Model:**", model_name, dict(epochs=epochs, d_model=d_model, heads=n_heads, layers=n_layers, dropout=dropout, lr=lr, batch=batch))
    st.write("**CUDA available:**", torch.cuda.is_available())
    st.write("**Shapes:** train", X_train.shape, "| test", X_test.shape)