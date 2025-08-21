# app.py — Hardened Streamlit app for Malaria Prediction (DHS-style)
# - Works even if PyTorch isn't available (hides Transformer options, keeps Ridge baseline)
# - Robust parsing of pfir_YYYY (target), pYYYY_M (rainfall), tYYYY_M (temperature)
# - Clear error messages + guardrails to avoid common Streamlit Cloud issues
#
# Files expected (you add them under ./data in your repo):
#   data/MalariaRate.csv       (has pfir_YYYY columns)
#   data/Rainfall.csv          (has pYYYY_M columns)
#   data/temperature.csv       (has tYYYY_M columns)

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

# Try import torch; if unavailable (Streamlit Cloud), continue without Transformers
TORCH_OK = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as _e:
    TORCH_OK = False

st.set_page_config(page_title="Malaria (DHS) — Predict", page_icon="🧬", layout="wide")

# ---------------- Patterns & helpers ----------------

PFIR_PAT = re.compile(r"^pfir_(\d{4})$")
RAIN_PAT = re.compile(r"^p(\d{4})_(\d{1,2})$")
TEMP_PAT = re.compile(r"^t(\d{4})_(\d{1,2})$")

@st.cache_data(show_spinner=True)
def load_csv(path):
    # more tolerant CSV loader
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def melt_pfir(df):
    cols = [c for c in df.columns if PFIR_PAT.match(str(c))]
    if not cols:
        return None, "No malaria columns matching pfir_YYYY found."
    long = df.melt(
        id_vars=[c for c in df.columns if c not in cols],
        value_vars=cols, var_name="pfir_col", value_name="malaria_rate"
    )
    years = long["pfir_col"].str.extract(r"pfir_(\d{4})")
    if years.isnull().all().bool():
        return None, "Could not extract year from pfir columns."
    long["year"] = pd.to_numeric(years[0], errors="coerce").astype("Int64")
    long = long.drop(columns=["pfir_col"]).dropna(subset=["year"])
    long["year"] = long["year"].astype(int)
    return long, None

def melt_indicator(df, pat, value_name):
    cols = [c for c in df.columns if pat.match(str(c))]
    if not cols:
        return None
    long = df.melt(
        id_vars=[c for c in df.columns if c not in cols],
        value_vars=cols, var_name="col", value_name=value_name
    )
    # Extract year & month from names like p2017_1 or t2016_12
    m = long["col"].str.extract(r"[pt]?(\d{4})_(\d{1,2})")
    long["year"] = pd.to_numeric(m[0], errors="coerce").astype("Int64")
    long["month"] = pd.to_numeric(m[1], errors="coerce").astype("Int64")
    long = long.drop(columns=["col"]).dropna(subset=["year","month"])
    long["year"] = long["year"].astype(int)
    long["month"] = long["month"].astype(int)
    return long

def annualize(df, value_name, how="mean"):
    if df is None or value_name not in df.columns or "year" not in df.columns:
        return None
    keys = [c for c in df.columns if c not in [value_name, "year", "month"]]
    if how == "sum":
        agg = df.groupby(keys+["year"], dropna=False)[value_name].sum().reset_index()
    else:
        agg = df.groupby(keys+["year"], dropna=False)[value_name].mean().reset_index()
    return agg

def common_key_candidates(df):
    prefs = ["dhsid","country","dhsregna","dhsregco","latnum","longnum","URBAN_RURA"]
    return [c for c in prefs if c in df.columns]

# ---------------- Optional Transformers ----------------
if TORCH_OK:
    class FTTransformerRegressor(nn.Module):
        def __init__(self, n_features: int, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
            super().__init__()
            self.n_features = n_features
            self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
            self.col_emb = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        def forward(self, x):
            toks = [self.proj[j](x[:, j:j+1]) + self.col_emb[j:j+1] for j in range(self.n_features)]
            h = self.encoder(torch.stack(toks, dim=1)).mean(dim=1)
            return self.head(h).squeeze(-1)

    class TabTransformerLiteRegressor(nn.Module):
        def __init__(self, n_features: int, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
            super().__init__()
            self.n_features = n_features
            self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
            self.col_emb = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
            self.cls = nn.Parameter(torch.zeros(1,1,d_model))
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        def forward(self, x):
            toks = [self.proj[j](x[:, j:j+1]) + self.col_emb[j:j+1] for j in range(self.n_features)]
            seq = torch.cat([self.cls.expand(x.size(0),-1,-1), torch.stack(toks, dim=1)], dim=1)
            h = self.encoder(seq)[:,0,:]
            return self.head(h).squeeze(-1)

    def fit_torch(model, X_train, y_train, X_val, y_val, epochs=120, lr=1e-3, batch=128):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        tr = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32)), batch_size=batch, shuffle=True)
        va = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(y_val, dtype=torch.float32)), batch_size=batch, shuffle=False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best, best_v, patience, wait = None, float("inf"), 15, 0
        for ep in range(epochs):
            model.train()
            for xb, yb in tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            # val
            model.eval(); v = 0.0
            with torch.no_grad():
                for xb, yb in va:
                    xb, yb = xb.to(device), yb.to(device)
                    v += loss_fn(model(xb), yb).item() * xb.size(0)
            v /= len(va.dataset)
            if v < best_v - 1e-6:
                best_v, best, wait = v, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience: break
        if best is not None: model.load_state_dict(best)
        return model

    def predict_torch(model, X):
        device = next(model.parameters()).device
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()

# ---------------- UI ----------------

st.title("🧬 Malaria Prediction (DHS-style)")
st.caption("Upload or commit your CSVs to ./data. The app handles wide-to-long, annualizes pYYYY_M/tYYYY_M, merges by keys+year, and evaluates models.")

st.sidebar.header("1) Data files")
malaria_path = st.sidebar.text_input("Malaria CSV (pfir_YYYY)", value="data/MalariaRate.csv")
rain_path = st.sidebar.text_input("Rainfall CSV (pYYYY_M)", value="data/Rainfall.csv")
temp_path = st.sidebar.text_input("Temperature CSV (tYYYY_M)", value="data/temperature.csv")

agg_method = st.sidebar.selectbox("Annualize monthly indicators by", options=["mean","sum"], index=0)

# Load
df_mal = df_rain = df_temp = None
mal_err = None
try:
    df_mal = load_csv(malaria_path)
except Exception as e:
    mal_err = str(e)

rain_warn = temp_warn = None
try:
    df_rain = load_csv(rain_path)
except Exception as e:
    rain_warn = str(e)
try:
    df_temp = load_csv(temp_path)
except Exception as e:
    temp_warn = str(e)

if mal_err:
    st.error(f"Malaria CSV error: {mal_err}")
    st.stop()
if rain_warn: st.sidebar.warning(f"Rainfall CSV not loaded: {rain_warn}")
if temp_warn: st.sidebar.warning(f"Temperature CSV not loaded: {temp_warn}")

with st.expander("File previews", expanded=False):
    st.write("**Malaria**", df_mal.head())
    if df_rain is not None: st.write("**Rainfall**", df_rain.head())
    if df_temp is not None: st.write("**Temperature**", df_temp.head())

# Melt to long
mal_long, mal_long_err = melt_pfir(df_mal)
if mal_long_err:
    st.error(mal_long_err + " Make sure your malaria file has columns like pfir_2017."); st.stop()

rain_ann = annualize(melt_indicator(df_rain, RAIN_PAT, "rainfall"), "rainfall", agg_method) if df_rain is not None else None
temp_ann = annualize(melt_indicator(df_temp, TEMP_PAT, "temperature"), "temperature", agg_method) if df_temp is not None else None

# Merge keys
st.sidebar.header("2) Merge keys")
key_cands = common_key_candidates(mal_long)
if rain_ann is not None: key_cands = [k for k in key_cands if k in rain_ann.columns]
if temp_ann is not None: key_cands = [k for k in key_cands if k in temp_ann.columns]

default_keys = [k for k in ["dhsid","country","dhsregna"] if k in mal_long.columns]
if not default_keys and "dhsid" in mal_long.columns: default_keys = ["dhsid"]
keys = st.sidebar.multiselect("Common keys (used with 'year')", options=sorted(list(set(key_cands))), default=default_keys)
keys = list(dict.fromkeys(keys))

# Merge frames
merged = mal_long.copy()
if rain_ann is not None:
    merged = pd.merge(merged, rain_ann, on=keys+["year"], how="left")
if temp_ann is not None:
    merged = pd.merge(merged, temp_ann, on=keys+["year"], how="left")

if merged.empty:
    st.error("Merged dataset is empty. Check selected keys or file compatibility."); st.stop()

with st.expander("Merged sample (year-level)", expanded=False):
    st.dataframe(merged.head(30), use_container_width=True)

# Indicators
st.sidebar.header("3) Indicators")
available_inds = [c for c in ["rainfall","temperature"] if c in merged.columns]
if not available_inds:
    st.error("No indicator columns available after merge. Ensure Rainfall and/or Temperature files are present with pYYYY_M/tYYYY_M columns.")
    st.stop()
indicators = st.sidebar.multiselect("Choose up to 3 indicators", options=available_inds, default=available_inds[:min(2, len(available_inds))])
if len(indicators) == 0:
    st.error("Pick at least one indicator."); st.stop()
if len(indicators) > 3:
    indicators = indicators[:3]

use_log = st.sidebar.checkbox("Apply log1p() to malaria_rate (invert for reporting)", value=False)

# Split
st.sidebar.header("4) Validation")
split_mode = st.sidebar.radio("Split mode", ["Random 80/20","By Year"], index=1)
train_end_year = st.sidebar.number_input("Train ≤ year", value=2016, step=1)
test_year = st.sidebar.number_input("Test year", value=2017, step=1)
random_state = st.sidebar.number_input("Random state (for 80/20)", min_value=0, value=42, step=1)

# Prepare matrices
df = merged.dropna(subset=["malaria_rate"]).copy()
if df.empty:
    st.error("No rows with malaria_rate after merge. Check files/keys."); st.stop()

X_raw = df[indicators].astype(np.float32).values
y_raw = df["malaria_rate"].astype(np.float32).values
y = np.log1p(np.clip(y_raw, 0, None)) if use_log else y_raw
invert = (np.expm1 if use_log else None)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_proc = scaler.fit_transform(imputer.fit_transform(X_raw))

if split_mode == "Random 80/20":
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X_proc, y, y_raw, test_size=0.2, random_state=int(random_state)
    )
else:
    train_mask = df["year"] <= int(train_end_year)
    test_mask  = df["year"] == int(test_year)
    train_df, test_df = df[train_mask].copy(), df[test_mask].copy()
    if train_df.empty or test_df.empty:
        st.error("Train or test set empty for selected years. Adjust years."); st.stop()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[indicators].astype(np.float32).values))
    y_train = np.log1p(np.clip(train_df["malaria_rate"].astype(np.float32).values, 0, None)) if use_log else train_df["malaria_rate"].astype(np.float32).values
    X_test  = scaler.transform(imputer.transform(test_df[indicators].astype(np.float32).values))
    y_test  = np.log1p(np.clip(test_df["malaria_rate"].astype(np.float32).values, 0, None)) if use_log else test_df["malaria_rate"].astype(np.float32).values
    y_raw_test = test_df["malaria_rate"].astype(np.float32).values

# Model selection
st.sidebar.header("5) Model")
model_opts = ["Ridge (Baseline)"]
if TORCH_OK:
    model_opts = ["FT-Transformer","TabTransformer-Lite"] + model_opts
model_name = st.sidebar.selectbox("Model", model_opts, index=0)

# Torch hyperparams only if torch present & transformer selected
if TORCH_OK and model_name in ["FT-Transformer","TabTransformer-Lite"]:
    epochs = st.sidebar.slider("Epochs", 20, 400, 120, 10)
    d_model = st.sidebar.selectbox("d_model", [32,64,128], index=1)
    n_heads = st.sidebar.selectbox("n_heads", [2,4,8], index=1)
    n_layers = st.sidebar.selectbox("layers", [1,2,3,4], index=1)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    lr = st.sidebar.selectbox("Learning rate", [1e-4,3e-4,1e-3,3e-3], index=2)
    batch = st.sidebar.selectbox("Batch size", [32,64,128,256], index=2)
else:
    epochs=d_model=n_heads=n_layers=dropout=lr=batch=None

# Train & predict
st.subheader("Results")
start = time.time()
if model_name == "Ridge (Baseline)":
    model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()),
                      ("ridge", Ridge(alpha=1.0, random_state=42))])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
else:
    n_features = X_train.shape[1]
    if model_name == "FT-Transformer":
        model = FTTransformerRegressor(n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    else:
        model = TabTransformerLiteRegressor(n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers+1, dropout=dropout)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = fit_torch(model, X_tr, y_tr, X_val, y_val, epochs=epochs, lr=lr, batch=batch)
    y_pred = predict_torch(model, X_test)

elapsed = time.time() - start

# Metrics
def eval_r2_mae(y_true, y_pred, y_true_raw, invert):
    y_pred_raw = invert(y_pred) if invert is not None else y_pred
    y_eval = y_true_raw if invert is not None else y_true
    return r2_score(y_eval, y_pred_raw), mean_absolute_error(y_eval, y_pred_raw), y_pred_raw

r2, mae, y_pred_raw = eval_r2_mae(y_test, y_pred, y_raw_test, invert)

c1, c2, c3 = st.columns(3)
c1.metric("R² (test)", f"{r2:.4f}")
c2.metric("MAE (test)", f"{mae:.4f}")
c3.metric("Time", f"{elapsed:.1f} s")

# Plot
import matplotlib.pyplot as plt
fig = plt.figure()
plt.scatter(y_raw_test, y_pred_raw, alpha=0.65)
mn, mx = float(np.min(y_raw_test)), float(np.max(y_raw_test))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("True malaria_rate")
plt.ylabel("Predicted malaria_rate")
plt.title("Predicted vs True (test)")
st.pyplot(fig, use_container_width=True)

# Debug pane
with st.expander("Selections & Debug"):
    st.write("**Keys used:**", keys)
    st.write("**Indicators:**", indicators, f"(annual={agg_method})")
    st.write("**Split:**", split_mode, f"train≤{train_end_year}, test={test_year}, random_state={random_state}")
    st.write("**Model:**", model_name)
    st.write("**Torch available:**", TORCH_OK)
    st.write("**Shapes:** train", X_train.shape, "| test", X_test.shape)