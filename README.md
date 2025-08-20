# Malaria Prediction (DHS-style) — Streamlit + Transformers

Understands these column patterns:
- **Target**: `pfir_YYYY`
- **Rainfall**: `pYYYY_M` (monthly; annualized by mean or sum)
- **Temperature**: `tYYYY_M` (monthly; annualized by mean or sum)

Automatically converts wide → long (by `year`), merges on keys (default: `dhsid`, `country`, `dhsregna`), and supports:
- Random **80/20** split
- **By-year** split (default train ≤ 2016, test = 2017)

Models:
- FT-Transformer (tabular transformer)
- TabTransformer-Lite (CLS-token variant)
- Ridge baseline

## Layout
```
├─ app.py
├─ requirements.txt
└─ data/
   ├─ MalariaRate.csv
   ├─ Rainfall.csv
   └─ temperature.csv
```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```