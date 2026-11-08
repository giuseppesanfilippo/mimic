# Mimic GUI — Python / Streamlit

Port of the [Mimic GUI](https://sanfilippo.shinyapps.io/mimic/) from R/Shiny to Python/Streamlit.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app exposes the same main controls: `N₁`, `N₂`, `a₁/N`, `a₂/N`, `pL(N)`, `pU(N)`, `c₁`, `c₂`, curve selection, and PDF/CDF plots. The implementation follows the finite exchangeability inversion and the four reference completions described in Lad & Sanfilippo (2020).
