---
title: SpVAR Colombia
emoji: 🇨🇴
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# SpVAR Colombia

Laboratorio de economía espacial departamental.

**ESDA (Moran + LISA) · Spatial VAR · Pronóstico 10 años**

- 32 departamentos + Bogotá D.C.
- Geometría DANE MGN 2018
- Pipeline: descriptivas, mapas, Moran, LISA, estimación SpVAR, IRF y pronóstico

## Deploy (Streamlit Cloud)

1. Sube este repo a GitHub (público).
2. Entra a https://share.streamlit.io
3. **New app** → elige el repo → Main file: `app.py`
4. Deploy.

## Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota

La primera carga puede tardar 1–3 minutos (descarga GeoJSON + ESDA + estimación).
Las siguientes usan caché de Streamlit.
