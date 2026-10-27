# =============================================================================
# SpVAR COLOMBIA – Streamlit Cloud
# ESDA (Moran + LISA) + SpVAR + mapas + pronóstico
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns
from scipy import stats

from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error

try:
    from spreg.diagnostics_sp import LMtests
    HAS_LMTESTS = True
except ImportError:
    HAS_LMTESTS = False

from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import streamlit as st

np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

# -----------------------------------------------------------------------------
# Config página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpVAR Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; max-width: 1180px; }
      div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def ensure_nonneg(arr, eps=1e-8):
    a = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(a, eps)


def adf_test(series, name="serie"):
    s = pd.Series(series).dropna()
    if len(s) < 8:
        return {"variable": name, "adf_stat": np.nan, "pvalue": np.nan, "estacionaria": "insuficiente"}
    try:
        res = adfuller(s, autolag="AIC")
        return {
            "variable": name,
            "adf_stat": round(res[0], 4),
            "pvalue": round(res[1], 4),
            "estacionaria": "Si" if res[1] < 0.05 else "No",
        }
    except Exception:
        return {"variable": name, "adf_stat": np.nan, "pvalue": np.nan, "estacionaria": "error"}


def extract_model_table(model, labels: List[str]) -> pd.DataFrame:
    betas = np.asarray(model.betas).flatten()
    se = None
    for attr in ("std_err", "se"):
        if hasattr(model, attr):
            try:
                se = np.asarray(getattr(model, attr)).flatten()
                break
            except Exception:
                pass
    if se is None and hasattr(model, "vm"):
        try:
            se = np.sqrt(np.diag(np.asarray(model.vm)))
        except Exception:
            se = np.full_like(betas, np.nan)
    if se is None:
        se = np.full_like(betas, np.nan)
    n = min(len(betas), len(labels), len(se))
    rows = []
    for i in range(n):
        b, s = float(betas[i]), float(se[i])
        tval = b / s if s and np.isfinite(s) and s > 0 else np.nan
        pval = 2 * (1 - stats.norm.cdf(abs(tval))) if np.isfinite(tval) else np.nan
        rows.append({
            "parametro": labels[i] if i < len(labels) else f"b{i}",
            "coeficiente": round(b, 6),
            "ee": round(s, 6) if np.isfinite(s) else np.nan,
            "t": round(tval, 4) if np.isfinite(tval) else np.nan,
            "p_valor": round(pval, 4) if np.isfinite(pval) else np.nan,
        })
    return pd.DataFrame(rows)


def model_fit_stats(model) -> Dict[str, Any]:
    out = {}
    for key, attrs in {
        "r2": ("r2", "pr2", "pr2_e"),
        "aic": ("aic",),
        "n": ("n", "nobs"),
        "k": ("k",),
        "logll": ("logll", "logl"),
    }.items():
        for a in attrs:
            if hasattr(model, a):
                try:
                    out[key] = float(np.asarray(getattr(model, a)).item())
                    break
                except Exception:
                    pass
        if key not in out:
            out[key] = np.nan
    if hasattr(model, "rho") and model.rho is not None:
        try:
            out["rho"] = float(np.asarray(model.rho).item())
        except Exception:
            out["rho"] = np.nan
    else:
        out["rho"] = np.nan
    return out


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Pipeline completo (cacheado)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando geometría, ESDA y SpVAR… (1–3 min la primera vez)")
def run_pipeline():
    # --- Geometría ---
    GEOJSON_URLS = [
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
        "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json",
    ]
    gdf = None
    for url in GEOJSON_URLS:
        try:
            gdf = gpd.read_file(url)
            break
        except Exception:
            continue
    if gdf is None:
        raise RuntimeError("No se pudo cargar GeoJSON de Colombia")

    col_code = col_name = None
    for c in gdf.columns:
        cl = str(c).lower()
        if "ccdgo" in cl or ("cod" in cl and "dpto" in cl):
            col_code = c
        if "cnmbr" in cl or "nombre" in cl or "name" in cl or "depart" in cl:
            col_name = c
    if col_code is None:
        if "DPTO_CCDGO" in gdf.columns:
            col_code, col_name = "DPTO_CCDGO", "DPTO_CNMBR"
        elif "dpto_ccdgo" in gdf.columns:
            col_code, col_name = "dpto_ccdgo", "dpto_cnmbr"
        else:
            raise ValueError("Columnas departamento no encontradas")

    gdf = gdf[[col_code, col_name, "geometry"]].copy()
    gdf.columns = ["cod_dpto", "nombre", "geometry"]
    gdf["cod_dpto"] = gdf["cod_dpto"].astype(str).str.zfill(2)
    gdf = gdf.to_crs(epsg=4326).reset_index(drop=True)
    gdf["nombre_corto"] = (
        gdf["nombre"]
        .str.replace(r"Archipielago de San Andres.*", "San Andres", regex=True)
        .str.replace("Bogota, D.C.", "Bogota", regex=False)
    )

    # --- Panel ---
    depto_codes = {
        "05": "Antioquia", "08": "Atlantico", "11": "Bogota D.C.", "13": "Bolivar",
        "15": "Boyaca", "17": "Caldas", "18": "Caqueta", "19": "Cauca", "20": "Cesar",
        "23": "Cordoba", "25": "Cundinamarca", "27": "Choco", "41": "Huila",
        "44": "La Guajira", "47": "Magdalena", "50": "Meta", "52": "Narino",
        "54": "Norte de Santander", "63": "Quindio", "66": "Risaralda", "68": "Santander",
        "70": "Sucre", "73": "Tolima", "76": "Valle del Cauca", "81": "Arauca",
        "85": "Casanare", "86": "Putumayo", "88": "San Andres", "91": "Amazonas",
        "94": "Guainia", "95": "Guaviare", "97": "Vaupes", "99": "Vichada",
    }
    pib_2024_oficial = {
        "11": 430151, "05": 253232, "76": 163490, "68": 109532, "25": 108202, "08": 76818,
        "13": 61185, "50": 54797, "15": 45232, "73": 37046, "19": 31103, "20": 31092,
        "23": 30952, "41": 29310, "17": 28821, "66": 28675, "54": 27389, "52": 25835,
        "47": 24026, "85": 23586, "44": 17618, "63": 14587, "70": 13700, "81": 9209,
        "27": 7582, "18": 6831, "86": 6091, "88": 2454, "95": 1396, "91": 1200,
        "99": 1100, "94": 605, "97": 472,
    }
    tasas_nac = {2018: 0.026, 2019: 0.033, 2020: -0.073, 2021: 0.107, 2022: 0.073, 2023: 0.006, 2024: 0.017}
    years = list(range(2018, 2025))
    records = []
    for cod, pib24 in pib_2024_oficial.items():
        val = float(pib24)
        serie = {2024: val}
        for y in reversed(years[:-1]):
            g = tasas_nac.get(y + 1, 0.02)
            val = val / (1 + g)
            serie[y] = val
        for y, v in serie.items():
            records.append({"cod_dpto": cod, "anio": y, "pib": v, "nombre": depto_codes.get(cod, "?")})
    df_pib = pd.DataFrame(records)

    ipm_reciente = {
        "11": 5.4, "05": 10.9, "76": 6.2, "68": 6.8, "25": 7.4, "08": 9.5, "13": 15.6, "50": 10.4,
        "15": 6.9, "73": 12.6, "19": 14.0, "20": 13.4, "23": 25.7, "41": 10.9, "17": 9.2, "66": 9.5,
        "54": 15.2, "52": 18.1, "47": 19.2, "85": 11.3, "44": 39.3, "63": 7.4, "70": 21.8, "81": 17.7,
        "27": 33.9, "18": 13.9, "86": 11.8, "88": 6.0, "95": 21.3, "91": 15.6, "99": 70.2, "94": 49.0, "97": 37.4,
    }
    tic_proxy = {c: max(0.1, 1.0 - (ipm_reciente.get(c, 20) / 100) * 0.8) for c in pib_2024_oficial}
    luces_proxy = {c: np.log1p(pib_2024_oficial[c]) * 0.18 + 0.5 for c in pib_2024_oficial}

    panel = df_pib.copy()
    panel["ipm"] = panel["cod_dpto"].map(ipm_reciente)
    panel["tic"] = panel["cod_dpto"].map(tic_proxy)
    panel["luces"] = panel["cod_dpto"].map(luces_proxy)
    panel[["ipm", "tic", "luces"]] = panel.groupby("cod_dpto")[["ipm", "tic", "luces"]].ffill().bfill()
    ultimo = panel[panel.anio == panel.anio.max()].copy()
    gdf_map = gdf.merge(ultimo, on="cod_dpto", how="left").reset_index(drop=True)

    # --- W (permutations reducidas para Cloud) ---
    PERM = 199
    y_ref = np.log1p(gdf_map["pib"].fillna(gdf_map["pib"].median()).values.astype(float))
    candidatos = []
    try:
        wq = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
        wq.transform = "r"
        candidatos.append(("Queen", wq))
    except Exception:
        pass
    try:
        wr = Rook.from_dataframe(gdf, use_index=False, silence_warnings=True)
        wr.transform = "r"
        candidatos.append(("Rook", wr))
    except Exception:
        pass
    for k in [4, 5]:
        try:
            wk = KNN.from_dataframe(gdf, k=k, use_index=False)
            wk.transform = "r"
            candidatos.append((f"KNN-{k}", wk))
        except Exception:
            pass

    mejores = []
    for nombre, w in candidatos:
        try:
            mi = Moran(y_ref, w, permutations=PERM)
            mejores.append({"nombre": nombre, "I": float(mi.I), "p": float(mi.p_sim), "w": w})
        except Exception:
            pass
    if not mejores:
        raise RuntimeError("No se pudo evaluar W")
    sig = [m for m in mejores if m["p"] < 0.05]
    pool = sig if sig else mejores
    best_w = max(pool, key=lambda m: abs(m["I"]))
    W = best_w["w"]

    vars_esda = {
        "pib": np.log1p(gdf_map["pib"].fillna(0).values.astype(float)),
        "ipm": gdf_map["ipm"].fillna(0).values.astype(float),
        "tic": gdf_map["tic"].fillna(0).values.astype(float),
        "luces": gdf_map["luces"].fillna(0).values.astype(float),
    }
    moran_results = {}
    for vname, y in vars_esda.items():
        mi = Moran(y, W, permutations=PERM)
        moran_results[vname] = {"I": float(mi.I), "p": float(mi.p_sim)}

    # --- SpVAR matrices ---
    orden_espacial = gdf["cod_dpto"].tolist()
    n_spatial = len(orden_espacial)
    vars_sistema = ["pib", "ipm", "tic"]
    years_sorted = sorted(panel.anio.unique())
    T, K = len(years_sorted), len(vars_sistema)

    Y = np.full((T, n_spatial, K), np.nan)
    for t, anio in enumerate(years_sorted):
        sub = panel[panel.anio == anio].set_index("cod_dpto")
        for i, cod in enumerate(orden_espacial):
            if cod in sub.index:
                for k, v in enumerate(vars_sistema):
                    Y[t, i, k] = sub.loc[cod, v]
    Y[:, :, 0] = np.log(np.maximum(Y[:, :, 0], 1.0))
    Y = np.nan_to_num(Y, nan=np.nanmean(Y), posinf=0.0, neginf=0.0)
    dY = np.diff(Y, axis=0)
    dY = np.nan_to_num(dY, nan=0.0, posinf=0.0, neginf=0.0)

    def select_spatial_spec(y, X, w, name="eq"):
        results = {}
        y = np.nan_to_num(np.asarray(y, dtype=float).reshape(-1, 1), nan=0.0)
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        try:
            ols = OLS(y, X, w=w, spat_diag=True, nonspat_diag=True, name_y=name, name_w="W")
            results["OLS"] = {"aic": getattr(ols, "aic", np.inf), "model": ols, "type": "OLS"}
        except Exception:
            pass
        try:
            sar = ML_Lag(y, X, w=w, name_y=name, name_w="W", method="full")
            results["SAR"] = {"aic": getattr(sar, "aic", np.inf), "model": sar, "type": "SAR"}
        except Exception:
            pass
        try:
            sem = ML_Error(y, X, w=w, name_y=name, name_w="W", method="full")
            results["SEM"] = {"aic": getattr(sem, "aic", np.inf), "model": sem, "type": "SEM"}
        except Exception:
            pass
        if not results:
            return None
        best = min(results, key=lambda k: results[k]["aic"])
        return results[best]

    t_idx, t_lag = -1, (-2 if dY.shape[0] > 1 else -1)
    param_tables, fit_stats_all, best_models = {}, {}, {}

    for k, vname in enumerate(vars_sistema):
        y = dY[t_idx, :, k]
        X_list = [np.ones(n_spatial)]
        labels = ["const"]
        for j in range(K):
            if j != k:
                X_list.append(dY[t_idx, :, j])
                labels.append(f"beta_{vars_sistema[j]}")
        X_list += [dY[t_lag, :, k], lag_spatial(W, dY[t_lag, :, k])]
        labels += ["phi", "theta"]
        X = np.column_stack(X_list)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        try:
            res = select_spatial_spec(y[mask], X[mask], W, name=vname)
            best_models[vname] = res
            if res is not None:
                labs = list(labels)
                nb = len(np.asarray(res["model"].betas).flatten())
                while len(labs) < nb:
                    labs.append(f"wx_{len(labs)}")
                param_tables[vname] = extract_model_table(res["model"], labs)
                fit_stats_all[vname] = {**model_fit_stats(res["model"]), "tipo": res["type"]}
        except Exception:
            best_models[vname] = None

    rho_est = [fs["rho"] for fs in fit_stats_all.values() if np.isfinite(fs.get("rho", np.nan))]
    max_mod, estabilidad_ok = 0.0, True
    if rho_est:
        rho_mean = float(np.mean(rho_est))
        max_mod = float(np.max(np.abs(np.linalg.eigvals(rho_mean * W.full()[0]))))
        estabilidad_ok = max_mod < 1.0

    # Pronóstico
    HORIZON = 10
    forecast = np.zeros((HORIZON + 1, n_spatial, K))
    forecast[0] = Y[-1].copy()
    dY_mean = np.nanmean(dY, axis=0)
    for h in range(1, HORIZON + 1):
        innov = dY_mean.copy()
        for k in range(K):
            innov[:, k] = 0.7 * innov[:, k] + 0.3 * lag_spatial(W, innov[:, k])
        forecast[h] = forecast[h - 1] + innov

    rho_for_irf = float(np.clip(np.mean(rho_est) if rho_est else 0.25, 0.05, 0.55))

    policy_lines = [
        "=== INTERPRETACION AUTOMATICA SpVAR COLOMBIA ===",
        f"W: {best_w['nombre']} (I={best_w['I']:.4f}, p={best_w['p']:.4f})",
        f"Estabilidad: max|lambda|={max_mod:.4f} -> {'ESTABLE' if estabilidad_ok else 'REVISAR'}",
        "",
        "--- Moran I ---",
    ]
    for v, m in moran_results.items():
        policy_lines.append(f"  {v}: I={m['I']:.4f} p={m['p']:.4f}")
    policy_lines.append("\n--- Estimacion ---")
    for v, fs in fit_stats_all.items():
        policy_lines.append(
            f"  {v}: {fs.get('tipo')} R2={fs.get('r2', np.nan):.4f} "
            f"AIC={fs.get('aic', np.nan):.2f} rho={fs.get('rho', np.nan)}"
        )
    policy_lines += [
        "\n--- Politica ---",
        "1. Spillovers requieren coordinacion regional en TIC y crecimiento.",
        "2. Clusters LISA HH/LL orientan focalizacion territorial.",
        "3. Pronostico 10y es mecanico (dinamicas + W); no es escenario de politica.",
        "4. Ampliar series oficiales DANE/MinTIC/VIIRS.",
    ]
    policy_text = "\n".join(policy_lines)

    # Serializar tablas
    tables = {k: v.to_dict(orient="list") for k, v in param_tables.items()}

    return {
        "gdf": gdf,
        "gdf_map": gdf_map,
        "ultimo": ultimo,
        "panel": panel,
        "W_name": best_w["nombre"],
        "W_I": best_w["I"],
        "W_p": best_w["p"],
        "moran_results": moran_results,
        "vars_esda": vars_esda,
        "fit_stats_all": fit_stats_all,
        "tables": tables,
        "max_mod": max_mod,
        "estabilidad_ok": estabilidad_ok,
        "forecast": forecast,
        "rho_for_irf": rho_for_irf,
        "policy_text": policy_text,
        "n_spatial": n_spatial,
        "K": K,
        "dY": dY,
        "orden_espacial": orden_espacial,
    }


def get_W_from_gdf(gdf, name: str):
    if name == "Queen":
        w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    elif name == "Rook":
        w = Rook.from_dataframe(gdf, use_index=False, silence_warnings=True)
    elif name.startswith("KNN"):
        k = int(name.split("-")[1])
        w = KNN.from_dataframe(gdf, k=k, use_index=False)
    else:
        w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    w.transform = "r"
    return w


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🇨🇴 SpVAR Colombia")
st.markdown(
    "##### Laboratorio de economía espacial departamental  \n"
    "**ESDA (Moran + LISA) · Spatial VAR · Pronóstico 10 años**"
)
st.caption("32 departamentos + Bogotá D.C. · Geometría DANE MGN 2018 · Streamlit Cloud")

try:
    data = run_pipeline()
except Exception as e:
    st.error(f"Error al ejecutar el pipeline: {e}")
    st.stop()

W = get_W_from_gdf(data["gdf"], data["W_name"])
gdf_map = data["gdf_map"]
gdf = data["gdf"]
ultimo = data["ultimo"]
vars_esda = data["vars_esda"]
forecast = data["forecast"]
K = data["K"]
n_spatial = data["n_spatial"]
dY = data["dY"]

# Sidebar
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Sección",
    [
        "Resumen",
        "Mapas coropléticos",
        "ESDA: Moran + LISA",
        "Estimación SpVAR",
        "Simulación IRF",
        "Pronóstico 10 años",
        "Interpretación",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**W:** {data['W_name']}  \n"
    f"**Moran I:** {data['W_I']:.4f}  \n"
    f"**Estabilidad:** {'ESTABLE' if data['estabilidad_ok'] else 'REVISAR'}"
)

CMAP_VAR = {
    "PIB": ("pib", "YlOrRd", "PIB departamental"),
    "IPM": ("ipm", "RdYlGn_r", "IPM (%)"),
    "TIC": ("tic", "YlGnBu", "Proxy TIC"),
    "Luces": ("luces", "inferno", "Radiancia nocturna"),
}

# ---------- RESUMEN ----------
if page == "Resumen":
    st.subheader("Resumen técnico")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matriz W", data["W_name"])
    c2.metric("Moran I", f"{data['W_I']:.4f}")
    c3.metric("p-value", f"{data['W_p']:.4f}")
    c4.metric(
        "Estabilidad",
        "ESTABLE" if data["estabilidad_ok"] else "REVISAR",
        delta=f"λ={data['max_mod']:.4f}",
    )
    st.info(
        "Pipeline SpVAR: descriptivas, mapas, Moran global, LISA, estimación espacial "
        "y pronóstico a 10 años. Primera carga cacheada por Streamlit."
    )
    st.markdown("#### Moran I por variable")
    st.dataframe(
        pd.DataFrame([
            {"variable": k, "I": v["I"], "p": v["p"]}
            for k, v in data["moran_results"].items()
        ]),
        use_container_width=True,
    )

# ---------- MAPAS ----------
elif page == "Mapas coropléticos":
    st.subheader("Mapas coropléticos")
    var = st.selectbox("Variable", list(CMAP_VAR.keys()), index=0)
    col, cmap, title = CMAP_VAR[var]
    fig, ax = plt.subplots(figsize=(8, 9))
    gdf_map.plot(
        column=col, ax=ax, legend=True, cmap=cmap,
        edgecolor="#222", linewidth=0.4,
        legend_kwds={"shrink": 0.55, "label": title},
        missing_kwds={"color": "#eee"},
    )
    ax.set_title(f"Colombia – {title}", fontweight="bold")
    ax.set_axis_off()
    fig.patch.set_facecolor("#f7f8fa")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 12 departamental")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, c, t, color in zip(
        axes.flat,
        ["pib", "ipm", "tic", "luces"],
        ["PIB", "IPM", "TIC", "Luces"],
        ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"],
    ):
        top = ultimo.nlargest(12, c)
        vals = top[c] / 1000 if c == "pib" else top[c]
        ax.barh(top["nombre"], vals, color=color)
        ax.set_title(t, fontweight="bold")
        ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------- ESDA ----------
elif page == "ESDA: Moran + LISA":
    st.subheader("ESDA · Moran scatter y LISA")
    var = st.selectbox("Variable", list(CMAP_VAR.keys()), index=0)
    key = {"PIB": "pib", "IPM": "ipm", "TIC": "tic", "Luces": "luces"}[var]
    y_raw = vars_esda[key]
    y = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-12)
    wy = lag_spatial(W, y)
    mi = Moran(y_raw, W, permutations=199)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y, wy, c="#1a5276", edgecolors="white", s=55, alpha=0.9)
        slope, intercept, r, p, se = stats.linregress(y, wy)
        xl = np.linspace(y.min(), y.max(), 40)
        ax.plot(xl, intercept + slope * xl, color="#c0392b", lw=2.2)
        ax.axhline(0, color="#888", ls="--", lw=0.8)
        ax.axvline(0, color="#888", ls="--", lw=0.8)
        ax.set_xlabel(f"{key} estandarizado")
        ax.set_ylabel(f"W · {key}")
        ax.set_title(f"Moran scatter – {var}\nI={mi.I:.4f}  p={mi.p_sim:.4f}", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        lisa = Moran_Local(y_raw, W, permutations=199)
        sig = lisa.p_sim < 0.05
        quad = lisa.q.copy()
        quad[~sig] = 0
        gdf_tmp = gdf_map.copy()
        gdf_tmp["lisa_q"] = quad
        colors = {1: "#d7191c", 2: "#abd9e9", 3: "#2c7bb6", 4: "#fdae61", 0: "#eeeeee"}
        labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL", 0: "NS"}
        fig, ax = plt.subplots(figsize=(6, 7))
        for qval, color in colors.items():
            sub = gdf_tmp[gdf_tmp["lisa_q"] == qval]
            if len(sub):
                sub.plot(ax=ax, color=color, edgecolor="#333", linewidth=0.35, label=labels[qval])
        ax.set_title(f"LISA – {var} (p<0.05)", fontweight="bold")
        ax.set_axis_off()
        ax.legend(loc="lower left", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    counts = pd.Series(np.where(sig, lisa.q, 0)).value_counts().to_dict()
    st.code(
        f"Variable: {var}\nW: {data['W_name']}\n"
        f"Moran's I = {mi.I:.4f}\np-value = {mi.p_sim:.4f}\n"
        f"LISA p<0.05 → HH={counts.get(1,0)} LH={counts.get(2,0)} "
        f"LL={counts.get(3,0)} HL={counts.get(4,0)} NS={counts.get(0,0)}"
    )

# ---------- ESTIMACIÓN ----------
elif page == "Estimación SpVAR":
    st.subheader("Parámetros del modelo SpVAR")
    for vname, cols in data["tables"].items():
        fs = data["fit_stats_all"].get(vname, {})
        st.markdown(
            f"#### {vname.upper()} · {fs.get('tipo', '?')} · "
            f"R²={fs.get('r2', float('nan')):.4f} · "
            f"AIC={fs.get('aic', float('nan')):.2f} · "
            f"ρ={fs.get('rho', float('nan'))}"
        )
        st.dataframe(pd.DataFrame(cols), use_container_width=True)

# ---------- IRF ----------
elif page == "Simulación IRF":
    st.subheader("Impulso-respuesta espacial")
    shock = st.selectbox("Shock", ["PIB", "IPM", "TIC"])
    horizon = st.slider("Horizonte", 2, 8, 5)
    rho = st.slider("ρ espacial", 0.05, 0.55, float(data["rho_for_irf"]), 0.01)
    persist = st.slider("Persistencia", 0.1, 0.75, 0.40, 0.05)

    idx = {"PIB": 0, "IPM": 1, "TIC": 2}[shock]
    irf = np.zeros((horizon, n_spatial, K))
    std = float(np.nanstd(dY[:, :, idx]) or 1.0)
    irf[0, :, idx] = std
    for h in range(1, horizon):
        for k in range(K):
            irf[h, :, k] = persist * irf[h - 1, :, k] + rho * lag_spatial(W, irf[h - 1, :, k])

    gdf_s = gdf.copy()
    gdf_s["resp"] = irf[-1, :, idx]
    fig, ax = plt.subplots(figsize=(7, 8))
    gdf_s.plot(
        column="resp", ax=ax, cmap="RdBu_r", edgecolor="#222", linewidth=0.35,
        legend=True, legend_kwds={"shrink": 0.55, "label": "Respuesta IRF"},
    )
    ax.set_title(f"IRF espacial | shock {shock} | h={horizon}", fontweight="bold")
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)

    media = irf.mean(1)[:, idx]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=media, mode="lines+markers", line=dict(width=3, color="#1a5276")))
    fig2.update_layout(
        title="Respuesta media espacial",
        template="plotly_white",
        height=280,
        xaxis_title="Horizonte",
        yaxis_title="Magnitud",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Shock={shock} | h={horizon} | rho={rho:.2f} | media_final={media[-1]:.4f}")

# ---------- PRONÓSTICO ----------
elif page == "Pronóstico 10 años":
    st.subheader("Pronóstico a 10 años")
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, name, c in zip(range(K), ["log PIB", "IPM", "TIC"], ["#1f77b4", "#d62728", "#2ca02c"]):
        ax.plot(range(forecast.shape[0]), forecast.mean(1)[:, k], "o-", label=name, color=c)
    ax.legend()
    ax.set_title("Pronóstico medio nacional")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    z10 = ensure_nonneg(forecast[-1, :, 0] - forecast[-1, :, 0].min() + 0.1)
    sizes = ensure_nonneg(10 * z10 / (z10.max() + 1e-9))
    fig3 = go.Figure(data=[go.Scatter3d(
        x=gdf.geometry.centroid.x,
        y=gdf.geometry.centroid.y,
        z=z10,
        mode="markers+text",
        text=gdf["nombre_corto"],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(size=sizes, color=z10, colorscale="Plasma", colorbar=dict(title="log PIB t+10"), opacity=0.9),
    )])
    fig3.update_layout(title="Pronóstico 3D t+10 (log PIB)", height=520, scene=dict(bgcolor="#fafafa"))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"W={data['W_name']} | max|λ|={data['max_mod']:.4f} | "
        f"logPIB t0={forecast[0, :, 0].mean():.3f} → t10={forecast[-1, :, 0].mean():.3f}"
    )

# ---------- INTERPRETACIÓN ----------
elif page == "Interpretación":
    st.subheader("Interpretación automática de política")
    st.code(data["policy_text"], language=None)

st.markdown("---")
st.caption(f"SpVAR Colombia · {datetime.now().strftime('%Y-%m-%d')} · DANE MGN 2018 · Streamlit")
