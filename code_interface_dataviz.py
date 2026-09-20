import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None

try:
    from scipy import stats
except Exception:
    stats = None

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")


# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Pilotage Flux d'Assistance", layout="wide")

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_PREVIEW_ROWS = 5000
PROFILE_MAX_ROWS = 50000


def first_existing_path(*filenames):
    for filename in filenames:
        path = os.path.join(ROOT, filename)
        if os.path.exists(path):
            return path
    return os.path.join(ROOT, filenames[0])

# --- 2. CHARGEMENT SECURISE DES DONNEES ET DU MODELE ---
def read_csv_auto(path, nrows=None):
    """Lecture robuste aux encodages usuels."""
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, nrows=nrows)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python", nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def available_columns(df, cols):
    return [col for col in cols if col in df.columns]


def mode_safe(series):
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def build_grouped_base(base_fusion):
    required = {"numero_dossier", "duree_corrigee", "matricule"}
    if base_fusion.empty or not required.issubset(base_fusion.columns):
        return pd.DataFrame()

    work = base_fusion.copy()
    if "date_debut_traitement" in work.columns:
        work["date_debut_traitement"] = pd.to_datetime(work["date_debut_traitement"], errors="coerce")
    if "teletravail_ind" not in work.columns and "lieu_travail" in work.columns:
        work["teletravail_ind"] = work["lieu_travail"].astype(str).str.upper().eq("TELE").astype(float)
    if "nb_garanties" not in work.columns:
        garanties = available_columns(
            work,
            ["top_DR", "top_VR", "top_rappat_valide", "top_poursuite", "top_recup", "top_autres_garanties"],
        )
        work["nb_garanties"] = work[garanties].sum(axis=1) if garanties else np.nan

    aggregations = {
        "duree_totale_dossier": ("duree_corrigee", lambda x: x.sum(min_count=1)),
        "nb_interventions": ("duree_corrigee", "size"),
        "nb_agents": ("matricule", "nunique"),
    }
    optional_aggs = {
        "experience_moyenne": ("experience", "mean"),
        "experience_max": ("experience", "max"),
        "temps_travail_moyen": ("temps_travail", "mean"),
        "part_teletravail": ("teletravail_ind", "mean"),
        "nb_garanties_dossier": ("nb_garanties", "max"),
        "delai_premier_traitement": ("delai_traitement_ouverture", "min"),
        "cause_intervention": ("cause_intervention", mode_safe),
        "formule": ("formule", mode_safe),
        "type_d_energie": ("type_d_energie", mode_safe),
        "outil_d_assistance": ("outil_d_assistance", mode_safe),
        "assistance_ou_administratif": ("assistance_ou_administratif", mode_safe),
        "site_dominant": ("site", mode_safe),
        "lieu_travail_dominant": ("lieu_travail", mode_safe),
        "type_de_contrat_dominant": ("type_de_contrat", mode_safe),
        "population_dominante": ("population", mode_safe),
        "info_dossier_manquante": ("info_dossier_manquante", "max"),
        "info_ressource_manquante": ("info_ressource_manquante", "max"),
    }
    for output_col, (source_col, func) in optional_aggs.items():
        if source_col in work.columns:
            aggregations[output_col] = (source_col, func)

    grouped = work.groupby("numero_dossier").agg(**aggregations).reset_index()
    if "date_debut_traitement" in work.columns:
        first_month = work.groupby("numero_dossier")["date_debut_traitement"].min().dt.month
        grouped["mois_premier_traitement"] = grouped["numero_dossier"].map(first_month)
    return grouped


def enrich_grouped_base(df):
    if df.empty:
        return df
    out = df.copy()
    for col in ["date_ouverture", "date_de_survenance", "date_premier_traitement"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "date_ouverture" in out.columns:
        out["annee"] = out["date_ouverture"].dt.year
        out["mois_num"] = out["date_ouverture"].dt.to_period("M").astype(str)
    if "formule_regroupee" not in out.columns and "formule" in out.columns:
        freq = out["formule"].value_counts(dropna=False)
        kept = freq[freq >= 1000].index
        out["formule_regroupee"] = np.where(out["formule"].isin(kept), out["formule"], "Autres")
    if "log_duree_totale_dossier" not in out.columns and "duree_totale_dossier" in out.columns:
        out["log_duree_totale_dossier"] = np.log1p(pd.to_numeric(out["duree_totale_dossier"], errors="coerce"))
    return out


@st.cache_data(show_spinner="Chargement des donnees...")
def load_data():
    base_paths = {
        "dossier": os.path.join(ROOT, "dossier.csv"),
        "temps": os.path.join(ROOT, "temps.csv"),
        "ressources": os.path.join(ROOT, "ressources.csv"),
        "fusion_model": os.path.join(ROOT, "base_model.csv"),
        "fusion_full": os.path.join(ROOT, "base_fusion_27052026.csv"),
        "fusion_alt": os.path.join(ROOT, "base_fusion_23042026.csv"),
        "grouped": os.path.join(ROOT, "base_dossier_eco.csv"),
    }

    sources = {
        "Dossiers bruts": read_csv_auto(base_paths["dossier"]) if os.path.exists(base_paths["dossier"]) else pd.DataFrame(),
        "Temps bruts": read_csv_auto(base_paths["temps"]) if os.path.exists(base_paths["temps"]) else pd.DataFrame(),
        "Ressources brutes": read_csv_auto(base_paths["ressources"]) if os.path.exists(base_paths["ressources"]) else pd.DataFrame(),
    }

    fusion_path = next(
        (p for p in [base_paths["fusion_model"], base_paths["fusion_full"], base_paths["fusion_alt"]] if os.path.exists(p)),
        None,
    )
    df = read_csv_auto(fusion_path) if fusion_path else pd.DataFrame()

    if os.path.exists(base_paths["grouped"]):
        df_impute = read_csv_auto(base_paths["grouped"])
    else:
        df_impute = build_grouped_base(df)
    df_impute = enrich_grouped_base(df_impute)

    return df, df_impute, sources


@st.cache_resource(show_spinner="Chargement du modele sauvegarde...")
def load_model():
    if joblib is None:
        return None
    for filename in ["best_model.pkl", "modele_duree.pkl"]:
        path = os.path.join(ROOT, filename)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                return None
    return None


def get_duration_features(df):
    num_features = available_columns(
        df,
        [
            "nb_interventions",
            "nb_agents",
            "experience_moyenne",
            "temps_travail_moyen",
            "nb_garanties_dossier",
            "part_teletravail",
            "delai_premier_traitement",
            "mois_premier_traitement",
            "info_dossier_manquante",
            "info_ressource_manquante",
        ],
    )
    cat_features = available_columns(
        df,
        [
            "cause_intervention",
            "formule",
            "formule_regroupee",
            "type_d_energie",
            "outil_d_assistance",
            "site_dominant",
            "lieu_travail_dominant",
        ],
    )
    return num_features, cat_features


@st.cache_resource(show_spinner="Entrainement des modeles de duree...")
def train_duration_models(version_key=1):
    _, base_grouped, _ = load_data()
    if base_grouped.empty or "duree_totale_dossier" not in base_grouped.columns:
        return None, pd.DataFrame(), None, [], [], [], pd.DataFrame(), pd.DataFrame()

    num_features, cat_features = get_duration_features(base_grouped)
    features = num_features + cat_features
    if not features:
        return None, pd.DataFrame(), None, [], [], [], pd.DataFrame(), pd.DataFrame()

    target = "duree_totale_dossier"
    model_data = base_grouped[features + [target]].copy()
    model_data[target] = pd.to_numeric(model_data[target], errors="coerce")
    model_data = model_data[model_data[target] > 0].dropna(subset=[target]).copy()

    for col in num_features:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
    for col in cat_features:
        model_data[col] = model_data[col].astype("object").fillna("NON_RENSEIGNE")

    X = model_data[features]
    y_seconds = model_data[target]
    y_log = np.log1p(y_seconds)

    X_train, X_valid, y_train_log, _ = train_test_split(X, y_log, test_size=0.2, random_state=42)
    y_valid_seconds = y_seconds.loc[X_valid.index]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="NON_RENSEIGNE")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, random_state=42),
    }

    try:
        import lightgbm as lgb

        models["LightGBM"] = lgb.LGBMRegressor(
            objective="huber",
            learning_rate=0.03,
            n_estimators=250,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            verbose=-1,
        )
    except Exception:
        pass

    try:
        import xgboost as xgb

        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
    except Exception:
        pass

    fitted = {}
    rows = []
    importances = []
    for name, regressor in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
        pipe.fit(X_train, y_train_log)
        pred_seconds = np.maximum(np.expm1(pipe.predict(X_valid)), 0)
        rows.append(
            {
                "Modele": name,
                "MAE_secondes": mean_absolute_error(y_valid_seconds, pred_seconds),
                "RMSE_secondes": np.sqrt(mean_squared_error(y_valid_seconds, pred_seconds)),
                "R2": r2_score(y_valid_seconds, pred_seconds),
            }
        )
        fitted[name] = pipe

        reg = pipe.named_steps["regressor"]
        if hasattr(reg, "feature_importances_"):
            for feature, score in zip(features, reg.feature_importances_):
                importances.append({"Modele": name, "variable": feature, "importance": float(score)})

    results = pd.DataFrame(rows).sort_values("MAE_secondes")
    best_name = results.iloc[0]["Modele"]
    return fitted, results, best_name, features, num_features, cat_features, model_data, pd.DataFrame(importances)


def dataframe_profile(df):
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "variable": df.columns,
            "type": df.dtypes.astype(str).values,
            "valeurs_manquantes": df.isna().sum().values,
            "taux_manquant": df.isna().mean().round(4).values,
            "nb_modalites": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )


def numeric_summary(df):
    cols = df.select_dtypes(include=np.number).columns.tolist()
    return df[cols].describe().T.round(2) if cols else pd.DataFrame()


def categorical_summary(df):
    cols = df.select_dtypes(exclude=np.number).columns.tolist()
    rows = []
    for col in cols:
        mode = df[col].mode(dropna=True)
        rows.append(
            {
                "variable": col,
                "nb_modalites": df[col].nunique(dropna=True),
                "modalite_principale": mode.iloc[0] if not mode.empty else None,
                "taux_manquant": df[col].isna().mean(),
            }
        )
    return pd.DataFrame(rows)


def bivariate_numeric_duration(df, target):
    num_cols = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col != target and df[col].nunique(dropna=True) > 1
    ]
    rows = []
    temp_target = pd.to_numeric(df[target], errors="coerce")
    for col in num_cols:
        temp = pd.DataFrame({"x": pd.to_numeric(df[col], errors="coerce"), "y": temp_target}).dropna()
        if len(temp) < 30:
            continue
        pearson = temp["x"].corr(temp["y"], method="pearson")
        spearman = temp["x"].corr(temp["y"], method="spearman")
        rows.append(
            {
                "variable": col,
                "correlation_pearson": pearson,
                "correlation_spearman": spearman,
                "force_lien": max(abs(pearson), abs(spearman)),
            }
        )
    columns = ["variable", "correlation_pearson", "correlation_spearman", "force_lien"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("force_lien", ascending=False)


def bivariate_categorical_duration(df, target):
    cat_cols = [col for col in df.select_dtypes(exclude=np.number).columns if df[col].nunique(dropna=True) <= 30]
    rows = []
    y = pd.to_numeric(df[target], errors="coerce")
    for col in cat_cols:
        temp = pd.DataFrame({"x": df[col].astype("object"), "y": y}).dropna()
        if temp["x"].nunique() < 2 or len(temp) < 30:
            continue
        grouped = temp.groupby("x")["y"]
        groups = [g.values for _, g in grouped if len(g) >= 5]
        p_value = np.nan
        test = "Kruskal"
        if stats is not None and len(groups) >= 2:
            try:
                p_value = stats.kruskal(*groups).pvalue
            except Exception:
                p_value = np.nan
        rows.append(
            {
                "variable": col,
                "nb_modalites": temp["x"].nunique(),
                "mediane_min": grouped.median().min(),
                "mediane_max": grouped.median().max(),
                "ecart_medianes": grouped.median().max() - grouped.median().min(),
                "test": test,
                "p_value": p_value,
            }
        )
    columns = ["variable", "nb_modalites", "mediane_min", "mediane_max", "ecart_medianes", "test", "p_value"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("ecart_medianes", ascending=False)


def chi2_table(df, var1, var2):
    if stats is None or var1 not in df.columns or var2 not in df.columns:
        return None
    table = pd.crosstab(df[var1], df[var2])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return pd.DataFrame(
        [{"variables": f"{var1} x {var2}", "chi2": chi2, "ddl": dof, "p_value": p_value}]
    )


def render_duration_hist(df, target, title, xlimit=None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    values = pd.to_numeric(df[target], errors="coerce").dropna()
    sns.histplot(values, bins=160, color="steelblue", ax=ax)
    ax.set_yscale("log")
    if xlimit is None and not values.empty:
        xlimit = min(float(values.quantile(0.995)), 50000)
    if xlimit:
        ax.set_xlim(0, xlimit)
    ax.set_title(title)
    ax.set_xlabel("Duree en secondes")
    ax.set_ylabel("Nombre d'observations - axe log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def render_boxplot(df, x, y, title):
    if x not in df.columns or y not in df.columns:
        st.info(f"Variable indisponible : {x}")
        return
    temp = df[[x, y]].copy()
    temp[y] = pd.to_numeric(temp[y], errors="coerce")
    temp = temp[temp[y] > 0].dropna()
    if temp.empty:
        st.info(f"Aucune donnee exploitable pour {x}.")
        return
    if temp[x].nunique() > 12:
        keep = temp[x].value_counts().head(12).index
        temp = temp[temp[x].isin(keep)]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=temp, x=x, y=y, color="#9ecae1", ax=ax)
    ax.set_yscale("log")
    ax.set_ylim(1, min(50000, max(10000, temp[y].quantile(0.995))))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Duree totale en secondes - axe log")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)
    plt.close(fig)


# Initialisation des donnees
base_fusion, Base_apres_imputation, bases_sources = load_data()
best_model = load_model()

# Noms exacts de vos colonnes d'experience
COL_EXP_FUSION = "experience"
COL_EXP_IMPUTE = "experience_moyenne"


# --- 3. EN-TETE ---
col1, col2 = st.columns([3, 1])
col1.write(
    """
    # Pilotage Operationnel des Flux d'Assistance pour deplacement en France
    *Interface d'exploration de la base d'activite, des analyses QDD/econometrie et du modele ML (2021 - 2022)*
    """
)
if os.path.exists(r'C:\\Users\\odilo\\Documents\\DU_python\\logo_univ.jpg'):
    col2.image(r'C:\\Users\\odilo\\Documents\\DU_python\logo_univ.jpg', width=200)

with col2:
    st.metric("Dossiers groupes", f"{Base_apres_imputation.shape[0]:,}" if not Base_apres_imputation.empty else "N/A")


# --- 4. BARRE LATERALE : FILTRES SUR VARIABLES CLES & RESET ---
st.sidebar.header("Filtres Globaux")

if st.sidebar.button("Reinitialiser tous les filtres", use_container_width=True):
    for key in list(st.session_state.keys()):
        if key.startswith("filter_"):
            del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")

df_fusion_filtree = base_fusion.copy()
df_impute_filtree = Base_apres_imputation.copy()

toutes_les_colonnes = sorted(list(set(base_fusion.columns.tolist() + Base_apres_imputation.columns.tolist())))

col_dossier = None
for c in toutes_les_colonnes:
    normalized_c = c.lower().replace("_", "").replace(" ", "")
    if "numerodossier" in normalized_c or "numdossier" in normalized_c:
        col_dossier = c
        break
if not col_dossier and "numero_dossier" in toutes_les_colonnes:
    col_dossier = "numero_dossier"

if col_dossier:
    vals_f = base_fusion[col_dossier].dropna().unique().astype(str).tolist() if col_dossier in base_fusion.columns else []
    vals_i = (
        Base_apres_imputation[col_dossier].dropna().unique().astype(str).tolist()
        if col_dossier in Base_apres_imputation.columns
        else []
    )
    options_dossier = sorted(list(set(vals_f + vals_i)))
    dossiers_selectionnes = st.sidebar.multiselect(
        "Filtrer par Numero Dossier :",
        options=options_dossier,
        key="filter_numero_dossier",
    )
    if dossiers_selectionnes:
        if col_dossier in df_fusion_filtree.columns:
            df_fusion_filtree = df_fusion_filtree[df_fusion_filtree[col_dossier].astype(str).isin(dossiers_selectionnes)]
        if col_dossier in df_impute_filtree.columns:
            df_impute_filtree = df_impute_filtree[df_impute_filtree[col_dossier].astype(str).isin(dossiers_selectionnes)]

for col in toutes_les_colonnes:
    col_lower = col.lower()
    normalized_col = col_lower.replace("_", "").replace(" ", "")
    if (
        "top" in col_lower
        or "log" in col_lower
        or "matricule" in col_lower
        or "numerodossier" in normalized_col
        or "numdossier" in normalized_col
        or "lieu_travail" in col_lower
        or "lieu_travil" in col_lower
        or "ressources" in col_lower
        or "manquantes" in col_lower
        or "manquntes" in col_lower
        or "codification" in col_lower
        or "intervalle" in col_lower
        or col == col_dossier
    ):
        continue

    serie_fusion = base_fusion[col] if col in base_fusion.columns else pd.Series(dtype=object)
    serie_impute = Base_apres_imputation[col] if col in Base_apres_imputation.columns else pd.Series(dtype=object)

    high_cardinality = (
        (not serie_fusion.empty and serie_fusion.dtype == "object" and serie_fusion.nunique(dropna=True) > 50)
        or (not serie_impute.empty and serie_impute.dtype == "object" and serie_impute.nunique(dropna=True) > 50)
    )
    if high_cardinality:
        continue

    key_name = f"filter_{col}"
    label_filtre = f"{col.replace('_', ' ').title()}"

    is_short_cat = (
        col in base_fusion.columns
        and (base_fusion[col].dtype == "object" or base_fusion[col].nunique(dropna=True) <= 15)
    ) or (
        col in Base_apres_imputation.columns
        and (Base_apres_imputation[col].dtype == "object" or Base_apres_imputation[col].nunique(dropna=True) <= 15)
    )

    if is_short_cat:
        vals_f = base_fusion[col].dropna().unique().astype(str).tolist() if col in base_fusion.columns else []
        vals_i = (
            Base_apres_imputation[col].dropna().unique().astype(str).tolist()
            if col in Base_apres_imputation.columns
            else []
        )
        options = sorted(list(set(vals_f + vals_i)))
        selection = st.sidebar.multiselect(f"Filtrer par {label_filtre} :", options=options, key=key_name)
        if selection:
            if col in df_fusion_filtree.columns:
                df_fusion_filtree = df_fusion_filtree[df_fusion_filtree[col].astype(str).isin(selection)]
            if col in df_impute_filtree.columns:
                df_impute_filtree = df_impute_filtree[df_impute_filtree[col].astype(str).isin(selection)]
    else:
        try:
            min_f = float(pd.to_numeric(base_fusion[col], errors="coerce").min()) if col in base_fusion.columns else float("inf")
            max_f = float(pd.to_numeric(base_fusion[col], errors="coerce").max()) if col in base_fusion.columns else float("-inf")
            min_i = (
                float(pd.to_numeric(Base_apres_imputation[col], errors="coerce").min())
                if col in Base_apres_imputation.columns
                else float("inf")
            )
            max_i = (
                float(pd.to_numeric(Base_apres_imputation[col], errors="coerce").max())
                if col in Base_apres_imputation.columns
                else float("-inf")
            )
        except Exception:
            continue
        min_val = min(min_f, min_i)
        max_val = max(max_f, max_i)
        if min_val != max_val and not (np.isinf(min_val) or np.isinf(max_val) or np.isnan(min_val) or np.isnan(max_val)):
            range_val = st.sidebar.slider(
                f"Intervalle {label_filtre} :",
                min_val,
                max_val,
                (min_val, max_val),
                key=key_name,
            )
            if col in df_fusion_filtree.columns:
                numeric_col = pd.to_numeric(df_fusion_filtree[col], errors="coerce")
                df_fusion_filtree = df_fusion_filtree[(numeric_col >= range_val[0]) & (numeric_col <= range_val[1])]
            if col in df_impute_filtree.columns:
                numeric_col = pd.to_numeric(df_impute_filtree[col], errors="coerce")
                df_impute_filtree = df_impute_filtree[(numeric_col >= range_val[0]) & (numeric_col <= range_val[1])]


# --- 5. STRUCTURE DES ONGLETS ---
onglets = st.tabs(
    [
        "Donnees Brutes & Filtrees",
        "Analyse Exploratoire (Dataviz & Statistiques)",
        "Econometrie",
        "Machine Learning",
    ]
)


# --- ONGLET 1 : EXPLORATION ET TABLEAUX ---
with onglets[0]:
    st.subheader("Visualisation et Controle des Donnees")

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Lignes restantes - Base Fusion", value=f"{df_fusion_filtree.shape[0]:,}")
    kpi2.metric(label="Dossiers restants - Base Groupee", value=f"{df_impute_filtree.shape[0]:,}")
    if "duree_totale_dossier" in df_impute_filtree.columns:
        duree = pd.to_numeric(df_impute_filtree["duree_totale_dossier"], errors="coerce")
        kpi3.metric(label="Duree mediane", value=f"{duree.median():,.0f} sec")

    st.markdown("---")

    choix_base = st.radio(
        "Selectionnez la base a afficher sous forme de tableau :",
        ["Base Fusionnee", "Base fusionnee groupee par numero de dossier", "Bases sources"],
        horizontal=True,
    )

    if choix_base == "Base Fusionnee":
        st.markdown("### Extrait de la Base Fusionnee")
        st.dataframe(df_fusion_filtree.head(1000), use_container_width=True, height=420)
        st.markdown("#### Analyse de repartition par variable")
        if not df_fusion_filtree.empty:
            var_choisie = st.selectbox("Choisir une variable a analyser (Base Fusion) :", df_fusion_filtree.columns, key="select_var_fusion")
            repartition = df_fusion_filtree[var_choisie].value_counts(dropna=False).reset_index()
            repartition.columns = [var_choisie, "effectif"]
            st.dataframe(repartition.head(100), use_container_width=True)
        st.markdown("#### Profil des colonnes")
        st.dataframe(dataframe_profile(df_fusion_filtree).sort_values("taux_manquant", ascending=False), use_container_width=True)

    elif choix_base == "Base fusionnee groupee par numero de dossier":
        st.markdown("### Extrait de la Base fusionnee groupee par numero de dossier")
        st.dataframe(df_impute_filtree.head(1000), use_container_width=True, height=420)
        st.markdown("#### Analyse de repartition par variable")
        if not df_impute_filtree.empty:
            var_choisie_imp = st.selectbox(
                "Choisir une variable a analyser (Base Groupee) :",
                df_impute_filtree.columns,
                key="select_var_impute",
            )
            repartition = df_impute_filtree[var_choisie_imp].value_counts(dropna=False).reset_index()
            repartition.columns = [var_choisie_imp, "effectif"]
            st.dataframe(repartition.head(100), use_container_width=True)
        st.markdown("#### Variables construites par regroupement")
        st.dataframe(dataframe_profile(df_impute_filtree).sort_values("taux_manquant", ascending=False), use_container_width=True)

    else:
        st.markdown("### Bases brutes sources")
        st.info(
            "Les trois bases brutes chargees par le dictionnaire `sources` sont affichees "
            "separement sous forme de tableaux : dossiers, temps et ressources."
        )

        source_tabs = st.tabs(list(bases_sources.keys()))
        for source_tab, (source_name, source_df) in zip(source_tabs, bases_sources.items()):
            with source_tab:
                c1, c2, c3 = st.columns(3)
                c1.metric("Lignes", f"{source_df.shape[0]:,}" if not source_df.empty else "0")
                c2.metric("Colonnes", f"{source_df.shape[1]:,}" if not source_df.empty else "0")
                c3.metric("Valeurs manquantes", f"{int(source_df.isna().sum().sum()):,}" if not source_df.empty else "0")

                if source_df.empty:
                    st.warning(f"La base `{source_name}` n'a pas ete chargee ou le fichier est absent.")
                else:
                    st.markdown(f"#### Tableau - {source_name}")
                    st.dataframe(source_df.head(1000), use_container_width=True, height=420)

                    st.markdown(f"#### Profil des colonnes - {source_name}")
                    st.dataframe(dataframe_profile(source_df), use_container_width=True, height=320)

    st.info(
        "Lecture QDD : la base fusionnee correspond au niveau intervention/traitement. "
        "La base groupee ramene l'information au niveau numero_dossier, ce qui permet "
        "d'analyser et de predire la duree totale de traitement du dossier."
    )


# --- ONGLET 2 : ANALYSE EXPLORATOIRE, DATAVIZ ET STATISTIQUES ---
with onglets[1]:
    st.subheader("Dataviz et statistiques descriptives autour de la duree totale")

    tab_dataviz, tab_univariee, tab_bivariee = st.tabs(["Graphiques pertinents", "Analyses univariees", "Analyses bivariees"])

    with tab_dataviz:
        if df_impute_filtree.empty or "duree_totale_dossier" not in df_impute_filtree.columns:
            st.warning("La base groupee ou la duree totale n'est pas disponible.")
        else:
            dff = df_impute_filtree[pd.to_numeric(df_impute_filtree["duree_totale_dossier"], errors="coerce") > 0].copy()
            st.markdown(
                """
                Le notebook QDD met en evidence une distribution tres asymetrique des durees :
                beaucoup de dossiers courts et une queue longue de dossiers complexes. Les graphiques
                utilisent donc souvent un axe logarithmique pour conserver la lecture des cas extremes.
                """
            )

            g1, g2 = st.columns(2)
            with g1:
                render_duration_hist(dff, "duree_totale_dossier", "Distribution globale de la duree totale")
                st.info("La concentration sur les petites durees et la queue droite justifient la transformation log1p utilisee en econometrie et en ML.")
            with g2:
                if "cause_intervention" in dff.columns:
                    prop = dff["cause_intervention"].value_counts(normalize=True).mul(100).reset_index()
                    prop.columns = ["cause_intervention", "proportion"]
                    fig, ax = plt.subplots(figsize=(9, 4.8))
                    sns.barplot(data=prop, x="cause_intervention", y="proportion", color="#74c476", ax=ax)
                    ax.tick_params(axis="x", rotation=35)
                    ax.set_title("Structure des dossiers par cause")
                    ax.set_xlabel("")
                    ax.set_ylabel("Proportion (%)")
                    st.pyplot(fig)
                    plt.close(fig)
                    st.info("La panne mecanique est generalement dominante, tandis que les causes rares doivent etre interpretees avec prudence.")

            g3, g4 = st.columns(2)
            with g3:
                render_boxplot(dff, "cause_intervention", "duree_totale_dossier", "Duree totale par cause d'intervention")
                st.info("Les causes differencient fortement les durees : les dossiers accident, vol/vandalisme, incendie ou bris de glace peuvent tirer la duree vers le haut.")
            with g4:
                render_boxplot(dff, "outil_d_assistance", "duree_totale_dossier", "Duree totale par outil d'assistance")
                st.info("L'outil d'assistance capte une partie des differences d'organisation et de complexite operationnelle.")

            g5, g6 = st.columns(2)
            with g5:
                render_boxplot(dff, "lieu_travail_dominant", "duree_totale_dossier", "Duree totale par lieu de travail dominant")
                st.info("Le lieu de travail est moins central que la complexite du dossier, mais il reste utile dans les modeles comme variable d'organisation.")
            with g6:
                if "nb_interventions" in dff.columns:
                    temp = dff[["nb_interventions", "duree_totale_dossier"]].copy()
                    temp["nb_interventions"] = pd.to_numeric(temp["nb_interventions"], errors="coerce")
                    temp["duree_totale_dossier"] = pd.to_numeric(temp["duree_totale_dossier"], errors="coerce")
                    temp = temp.dropna()
                    fig, ax = plt.subplots(figsize=(9, 4.8))
                    sns.scatterplot(data=temp.sample(min(len(temp), 8000), random_state=42), x="nb_interventions", y="duree_totale_dossier", alpha=0.25, ax=ax)
                    ax.set_yscale("log")
                    ax.set_title("Complexite operationnelle et duree")
                    ax.set_xlabel("Nombre d'interventions")
                    ax.set_ylabel("Duree totale - axe log")
                    st.pyplot(fig)
                    plt.close(fig)
                    st.info("Plus le dossier mobilise d'interventions, plus la duree totale tend a augmenter, ce qui rejoint les resultats Ridge.")

    with tab_univariee:
        if df_impute_filtree.empty:
            st.warning("Aucune donnee disponible avec les filtres selectionnes.")
        else:
            st.markdown("### Statistiques numeriques")
            st.dataframe(numeric_summary(df_impute_filtree), use_container_width=True)

            st.markdown("### Statistiques qualitatives")
            st.dataframe(categorical_summary(df_impute_filtree), use_container_width=True)

            if "duree_totale_dossier" in df_impute_filtree.columns:
                d = pd.to_numeric(df_impute_filtree["duree_totale_dossier"], errors="coerce").dropna()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Moyenne", f"{d.mean():,.0f} sec")
                k2.metric("Mediane", f"{d.median():,.0f} sec")
                k3.metric("P90", f"{d.quantile(0.90):,.0f} sec")
                k4.metric("P95", f"{d.quantile(0.95):,.0f} sec")
                st.info(
                    "Analyse univariee : la moyenne superieure a la mediane signale une asymetrie positive. "
                    "Les percentiles eleves isolent les dossiers longs, utiles pour le pilotage metier."
                )

    with tab_bivariee:
        if df_impute_filtree.empty or "duree_totale_dossier" not in df_impute_filtree.columns:
            st.warning("La duree totale est necessaire pour les analyses bivariees.")
        else:
            st.markdown("### Variables numeriques en lien avec la duree totale")
            num_links = bivariate_numeric_duration(df_impute_filtree, "duree_totale_dossier")
            st.dataframe(num_links.round(4), use_container_width=True)
            if not num_links.empty:
                top_num = num_links.iloc[0]
                st.info(
                    f"Lecture : la variable numerique la plus liee a la duree est `{top_num['variable']}` "
                    f"(force de lien {top_num['force_lien']:.3f}). Cela confirme que la complexite quantitative "
                    "du dossier est centrale."
                )

            st.markdown("### Variables qualitatives en lien avec la duree totale")
            cat_links = bivariate_categorical_duration(df_impute_filtree, "duree_totale_dossier")
            st.dataframe(cat_links.round(4), use_container_width=True)
            if not cat_links.empty:
                top_cat = cat_links.iloc[0]
                st.info(
                    f"Lecture : `{top_cat['variable']}` presente le plus grand ecart de medianes "
                    f"({top_cat['ecart_medianes']:,.0f} sec). Les modalites ne portent donc pas le meme niveau de complexite."
                )

            st.markdown("### Tests QDD issus du notebook")
            chi_rows = []
            for var1, var2 in [
                ("cause_intervention", "assistance_ou_administratif"),
                ("assistance_ou_administratif", "outil_d_assistance"),
                ("lieu_travail_dominant", "site_dominant"),
            ]:
                test = chi2_table(df_impute_filtree, var1, var2)
                if test is not None:
                    chi_rows.append(test)
            if chi_rows:
                st.dataframe(pd.concat(chi_rows, ignore_index=True).round(4), use_container_width=True)
            st.info(
                "Dans le notebook QDD, les tests Chi2 montrent notamment un lien significatif entre cause "
                "d'intervention et type assistance/administratif, ainsi qu'entre lieu de travail, population, site "
                "et type de contrat. Ces liens justifient l'integration de variables qualitatives dans les modeles."
            )


# --- ONGLET 3 : ECONOMETRIE ---
with onglets[2]:
    st.subheader("Modelisation Econometrique")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Modele retenu", "RidgeCV")
    e2.metric("R2 validation", "0.5723")
    e3.metric("R2 CV moyen", "0.5644")
    e4.metric("Alpha Ridge", "6.158")

    st.markdown(
        """
        Le notebook QDD/econometrie estime la duree totale transformee en `log1p`.
        La Ridge est retenue car elle stabilise les coefficients en presence de colinearites,
        notamment entre nombre d'agents, nombre d'interventions et garanties du dossier.
        """
    )

    perf_path = os.path.join(ROOT, "tableau_performance_final_modele_ridge.xlsx")
    if os.path.exists(perf_path):
        perf = pd.read_excel(perf_path)
        st.markdown("### Tableau de performance exporte")
        st.dataframe(perf, use_container_width=True)

    coef_path = os.path.join(ROOT, "coefficients_ridge.csv")
    if os.path.exists(coef_path):
        coef_df = read_csv_auto(coef_path)
        coef_col = "coef_ridge" if "coef_ridge" in coef_df.columns else coef_df.select_dtypes(include=np.number).columns[0]
        if "coef_abs" not in coef_df.columns:
            coef_df["coef_abs"] = coef_df[coef_col].abs()
        coef_df["variable_lisible"] = coef_df["variable"].astype(str).str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
        coef_df["effet"] = np.where(coef_df[coef_col] >= 0, "Allonge la duree", "Reduit la duree")
        top_coef = coef_df.sort_values("coef_abs", ascending=False).head(25)

        st.markdown("### Principaux coefficients Ridge")
        st.dataframe(top_coef[["variable_lisible", coef_col, "coef_abs", "effet"]], use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            data=top_coef.sort_values("coef_abs"),
            x=coef_col,
            y="variable_lisible",
            hue="effet",
            palette={"Allonge la duree": "#d55e00", "Reduit la duree": "#0072b2"},
            dodge=False,
            ax=ax,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Effets econometriques sur log(duree totale)")
        ax.set_xlabel("Coefficient Ridge")
        ax.set_ylabel("")
        ax.legend(title="")
        st.pyplot(fig)
        plt.close(fig)

    diag = pd.DataFrame(
        {
            "Diagnostic": ["Validation simple", "Validation croisee", "Shapiro-Wilk", "Breusch-Pagan", "Durbin-Watson"],
            "Resultat notebook": [
                "MAE log = 0.5762 ; RMSE log = 0.7618 ; R2 = 0.5723",
                "R2 CV moyen = 0.5644 ; RMSE CV moyen = 0.7707",
                "p-value tres faible (2.64e-27)",
                "heteroscedasticite significative",
                "environ 1.99",
            ],
            "Commentaire": [
                "Le modele explique une part substantielle de la variabilite en log-duree.",
                "Les performances restent stables en validation croisee.",
                "Les residus ne sont pas normaux, ce qui est attendu sur des durees tres asymetriques.",
                "La variance des erreurs varie selon les niveaux de prediction.",
                "Pas d'autocorrelation notable des residus.",
            ],
        }
    )
    st.markdown("### Diagnostics et commentaire")
    st.dataframe(diag, use_container_width=True)
    st.info(
        "Conclusion econometrique : la duree totale depend d'abord de la complexite du dossier "
        "(interventions, agents, garanties), puis des caracteristiques metier comme la cause, l'outil "
        "d'assistance et certaines dimensions d'organisation."
    )


# --- ONGLET 4 : MACHINE LEARNING ---
with onglets[3]:
    st.subheader("Predictions de Machine Learning")

    st.markdown(
        """
        Le notebook Machine Learning compare Random Forest, LightGBM et XGBoost sur une cible
        `log1p(duree_totale_dossier)`. Les predictions sont ensuite reconverties en secondes
        avec `expm1`. Le meilleur modele du notebook est un LightGBM optimise.
        """
    )

    nb_k1, nb_k2, nb_k3, nb_k4 = st.columns(4)
    nb_k1.metric("Train notebook", "68,994")
    nb_k2.metric("Validation notebook", "17,249")
    nb_k3.metric("MAE LightGBM opt.", "424.75 sec")
    nb_k4.metric("Gain vs baseline", "54.04%")

    models, results_df, best_name, features, num_features, cat_features, model_data, importances_df = train_duration_models(version_key=2)
    if not models:
        st.warning("Impossible d'entrainer les modeles : base groupee ou cible indisponible.")
    else:
        perf_col, comment_col = st.columns([2, 1])
        with perf_col:
            st.markdown("### Benchmark reentraine dans l'application")
            st.dataframe(results_df.round(3), use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=results_df, x="MAE_secondes", y="Modele", color="#9ecae1", ax=ax)
            ax.set_title("Comparaison des modeles par MAE")
            ax.set_xlabel("MAE validation (secondes)")
            ax.set_ylabel("")
            st.pyplot(fig)
            plt.close(fig)
        with comment_col:
            st.metric("Meilleur modele app", best_name)
            st.metric("MAE app", f"{results_df.iloc[0]['MAE_secondes']:,.0f} sec")
            st.metric("R2 app", f"{results_df.iloc[0]['R2']:.3f}")
            st.info(
                "Le classement peut varier selon les librairies disponibles et la base chargee. "
                "L'application retient automatiquement le modele avec la MAE la plus faible."
            )

        if not importances_df.empty:
            st.markdown("### Importance des variables")
            best_imp = importances_df[importances_df["Modele"] == best_name].sort_values("importance", ascending=False).head(10)
            if best_imp.empty:
                best_imp = importances_df.sort_values("importance", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(data=best_imp.sort_values("importance"), x="importance", y="variable", color="teal", ax=ax)
            ax.set_title("Top variables du modele ML")
            ax.set_xlabel("Importance")
            ax.set_ylabel("")
            st.pyplot(fig)
            plt.close(fig)
            st.info("Le notebook ML identifie les variables de volume/complexite et les caracteristiques metier comme facteurs decisifs.")

        st.markdown("### Segmentation metier")
        seg = pd.DataFrame(
            {
                "Indicateur": ["Seuil P95 erreur notebook", "Validation - Standard", "Validation - Complexe", "Inference - Standard", "Inference - Complexe"],
                "Valeur": ["1,574 sec", "16,386", "863", "9,740", "2,166"],
                "Commentaire": [
                    "Seuil utilise pour reperer les dossiers ou l'erreur ou la duree attendue devient critique.",
                    "La majorite des dossiers reste dans le segment standard.",
                    "Le segment complexe concentre les cas atypiques ou difficiles a predire.",
                    "Prediction sur dossiers incomplets classes standard.",
                    "Prediction sur dossiers incomplets classes complexes.",
                ],
            }
        )
        st.dataframe(seg, use_container_width=True)

        st.markdown("---")
        st.markdown("### Interface de prediction de duree")
        selected_model = st.selectbox(
            "Modele a utiliser pour la simulation",
            list(models.keys()),
            index=list(models.keys()).index(best_name),
        )

        user_values = {}
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Complexite du dossier**")
            if "nb_interventions" in features:
                user_values["nb_interventions"] = st.number_input(
                    "Nombre d'interventions",
                    min_value=1,
                    max_value=300,
                    value=int(pd.to_numeric(model_data["nb_interventions"], errors="coerce").median()),
                )
            if "nb_agents" in features:
                user_values["nb_agents"] = st.number_input(
                    "Nombre d'agents",
                    min_value=1,
                    max_value=100,
                    value=int(pd.to_numeric(model_data["nb_agents"], errors="coerce").median()),
                )
            if "nb_garanties_dossier" in features:
                user_values["nb_garanties_dossier"] = st.number_input(
                    "Nombre de garanties",
                    min_value=0,
                    max_value=20,
                    value=int(pd.to_numeric(model_data["nb_garanties_dossier"], errors="coerce").median()),
                )

        with col_b:
            st.markdown("**Organisation et delais**")
            if "part_teletravail" in features:
                user_values["part_teletravail"] = st.slider(
                    "Part teletravail",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pd.to_numeric(model_data["part_teletravail"], errors="coerce").median()),
                    step=0.05,
                )
            if "experience_moyenne" in features:
                user_values["experience_moyenne"] = st.number_input(
                    "Experience moyenne",
                    min_value=0.0,
                    max_value=5000.0,
                    value=float(pd.to_numeric(model_data["experience_moyenne"], errors="coerce").median()),
                    step=10.0,
                )
            if "delai_premier_traitement" in features:
                user_values["delai_premier_traitement"] = st.number_input(
                    "Delai premier traitement",
                    min_value=0.0,
                    max_value=5000.0,
                    value=float(pd.to_numeric(model_data["delai_premier_traitement"], errors="coerce").median()),
                    step=1.0,
                )
            if "mois_premier_traitement" in features:
                user_values["mois_premier_traitement"] = st.number_input(
                    "Mois premier traitement",
                    min_value=1,
                    max_value=12,
                    value=int(pd.to_numeric(model_data["mois_premier_traitement"], errors="coerce").median()),
                )

        with col_c:
            st.markdown("**Caracteristiques qualitatives**")
            for col in cat_features:
                values = sorted(model_data[col].dropna().astype(str).unique().tolist())
                if values:
                    default_value = str(model_data[col].mode().iloc[0])
                    default_index = values.index(default_value) if default_value in values else 0
                    user_values[col] = st.selectbox(col, values, index=default_index)

        for col in features:
            if col not in user_values:
                if col in num_features:
                    user_values[col] = float(pd.to_numeric(model_data[col], errors="coerce").median())
                else:
                    user_values[col] = str(model_data[col].mode().iloc[0])

        if st.button("Predire la duree totale", type="primary", use_container_width=True):
            input_df = pd.DataFrame([user_values])[features]
            pred_log = models[selected_model].predict(input_df)[0]
            pred_seconds = max(float(np.expm1(pred_log)), 0.0)
            pred_minutes = pred_seconds / 60
            pred_hours = pred_seconds / 3600

            r1, r2, r3 = st.columns(3)
            r1.metric("Duree predite", f"{pred_seconds:,.0f} sec")
            r2.metric("Equivalent minutes", f"{pred_minutes:,.1f} min")
            r3.metric("Equivalent heures", f"{pred_hours:,.2f} h")

            ref_median = model_data["duree_totale_dossier"].median()
            position = min(pred_seconds / max(ref_median * 3, 1), 1)
            st.progress(float(position))

            q25 = model_data["duree_totale_dossier"].quantile(0.25)
            q90 = model_data["duree_totale_dossier"].quantile(0.90)
            q95 = model_data["duree_totale_dossier"].quantile(0.95)
            if pred_seconds >= q95:
                st.error("Dossier tres long probable : la prediction depasse le P95 observe.")
            elif pred_seconds >= q90:
                st.warning("Dossier potentiellement long : la duree predite se situe dans le haut de la distribution.")
            elif pred_seconds <= q25:
                st.success("Dossier probablement court au regard de la distribution observee.")
            else:
                st.info("Dossier situe dans une zone intermediaire de duree predite.")

        st.markdown("### Variables utilisees par le modele")
        st.write(", ".join(features))
