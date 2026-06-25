import os
import math
import requests as req
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import httpx
import io
from fpdf import FPDF
from datetime import datetime, date, timedelta
from supabase import create_client, Client

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Altileo - Audit Energetique",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. DESIGN SYSTEM (CSS)
# ============================================================
st.markdown("""<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ========== Global ========== */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
/* ========== Sidebar ========== */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { font-size: 0.65rem !important; text-transform: uppercase; letter-spacing: 2px; color: var(--text-color) !important; font-weight: 600 !important; margin-top: 0.6rem !important; margin-bottom: 0.4rem !important; padding-bottom: 0.3rem; border-bottom: 1px solid var(--secondary-background-color); opacity: 0.7; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size: 0.85rem !important; font-weight: 600 !important; color: var(--text-color) !important; letter-spacing: 0.3px; margin-bottom: 0.5rem !important; }
/* ========== Header banner ========== */
.app-header { background: linear-gradient(135deg, #1B3A5C 0%, #264d73 60%, #1B3A5C 100%); padding: 1.6rem 2rem; border-radius: 6px; margin-bottom: 1.2rem; }
.app-header h1 { font-size: 1.5rem; font-weight: 700; color: #ffffff; letter-spacing: 3px; margin: 0; }
.app-header p { font-size: 0.8rem; color: #a3bdd4; margin: 0.3rem 0 0 0; font-weight: 400; letter-spacing: 0.4px; }
/* ========== KPI Metrics ========== */
div[data-testid="stMetric"] { background-color: var(--secondary-background-color); padding: 1rem 1.2rem; border-radius: 6px; border: 1px solid var(--background-color); border-left: 4px solid #2D8C5A; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-color) !important; font-weight: 600 !important; opacity: 0.8; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; color: var(--text-color) !important; }
/* ========== Tabs ========== */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid var(--secondary-background-color); background-color: transparent; }
.stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.78rem; letter-spacing: 0.2px; padding: 0.7rem 1.4rem; color: var(--text-color); border-bottom: 2px solid transparent; margin-bottom: -2px; background-color: transparent !important; opacity: 0.7; }
.stTabs [aria-selected="true"] { color: #2D8C5A !important; border-bottom-color: #2D8C5A !important; font-weight: 600; opacity: 1; }
/* ========== Buttons ========== */
button[data-testid="stBaseButton-primary"] { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.78rem !important; letter-spacing: 0.4px !important; border-radius: 5px !important; }
button[data-testid="stBaseButton-secondary"] { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.75rem !important; border-radius: 5px !important; }
/* ========== Dataframes ========== */
[data-testid="stDataFrame"] { border: 1px solid var(--secondary-background-color); border-radius: 5px; overflow: hidden; }
/* ========== Alert / Info boxes ========== */
.stAlert { border-radius: 5px; font-size: 0.83rem; }
/* ========== Dividers ========== */
hr { border-color: var(--secondary-background-color) !important; }
/* ========== Expander ========== */
[data-testid="stExpander"] details summary p { font-weight: 500 !important; font-size: 0.82rem !important; color: var(--text-color); }
/* ========== Plotly toolbar cleanup ========== */
.modebar-group { background-color: transparent !important; }
/* ========== Section titles in main area ========== */
.section-title { font-size: 0.95rem; font-weight: 600; color: var(--text-color); margin: 1rem 0 0.8rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid var(--secondary-background-color); }
/* ========== Stat card for monitoring ========== */
.stat-card { background-color: var(--secondary-background-color); border: 1px solid var(--background-color); border-radius: 6px; padding: 0.8rem 1rem; text-align: center; }
.stat-card .stat-label { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-color); font-weight: 600; opacity: 0.7; }
.stat-card .stat-value { font-size: 1.1rem; font-weight: 700; color: var(--text-color); margin-top: 0.2rem; }
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# Matplotlib global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'text.color': '#8DA0B3',
    'axes.labelcolor': '#8DA0B3',
    'xtick.color': '#8DA0B3',
    'ytick.color': '#8DA0B3',
})

# ============================================================
# 3. IDENTIFIANTS SUPABASE
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ============================================================
# 4. FONCTIONS : AUTO-DETECTION
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_available_tables() -> list:
    """Detecte les tables disponibles via le schema PostgREST."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return ["mesures_fady"]
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{SUPABASE_URL}/rest/v1/",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
        if resp.status_code == 200:
            spec = resp.json()
            tables = []
            for path in spec.get('paths', {}).keys():
                name = path.strip('/')
                if name and not name.startswith('rpc/'):
                    tables.append(name)
            if not tables:
                for name in spec.get('definitions', {}).keys():
                    tables.append(name)
            if tables:
                return sorted(tables)
    except Exception:
        pass
    return ["mesures_fady"]


@st.cache_data(ttl=120, show_spinner=False)
def fetch_available_sensors(table_name: str) -> list:
    """Detecte les capteurs uniques dans une table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Echantillonner debut et fin de la table pour couvrir tous les capteurs
        resp1 = supabase.table(table_name).select('capteur')\
            .order('timestamp', desc=False).limit(500).execute()
        resp2 = supabase.table(table_name).select('capteur')\
            .order('timestamp', desc=True).limit(500).execute()
        sensors = set()
        for resp in [resp1, resp2]:
            if resp.data:
                sensors.update(row['capteur'] for row in resp.data if 'capteur' in row)
        return sorted(sensors)
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_table_date_range(table_name: str) -> tuple:
    """Detecte la premiere et la derniere date disponibles dans la table."""
    default_start = date(2025, 11, 1)
    default_end = date(2026, 3, 31)
    if not SUPABASE_URL or not SUPABASE_KEY:
        return default_start, default_end
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp_min = supabase.table(table_name).select('timestamp').order('timestamp', desc=False).limit(1).execute()
        resp_max = supabase.table(table_name).select('timestamp').order('timestamp', desc=True).limit(1).execute()
        
        d_min, d_max = default_start, default_end
        if resp_min.data and 'timestamp' in resp_min.data[0]:
            try:
                d_min = pd.to_datetime(resp_min.data[0]['timestamp']).date()
            except Exception:
                pass
                
        if resp_max.data and 'timestamp' in resp_max.data[0]:
            try:
                d_max = pd.to_datetime(resp_max.data[0]['timestamp']).date()
            except Exception:
                pass
                
        return d_min, d_max
    except Exception:
        return default_start, default_end

# ============================================================
# 5. FONCTIONS : DONNEES ANALYSE (Logique Metier)
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_supabase_data(start_date: str, end_date: str, table_name: str, sensors: tuple):
    """Recupere les donnees brutes depuis Supabase avec pagination."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Identifiants Supabase introuvables.")
        return pd.DataFrame()

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    start_ts = f"{start_date}T00:00:00"
    end_ts = f"{end_date}T23:59:59"
    all_data, page_size, start_range = [], 1000, 0

    while True:
        response = supabase.table(table_name).select('timestamp, capteur, valeur')\
            .in_('capteur', list(sensors)).gte('timestamp', start_ts)\
            .lte('timestamp', end_ts).order('timestamp', desc=False)\
            .range(start_range, start_range + page_size - 1).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        start_range += len(data)
        if len(data) < page_size:
            break

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


@st.cache_data(show_spinner=False)
def process_data(df: pd.DataFrame, sensor_comp: str, sensor_cvc: str, tarifs: dict) -> pd.DataFrame:
    """Calcule puissance, energie, couts par categorie et plage tarifaire."""
    if df.empty:
        return df

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['capteur', 'timestamp']).reset_index(drop=True)
    processed_dfs = []
    cos_phi = tarifs['cos_phi']

    for capteur in [sensor_comp, sensor_cvc]:
        df_sensor = df[df['capteur'] == capteur].copy()
        if df_sensor.empty:
            continue

        # Gestion des gaps (clipping a 30 min max)
        df_sensor['delta_hours'] = df_sensor['timestamp'].diff().dt.total_seconds() / 3600.0
        df_sensor['delta_hours'] = df_sensor['delta_hours'].fillna(0).clip(upper=30.0 / 60.0)

        if capteur == sensor_comp:
            df_sensor['puissance_kw'] = (math.sqrt(3) * 400 * df_sensor['valeur'] * cos_phi) / 1000.0
            df_sensor['categorie_conso'] = 'compresseur'
        else:
            df_sensor['puissance_kw'] = (230 * df_sensor['valeur'] * 1.0) / 1000.0
            df_sensor['categorie_conso'] = 'hvac'
            df_sensor.loc[df_sensor['valeur'] > 6.0, 'categorie_conso'] = 'resistance'

        # Integration trapezoidale
        df_sensor['puissance_moyenne'] = (
            df_sensor['puissance_kw'] + df_sensor['puissance_kw'].shift(1).fillna(df_sensor['puissance_kw'])
        ) / 2.0
        df_sensor['energie_kwh'] = df_sensor['puissance_moyenne'] * df_sensor['delta_hours']

        hour = df_sensor['timestamp'].dt.hour
        month = df_sensor['timestamp'].dt.month

        df_sensor['type_heure'] = 'HP'
        df_sensor.loc[(hour >= 22) | (hour < 6), 'type_heure'] = 'HC'

        df_sensor['saison'] = 'ete'
        df_sensor.loc[month.isin([11, 12, 1, 2, 3]), 'saison'] = 'hiver'
        df_sensor['date'] = df_sensor['timestamp'].dt.date

        tarif = pd.Series(tarifs['hp_ete'], index=df_sensor.index)
        tarif.loc[(df_sensor['saison'] == 'hiver') & (df_sensor['type_heure'] == 'HP')] = tarifs['hp_hiv']
        tarif.loc[(df_sensor['saison'] == 'hiver') & (df_sensor['type_heure'] == 'HC')] = tarifs['hc_hiv']
        tarif.loc[(df_sensor['saison'] == 'ete') & (df_sensor['type_heure'] == 'HC')] = tarifs['hc_ete']

        df_sensor['cout_euros'] = df_sensor['energie_kwh'] * (tarif + tarifs['turpe'] + tarifs['taxes'])
        processed_dfs.append(df_sensor)

    return pd.concat(processed_dfs, ignore_index=True)


@st.cache_data(show_spinner=False)
def generate_detailed_dataframes(df: pd.DataFrame):
    """Genere les bilans journalier et hebdomadaire."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --- Journalier ---
    daily_records = []
    for d in sorted(df['date'].unique()):
        df_d = df[df['date'] == d]

        def get_val(cat, th):
            return df_d[(df_d['categorie_conso'] == cat) & (df_d['type_heure'] == th)]['energie_kwh'].sum()

        c_hp, c_hc = get_val('compresseur', 'HP'), get_val('compresseur', 'HC')
        r_hp, r_hc = get_val('resistance', 'HP'), get_val('resistance', 'HC')
        v_hp, v_hc = get_val('hvac', 'HP'), get_val('hvac', 'HC')
        tot = c_hp + c_hc + r_hp + r_hc + v_hp + v_hc

        daily_records.append({
            'Date': str(d),
            'Comp. HP (kWh)': round(c_hp, 2), 'Comp. HC (kWh)': round(c_hc, 2),
            'Resist. HP (kWh)': round(r_hp, 2), 'Resist. HC (kWh)': round(r_hc, 2),
            'CVC HP (kWh)': round(v_hp, 2), 'CVC HC (kWh)': round(v_hc, 2),
            'Total (kWh)': round(tot, 2)
        })

    # --- Hebdomadaire ---
    df_w = df.copy()
    df_w['timestamp'] = pd.to_datetime(df_w['timestamp'])
    df_w['week'] = df_w['timestamp'].dt.to_period('W-SUN').apply(
        lambda r: f"{r.start_time.strftime('%d/%m/%Y')} - {r.end_time.strftime('%d/%m/%Y')}"
    )
    weekly_records = []

    for w in sorted(df_w['week'].unique()):
        df_w_data = df_w[df_w['week'] == w]

        def get_val_w(cat, th):
            return df_w_data[(df_w_data['categorie_conso'] == cat) & (df_w_data['type_heure'] == th)]['energie_kwh'].sum()

        c_hp, c_hc = get_val_w('compresseur', 'HP'), get_val_w('compresseur', 'HC')
        r_hp, r_hc = get_val_w('resistance', 'HP'), get_val_w('resistance', 'HC')
        v_hp, v_hc = get_val_w('hvac', 'HP'), get_val_w('hvac', 'HC')
        tot = c_hp + c_hc + r_hp + r_hc + v_hp + v_hc

        weekly_records.append({
            'Semaine': w,
            'Comp. HP (kWh)': round(c_hp, 2), 'Comp. HC (kWh)': round(c_hc, 2),
            'Resist. HP (kWh)': round(r_hp, 2), 'Resist. HC (kWh)': round(r_hc, 2),
            'CVC HP (kWh)': round(v_hp, 2), 'CVC HC (kWh)': round(v_hc, 2),
            'Total (kWh)': round(tot, 2)
        })

    return pd.DataFrame(daily_records), pd.DataFrame(weekly_records)


def calculate_period_summary(df: pd.DataFrame, saison: str, tarifs: dict) -> dict:
    """Calcule la synthese mensuelle pour une saison (HT)."""
    if df.empty:
        return {}

    df['heure'] = df['timestamp'].dt.hour
    jour_kw = df.groupby(['type_heure', 'heure'])['puissance_kw'].mean().reset_index()

    kwh_jour_hp = jour_kw[jour_kw['type_heure'] == 'HP']['puissance_kw'].sum()
    kwh_jour_hc = jour_kw[jour_kw['type_heure'] == 'HC']['puissance_kw'].sum()

    kwh_mois_hp = kwh_jour_hp * 30.0
    kwh_mois_hc = kwh_jour_hc * 30.0

    tarif_hp = tarifs['hp_hiv'] if saison == 'hiver' else tarifs['hp_ete']
    tarif_hc = tarifs['hc_hiv'] if saison == 'hiver' else tarifs['hc_ete']

    f_hp = kwh_mois_hp * tarif_hp
    f_hc = kwh_mois_hc * tarif_hc
    turpe = (kwh_mois_hp + kwh_mois_hc) * tarifs['turpe']
    taxes = (kwh_mois_hp + kwh_mois_hc) * tarifs['taxes']

    total_ht = f_hp + f_hc + turpe + taxes

    return {
        'kwh_mois_hp': kwh_mois_hp, 'kwh_mois_hc': kwh_mois_hc,
        'kwh_mois_total': kwh_mois_hp + kwh_mois_hc,
        'tarif_hp': tarif_hp, 'tarif_hc': tarif_hc,
        'tarif_turpe': tarifs['turpe'], 'tarif_taxes': tarifs['taxes'],
        'f_hp': f_hp, 'f_hc': f_hc, 'turpe': turpe, 'taxes': taxes,
        'total_ht': total_ht
    }

# ============================================================
# 6. FONCTIONS : MONITORING
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_monitoring_data(table_name: str, sensors: tuple, start_ts: str, end_ts: str):
    """Recupere les donnees brutes pour le monitoring (cache court)."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return pd.DataFrame()
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        all_data, page_size, start_range = [], 1000, 0

        while True:
            response = supabase.table(table_name).select('timestamp, capteur, valeur')\
                .in_('capteur', list(sensors)).gte('timestamp', start_ts)\
                .lte('timestamp', end_ts).order('timestamp', desc=False)\
                .range(start_range, start_range + page_size - 1).execute()
            data = response.data
            if not data:
                break
            all_data.extend(data)
            start_range += len(data)
            if len(data) < page_size:
                break

        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ============================================================
# 6b. FONCTIONS : NORD POOL SPOT
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nordpool_prices(start_date_str: str, end_date_str: str):
    """Recupere les prix Day-Ahead France depuis l'API Nord Pool.
    Retourne un DataFrame avec colonnes : date, hour (0-23), price_eur_mwh.
    Les quart-horaires sont moyennes par heure."""
    records = []
    current = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    while current <= end:
        url = (
            f"https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
            f"?date={current}&market=DayAhead&deliveryArea=FR&currency=EUR"
        )
        try:
            resp = req.get(url, timeout=15)
            if resp.status_code == 200:
                entries = resp.json().get('multiAreaEntries', [])
                # 96 quart-horaires -> 24 heures (moyenne de 4)
                for i in range(0, min(len(entries), 96), 4):
                    chunk = entries[i:i + 4]
                    prices = [e['entryPerArea'].get('FR', 0) for e in chunk if 'entryPerArea' in e]
                    if prices:
                        records.append({
                            'date': current.isoformat(),
                            'hour': i // 4,
                            'price_eur_mwh': sum(prices) / len(prices)
                        })
        except Exception:
            pass
        current += timedelta(days=1)

    return pd.DataFrame(records) if records else pd.DataFrame()

# ============================================================
# 7. INTERFACE : EN-TETE
# ============================================================

st.markdown("""
<div class="app-header">
    <h1>ALTILEO</h1>
    <p>Plateforme d'audit energetique et de modelisation financiere</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 8. INTERFACE : BARRE LATERALE
# ============================================================

with st.sidebar:
    st.markdown("## Altileo PRO")

    # --- Source de donnees ---
    st.markdown("### Source de donnees")
    available_tables = fetch_available_tables()
    t_name = st.selectbox(
        "Table",
        available_tables,
        index=available_tables.index("mesures_fady") if "mesures_fady" in available_tables else 0,
        label_visibility="collapsed"
    )

    detected_sensors = fetch_available_sensors(t_name)

    # --- Capteurs d'analyse ---
    st.markdown("### Capteurs d'analyse")
    if detected_sensors:
        comp_idx = detected_sensors.index("courant_1") if "courant_1" in detected_sensors else 0
        cvc_idx = detected_sensors.index("courant_2") if "courant_2" in detected_sensors else min(1, len(detected_sensors) - 1)
        s_comp = st.selectbox("Capteur froid (compresseur)", detected_sensors, index=comp_idx)
        s_cvc = st.selectbox("Capteur CVC / resistance", detected_sensors, index=cvc_idx)
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            s_comp = st.text_input("Capteur froid", "courant_1")
        with col_c2:
            s_cvc = st.text_input("Capteur CVC", "courant_2")

    # --- Periode d'analyse ---
    st.markdown("### Periode d'analyse")
    min_date, max_date = fetch_table_date_range(t_name)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        d_start = st.date_input("Debut", min_date)
    with col_d2:
        d_end = st.date_input("Fin", max_date)

    # --- Tarifs ---
    with st.expander("Parametres physiques et tarifs"):
        tarifs = {
            'cos_phi': st.number_input("Cos Phi (Froid)", value=0.92, format="%.2f"),
            'hp_hiv': st.number_input("HP Hiver (EUR/kWh)", value=0.12618, format="%.5f"),
            'hc_hiv': st.number_input("HC Hiver (EUR/kWh)", value=0.09779, format="%.5f"),
            'hp_ete': st.number_input("HP Ete (EUR/kWh)", value=0.07869, format="%.5f"),
            'hc_ete': st.number_input("HC Ete (EUR/kWh)", value=0.07424, format="%.5f"),
            'turpe': st.number_input("TURPE (EUR/kWh)", value=0.04000, format="%.5f"),
            'taxes': st.number_input("Taxes CSPE (EUR/kWh)", value=0.02100, format="%.5f"),
        }

    with st.expander("Parametres thermiques (HACCP)"):
        t_consigne = st.number_input("Temperature consigne (C)", value=-18.0, step=0.5)
        t_max = st.number_input("Limite HACCP (C)", value=-15.0, step=0.5)
        pente_rechauffement = st.number_input("Rechauffement (C/h)", value=0.21, step=0.01)
        pente_refroidissement = st.number_input("Refroidissement (C/h)", value=-0.70, step=0.05)

    st.divider()

    # --- Modelisation ---
    st.markdown("### Modelisation Altileo")
    h = st.slider("Delestage HP (heures)", 0.0, 16.0, 8.0, 0.5)
    r = st.slider("Rattrapage HC (%)", 0, 100, 5, 1) / 100.0
    sec = st.slider("Marge de securite (%)", 0, 50, 5, 1) / 100.0
    cop = st.slider("Bonus COP nuit (%)", 0, 50, 5, 1) / 100.0

    abo = st.number_input("Gain abonnement kVA (EUR/mois/ch)", value=0.0)
    saas = st.number_input("Abonnement SaaS (EUR/an/ch)", value=240.0)

    st.divider()

    # --- Deploiement ---
    st.markdown("### Deploiement")
    nb = st.number_input("Nombre de chambres", min_value=1, value=1)
    pr = st.number_input("Prix installation / chambre (EUR)", value=1500.0)
    cee = st.number_input("Prime CEE globale (EUR)", value=0.0)

    st.divider()

    # --- Simulation Spot ---
    st.markdown("### Simulation Spot")
    spot_delest_h = st.slider("Heures delestees / jour", 0, 12, 6)
    spot_margin = st.number_input("Marge fournisseur (EUR/MWh)", value=5.0, step=1.0)
    spot_days = st.slider("Jours d'historique Nord Pool", 30, 180, 90, step=10)

    st.divider()

    # --- Bouton d'analyse ---
    if st.button("Lancer l'analyse", type="primary", use_container_width=True):
        st.session_state['run_analysis'] = True

# ============================================================
# 9. MOTEUR DE CALCUL
# ============================================================

analysis_run = st.session_state.get('run_analysis', False)

# Variables d'analyse (initialisees a None)
df_proc = None
f_ref_an = k_ref_an = gain_an_net = gain_an_brut = 0
k_sauves_an = pct = roi = i_net = 0
s_h = s_e = None

# Variables Spot (initialisees a None ou valeurs par defaut)
spot_df = pd.DataFrame()
res_df = pd.DataFrame()
fig_evo = None
fig_compare = None
fig_monthly = None
n_days_loaded = 0
spot_avg_overall = 0.0
cost_spot_annual = 0.0
cost_altileo_annual = 0.0
kwh_saved_annual = 0.0
gain_spot_brut = 0.0
gain_spot_net = 0.0
gain_vs_actuel = 0.0


if analysis_run:
    with st.spinner("Analyse en cours..."):
        df_raw = fetch_supabase_data(str(d_start), str(d_end), t_name, tuple([s_comp, s_cvc]))

        if df_raw.empty:
            st.warning("Aucune donnee trouvee pour cette periode et cette table.")
            analysis_run = False
        else:
            df_proc = process_data(df_raw, s_comp, s_cvc, tarifs)

            sum_h = calculate_period_summary(
                df_proc[df_proc['saison'] == 'hiver'].copy(), 'hiver', tarifs
            )
            sum_e = calculate_period_summary(
                df_proc[df_proc['saison'] == 'ete'].copy(), 'ete', tarifs
            )

            s_h = sum_h if sum_h else sum_e
            s_e = sum_e if sum_e else sum_h

            # --- Generation des profils mensuels ---
            df_spot_calc = df_proc.copy()
            df_spot_calc['heure'] = df_spot_calc['timestamp'].dt.hour
            df_spot_calc['mois'] = df_spot_calc['timestamp'].dt.month
            
            comp_global = df_spot_calc[df_spot_calc['categorie_conso'] == 'compresseur']\
                .groupby('heure')['puissance_kw'].mean().reindex(range(24), fill_value=0)
            other_global = df_spot_calc[df_spot_calc['categorie_conso'] != 'compresseur']\
                .groupby('heure')['puissance_kw'].mean().reindex(range(24), fill_value=0)
            global_all_hourly = comp_global + other_global
            
            monthly_profiles = {}
            for mois in range(1, 13):
                df_mois = df_spot_calc[df_spot_calc['mois'] == mois]
                if not df_mois.empty:
                    comp_m = df_mois[df_mois['categorie_conso'] == 'compresseur']\
                        .groupby('heure')['puissance_kw'].mean().reindex(range(24), fill_value=0)
                    other_m = df_mois[df_mois['categorie_conso'] != 'compresseur']\
                        .groupby('heure')['puissance_kw'].mean().reindex(range(24), fill_value=0)
                    monthly_profiles[mois] = comp_m + other_m

            # --- Simulation Classique HC/HP sur 12 mois ---
            jours_par_mois = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
            noms_mois = {1: "Janvier", 2: "Fevrier", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin", 7: "Juillet", 8: "Aout", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Decembre"}
            
            hchp_monthly_results = []
            f_ref_tot = 0.0
            k_ref_tot = 0.0
            gain_tot_brut = 0.0
            k_sauves_tot = 0.0

            mois_dispos = sorted(monthly_profiles.keys())
            if not mois_dispos:
                mois_dispos = [date.today().month]

            for m in mois_dispos:
                jours = jours_par_mois[m]
                saison = 'hiver' if m in [11, 12, 1, 2, 3] else 'ete'
                prof = monthly_profiles.get(m, global_all_hourly)
                
                # Separation HP/HC de base (HP de 6h a 22h)
                kwh_hp_jour = sum(prof[hh] for hh in range(6, 22))
                kwh_hc_jour = sum(prof[hh] for hh in range(24) if hh < 6 or hh >= 22)
                
                k_hp_mois = kwh_hp_jour * jours * nb
                k_hc_mois = kwh_hc_jour * jours * nb
                
                t_hp = tarifs['hp_hiv'] if saison == 'hiver' else tarifs['hp_ete']
                t_hc = tarifs['hc_hiv'] if saison == 'hiver' else tarifs['hc_ete']
                
                f_hp_base = k_hp_mois * t_hp
                f_hc_base = k_hc_mois * t_hc
                turpe_base = (k_hp_mois + k_hc_mois) * tarifs['turpe']
                taxes_base = (k_hp_mois + k_hc_mois) * tarifs['taxes']
                total_ht_base = f_hp_base + f_hc_base + turpe_base + taxes_base
                
                k_effaces = k_hp_mois * (h / 16.0)
                k_rattrapes = k_effaces * r * (1.0 - cop) * (1.0 + sec)
                
                k_hp_sim = k_hp_mois - k_effaces
                k_hc_sim = k_hc_mois + k_rattrapes
                
                f_hp_sim = k_hp_sim * t_hp
                f_hc_sim = k_hc_sim * t_hc
                turpe_sim = (k_hp_sim + k_hc_sim) * tarifs['turpe']
                taxes_sim = (k_hp_sim + k_hc_sim) * tarifs['taxes']
                
                total_ht_sim = (f_hp_sim + f_hc_sim + turpe_sim + taxes_sim) - (abo * nb)
                gain_mois = total_ht_base - total_ht_sim
                
                f_ref_tot += total_ht_base
                k_ref_tot += (k_hp_mois + k_hc_mois)
                gain_tot_brut += gain_mois
                k_sauves_tot += (k_effaces - k_rattrapes)
                
                saas_mois = (saas * nb) * (jours / 365.0)
                
                hchp_monthly_results.append({
                    'mois_num': m,
                    'mois_nom': noms_mois[m],
                    'saison': saison,
                    'jours': jours,
                    'cout_base': total_ht_base,
                    'cout_sim': total_ht_sim + saas_mois,
                    'gain_net': gain_mois - saas_mois
                })

            jours_totaux = sum(jours_par_mois[m] for m in mois_dispos)
            saas_tot = (saas * nb) * (jours_totaux / 365.0)
            gain_tot_net = gain_tot_brut - saas_tot
            pct = (gain_tot_brut / f_ref_tot * 100) if f_ref_tot > 0 else 0
            
            i_net = max(0, pr * nb - cee)
            gain_moyen_mensuel = (gain_tot_net / jours_totaux) * (365.0 / 12.0) if jours_totaux > 0 else 0
            roi = (i_net / gain_moyen_mensuel) if gain_moyen_mensuel > 0 else 0

            # Calcul thermique theorique
            derive_max_hchp = h * pente_rechauffement
            temp_max_hchp = t_consigne + derive_max_hchp

            # --- Generation des graphiques (en memoire) ---
            CHART_COLORS = ['#1B3A5C', '#2D8C5A', '#8DA0B3', '#C5CED6']
            CHART_LABELS = ['HC', 'HP', 'TURPE', 'Taxes']

            def draw_chart(s, title):
                fig, ax = plt.subplots(figsize=(6, 5))
                if not s:
                    ax.axis('off')
                    return fig

                k_hp_base = s.get('kwh_mois_hp', 0) * nb
                k_hc_base = s.get('kwh_mois_hc', 0) * nb
                k_eff = k_hp_base * (h / 16.0)
                k_rat = k_eff * r * (1.0 - cop) * (1.0 + sec)

                k_hp_sim = k_hp_base - k_eff
                k_hc_sim = k_hc_base + k_rat

                f_hp_ap = k_hp_sim * s.get('tarif_hp', 0)
                f_hc_ap = k_hc_sim * s.get('tarif_hc', 0)
                t_ap = (k_hp_sim + k_hc_sim) * s.get('tarif_turpe', 0)
                tax_ap = (k_hp_sim + k_hc_sim) * s.get('tarif_taxes', 0)

                b_av = [
                    s.get('f_hc', 0) * nb,
                    s.get('f_hp', 0) * nb,
                    s.get('turpe', 0) * nb,
                    s.get('taxes', 0) * nb
                ]
                b_ap = [f_hc_ap, f_hp_ap, t_ap, tax_ap]

                c_av, c_ap = 0, 0
                for i in range(4):
                    ax.bar('Actuel', b_av[i], bottom=c_av, color=CHART_COLORS[i], width=0.45)
                    ax.bar('Altileo', b_ap[i], bottom=c_ap, color=CHART_COLORS[i], width=0.45)
                    if b_av[i] > 15:
                        ax.text(0, c_av + b_av[i] / 2, f"{b_av[i]:.0f} EUR",
                                ha='center', color='w', fontweight='bold', fontsize=8)
                    if b_ap[i] > 15:
                        ax.text(1, c_ap + b_ap[i] / 2, f"{b_ap[i]:.0f} EUR",
                                ha='center', color='w', fontweight='bold', fontsize=8)
                    c_av += b_av[i]
                    c_ap += b_ap[i]

                ax.set_title(f"Saison {title} (HT)", pad=20, fontweight="bold", fontsize=11, color='#8DA0B3')
                ax.set_ylabel("Euros / mois", fontsize=9, color='#8DA0B3')

                # Ajuster l'axe Y pour eviter que l'annotation soit coupee
                ax.set_ylim(0, max(c_av, c_ap) * 1.25)

                # Annotation gain
                g = c_av - c_ap
                if g > 0:
                    ax.annotate(
                        f"-{g:.0f} EUR\n(-{g / c_av * 100:.1f} %)",
                        xy=(1, c_ap), xytext=(0, c_av * 1.02),
                        arrowprops=dict(arrowstyle="->", color="#8DA0B3", lw=1.8),
                        color="#8DA0B3", fontweight="bold", ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#0E1117", ec="#8DA0B3", lw=1)
                    )

                # Legende
                legend_handles = [Patch(facecolor=c, label=l) for c, l in zip(CHART_COLORS, CHART_LABELS)]
                ax.legend(
                    handles=legend_handles, loc='upper right', fontsize=8,
                    framealpha=0.0, edgecolor='none', labelcolor='#8DA0B3'
                )

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#8DA0B3')
                ax.spines['bottom'].set_color('#8DA0B3')
                ax.tick_params(colors='#8DA0B3')
                fig.tight_layout()

                return fig

            fig_h = draw_chart(s_h, "HIVER")
            fig_e = draw_chart(s_e, "ETE")

# ============================================================
# 10. ONGLETS
# ============================================================

tab_dashboard, tab_details, tab_charts, tab_spot, tab_spot_charts, tab_thermal, tab_monitoring = st.tabs([
    "Simulation Delestage (Contrat HC/HP)",
    "Audit detaille (HC/HP)",
    "Graphiques (HC/HP)",
    "Simulation Delestage (Prix SPOT)",
    "Graphiques (Prix SPOT)",
    "Validation Thermique",
    "Monitoring capteurs"
])

# ----------------------------------------------------------
# ONGLET 1 : INDICATEURS FINANCIERS
# ----------------------------------------------------------
with tab_dashboard:
    if not analysis_run:
        st.info("Configurez les parametres dans le panneau lateral puis lancez l'analyse.")
    else:
        nb_mois = len(mois_dispos)
        st.markdown(f'<p class="section-title">Detail Mensuel ({nb_mois} mois etudies)</p>', unsafe_allow_html=True)
        
        for res in hchp_monthly_results:
            st.markdown(f"**Bilan {res['mois_nom']} ({res['jours']} jours - Tarif {res['saison'].capitalize()})**")
            c1, c2, c3, c4 = st.columns(4)
            c_base = res['cout_base']
            c_sim = res['cout_sim']
            g_net = res['gain_net']
            
            with c1:
                st.metric("HC/HP Classique", f"{c_base:,.0f} EUR")
            with c2:
                # empty column to keep alignment with Spot
                st.write("")
            with c3:
                d = ((c_sim - c_base)/c_base*100) if c_base > 0 else 0
                st.metric("HC/HP AVEC Altileo", f"{c_sim:,.0f} EUR", f"{d:+.1f}% vs Classique", delta_color="inverse")
            with c4:
                st.metric("Gain Net Mensuel", f"{g_net:,.0f} EUR")
            st.write("")

        st.divider()
        st.markdown(f'<p class="section-title">Synthese globale sur la periode etudiee ({jours_totaux} jours)</p>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                label="Facture de reference",
                value=f"{f_ref_tot:,.0f} EUR",
                delta=f"{k_ref_tot:,.0f} kWh",
                delta_color="off"
            )
        with col2:
            st.metric(
                label="Gain Brut d'exploit.",
                value=f"{gain_tot_brut:,.0f} EUR",
                delta="Avant abo SaaS",
                delta_color="off"
            )
        with col3:
            st.metric(
                label="Gain Net d'exploit.",
                value=f"{gain_tot_net:,.0f} EUR",
                delta=f"-{pct:.1f} % (brut)"
            )
        with col4:
            st.metric(
                label="Impact environnemental",
                value=f"{(k_sauves_tot * 0.05) / 1000:,.1f} t",
                delta="CO2 evite"
            )
        with col5:
            st.metric(
                label="Retour investissement",
                value=f"{roi:.1f} mois",
                delta=f"soit {roi/12:.1f} ans",
                delta_color="off"
            )

        st.divider()
        st.markdown('<p class="section-title">Parametres de modelisation retenus</p>', unsafe_allow_html=True)

        param_col1, param_col2 = st.columns(2)
        with param_col1:
            st.markdown(f"**Chambres equipees :** {nb}")
            st.markdown(f"**Delestage journalier :** {h} heures")
        with param_col2:
            st.markdown(f"**Rattrapage thermique :** {r * 100:.0f} % (marge securite {sec * 100:.0f} %)")
            st.markdown(f"**Abonnement SaaS :** {saas * nb:.0f} EUR / an")

        st.divider()
        st.markdown('<p class="section-title">Validation Thermique HACCP</p>', unsafe_allow_html=True)
        if temp_max_hchp > t_max:
            st.error(f"⚠️ **RISQUE SANITAIRE** : La temperature maximale simulee atteint **{temp_max_hchp:.2f} C**, depassant la limite HACCP de **{t_max:.2f} C**.")
        else:
            st.success(f"✅ **CONFORME** : La temperature maximale simulee est de **{temp_max_hchp:.2f} C** (limite: {t_max:.2f} C).")

        st.divider()
        
        # --- Export PDF ---
        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.set_text_color(27, 58, 92)
            pdf.cell(0, 10, "Rapport d'Audit Energetique Altileo", ln=1, align="C")
            
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(5)
            pdf.cell(0, 6, f"Date de l'audit : {date.today().strftime('%d/%m/%Y')}", ln=1)
            try:
                pdf.cell(0, 6, f"Periode analysee : du {d_start.strftime('%d/%m/%Y')} au {d_end.strftime('%d/%m/%Y')}", ln=1)
            except:
                pdf.cell(0, 6, f"Periode analysee : du {d_start} au {d_end}", ln=1)
            pdf.cell(0, 6, f"Nombre de chambres equipees : {nb}", ln=1)
            pdf.cell(0, 6, f"Delestage programme : {h} heures/jour", ln=1)
            
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(27, 58, 92)
            pdf.cell(0, 10, "Resultats Financiers (HT)", ln=1)
            
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 6, f"Facture de reference : {f_ref_an:,.0f} EUR/an", ln=1)
            pdf.cell(0, 6, f"Gain d'exploitation brut : {gain_an_brut:,.0f} EUR/an", ln=1)
            pdf.cell(0, 6, f"Abonnement SaaS : {saas * nb:,.0f} EUR/an", ln=1)
            pdf.cell(0, 6, f"Gain d'exploitation NET : {gain_an_net:,.0f} EUR/an", ln=1)
            pdf.cell(0, 6, f"Retour sur investissement (ROI) : {roi:.1f} mois (soit {roi/12:.1f} ans)", ln=1)
            pdf.cell(0, 6, f"Impact carbone : {(k_sauves_an * 0.05) / 1000:,.1f} tonnes de CO2 evitees/an", ln=1)
            pdf.cell(0, 6, f"Validation HACCP : {'ECHEC' if temp_max_hchp > t_max else 'CONFORME'} (Temp max: {temp_max_hchp:.2f} C)", ln=1)
            
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(27, 58, 92)
            pdf.cell(0, 10, "Modelisation Mensuelle", ln=1)
            
            img_h = io.BytesIO()
            fig_h.savefig(img_h, format='png', bbox_inches='tight')
            pdf.image(img_h, x=10, y=pdf.get_y(), w=90)
            
            img_e = io.BytesIO()
            fig_e.savefig(img_e, format='png', bbox_inches='tight')
            pdf.image(img_e, x=110, y=pdf.get_y(), w=90)
            
            return bytes(pdf.output())
            
        pdf_bytes = create_pdf()
        st.download_button(
            label="📄 Telecharger le rapport PDF",
            data=pdf_bytes,
            file_name=f"Altileo_Audit_{date.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary"
        )

# ----------------------------------------------------------
# ONGLET 2 : AUDIT DETAILLE
# ----------------------------------------------------------
with tab_details:
    if not analysis_run or df_proc is None:
        st.info("Lancez l'analyse pour consulter le detail des consommations.")
    else:
        df_daily, df_weekly = generate_detailed_dataframes(df_proc)

        st.markdown('<p class="section-title">Bilan hebdomadaire</p>', unsafe_allow_html=True)
        st.dataframe(df_weekly, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-title">Bilan journalier</p>', unsafe_allow_html=True)
        st.dataframe(df_daily, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# ONGLET 3 : MODELISATION GRAPHIQUE
# ----------------------------------------------------------
with tab_charts:
    if not analysis_run or s_h is None:
        st.info("Lancez l'analyse pour afficher les graphiques de modelisation.")
    else:
        st.markdown('<p class="section-title">Comparaison mensuelle : Actuel vs Altileo (HT)</p>', unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.pyplot(fig_h, use_container_width=True)
        with col_chart2:
            st.pyplot(fig_e, use_container_width=True)

# ----------------------------------------------------------
# ONGLET 4 : MONITORING CAPTEURS
# ----------------------------------------------------------
with tab_monitoring:
    st.markdown('<p class="section-title">Monitoring des capteurs</p>', unsafe_allow_html=True)

    # --- Controles ---
    mon_col1, mon_col2, mon_col3 = st.columns([3, 2, 1])

    with mon_col1:
        all_sensors = fetch_available_sensors(t_name)
        default_sensors = all_sensors[:2] if len(all_sensors) >= 2 else all_sensors
        mon_sensors = st.multiselect(
            "Capteurs a afficher",
            all_sensors,
            default=default_sensors,
            key="mon_sensors"
        )

    with mon_col2:
        period_options = [
            "Derniere heure",
            "6 dernieres heures",
            "24 dernieres heures",
            "48 dernieres heures",
            "7 derniers jours",
            "30 derniers jours",
            "Personnalisee"
        ]
        period_choice = st.selectbox("Periode", period_options, index=2, key="mon_period")

    with mon_col3:
        st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button("Actualiser", key="mon_refresh", type="secondary")

    # Periode personnalisee
    if period_choice == "Personnalisee":
        mon_date_col1, mon_date_col2 = st.columns(2)
        with mon_date_col1:
            mon_start_date = st.date_input("Du", date.today() - timedelta(days=7), key="mon_start")
        with mon_date_col2:
            mon_end_date = st.date_input("Au", date.today(), key="mon_end")
        mon_start_ts = f"{mon_start_date}T00:00:00"
        mon_end_ts = f"{mon_end_date}T23:59:59"
    else:
        period_deltas = {
            "Derniere heure": timedelta(hours=1),
            "6 dernieres heures": timedelta(hours=6),
            "24 dernieres heures": timedelta(hours=24),
            "48 dernieres heures": timedelta(hours=48),
            "7 derniers jours": timedelta(days=7),
            "30 derniers jours": timedelta(days=30),
        }
        delta = period_deltas[period_choice]
        # Arrondir a la minute pour stabiliser le cache
        now_rounded = datetime.now().replace(second=0, microsecond=0)
        mon_start_ts = (now_rounded - delta).strftime('%Y-%m-%dT%H:%M:%S')
        mon_end_ts = now_rounded.strftime('%Y-%m-%dT%H:%M:%S')

    # Forcer le rafraichissement
    if refresh_clicked:
        fetch_monitoring_data.clear()
        st.rerun()

    # --- Affichage des donnees ---
    if not mon_sensors:
        st.info("Selectionnez au moins un capteur pour afficher les donnees.")
    else:
        df_mon = fetch_monitoring_data(t_name, tuple(mon_sensors), mon_start_ts, mon_end_ts)

        if df_mon.empty:
            st.info("Aucune donnee disponible pour la periode et les capteurs selectionnes.")
        else:
            df_mon['timestamp'] = pd.to_datetime(df_mon['timestamp'])
            df_mon['valeur'] = pd.to_numeric(df_mon['valeur'], errors='coerce')
            df_mon = df_mon.sort_values('timestamp')

            # --- Graphique Plotly ---
            PLOTLY_COLORS = [
                '#1B3A5C', '#2D8C5A', '#C4652B', '#7C3E8C',
                '#B8860B', '#5F7D8E', '#C75D5D', '#4A8C8C'
            ]

            fig = go.Figure()

            for i, sensor in enumerate(mon_sensors):
                df_s = df_mon[df_mon['capteur'] == sensor]
                if df_s.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=df_s['timestamp'],
                    y=df_s['valeur'],
                    name=sensor,
                    mode='lines',
                    line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=1.5),
                    hovertemplate=(
                        '<b>%{fullData.name}</b><br>'
                        'Valeur : %{y:.2f}<br>'
                        '%{x|%d/%m/%Y %H:%M:%S}'
                        '<extra></extra>'
                    )
                ))

            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", size=10, color='var(--text-color)'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(136, 152, 168, 0.2)',
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(136, 152, 168, 0.5)',
                    rangeslider=dict(visible=True, thickness=0.04, bgcolor='rgba(0,0,0,0)'),
                ),
                yaxis=dict(
                    gridwidth=1,
                    title='Valeur',
                    title_font=dict(size=11, color='#7a8a9a')
                )
            )

            st.plotly_chart(fig, use_container_width=True, config={
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            })

            # --- Statistiques par capteur ---
            st.markdown('<p class="section-title">Statistiques</p>', unsafe_allow_html=True)

            stat_cols = st.columns(min(len(mon_sensors), 4))
            for i, sensor in enumerate(mon_sensors):
                df_s = df_mon[df_mon['capteur'] == sensor]
                if df_s.empty:
                    continue
                with stat_cols[i % len(stat_cols)]:
                    st.markdown(f"**{sensor}**")
                    last_val = df_s['valeur'].iloc[-1]
                    st.metric("Derniere valeur", f"{last_val:.2f}")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Min", f"{df_s['valeur'].min():.2f}")
                    sc2.metric("Moy", f"{df_s['valeur'].mean():.2f}")
                    sc3.metric("Max", f"{df_s['valeur'].max():.2f}")

            # Horodatage
            st.caption(f"Derniere actualisation : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# ----------------------------------------------------------
# ONGLET 5 : SIMULATION SPOT (DONNEES NORD POOL)
# ----------------------------------------------------------
with tab_spot:
    st.markdown('<p class="section-title">Simulation contrat Spot -- Donnees Nord Pool France</p>', unsafe_allow_html=True)

    if not analysis_run:
        st.info("Lancez d'abord l'analyse dans l'onglet Simulation Delestage (Contrat HC/HP).")
    else:
        # --- Chargement des prix Nord Pool ---
        spot_end_date = date.today() - timedelta(days=1)
        spot_start_date = spot_end_date - timedelta(days=spot_days)

        with st.spinner(f"Chargement des prix Spot Nord Pool ({spot_days} jours)..."):
            spot_df = fetch_nordpool_prices(spot_start_date.isoformat(), spot_end_date.isoformat())

        if spot_df.empty:
            st.error("Impossible de charger les donnees Nord Pool. Verifiez votre connexion.")
        else:
            n_days_loaded = spot_df['date'].nunique()
            spot_avg_overall = spot_df['price_eur_mwh'].mean()

            st.success(f"{n_days_loaded} jours charges ({spot_start_date.strftime('%d/%m/%Y')} au {spot_end_date.strftime('%d/%m/%Y')}) -- Prix moyen : **{spot_avg_overall:.1f} EUR/MWh**")

            # --- Profil horaire moyen (depuis les vraies donnees) ---
            avg_profile = spot_df.groupby('hour')['price_eur_mwh'].mean().reindex(range(24), fill_value=0)

            # Identifier les heures les plus cheres en moyenne pour le graphique (hors 2h-12h)
            valid_hours = [h for h in range(24) if not (2 <= h < 12)]
            top_hours_avg = avg_profile.loc[valid_hours].nlargest(spot_delest_h).index.tolist()

            bar_colors = ['#E74C3C' if hh in top_hours_avg else '#1B3A5C' for hh in range(24)]

            fig_profile = go.Figure()
            fig_profile.add_trace(go.Bar(
                x=[f"{hh:02d}h" for hh in range(24)],
                y=avg_profile.values,
                marker_color=bar_colors,
                hovertemplate='<b>%{x}</b><br>%{y:.1f} EUR/MWh<extra></extra>'
            ))
            fig_profile.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=10),
                yaxis=dict(title="EUR / MWh", showgrid=True, gridcolor='rgba(136,152,168,0.2)'),
                xaxis=dict(showgrid=False, showline=True, linecolor='rgba(136,152,168,0.5)'),
                title=dict(
                    text=f"Profil horaire moyen Nord Pool FR -- {spot_delest_h}h delestees/jour (en rouge)",
                    font=dict(size=13), x=0.5
                ),
                showlegend=False
            )
            st.plotly_chart(fig_profile, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

            # --- Simulation jour par jour ---
            daily_results = []
            for date_val, day_df in spot_df.groupby('date'):
                prices_day = day_df.set_index('hour')['price_eur_mwh'].reindex(range(24), fill_value=spot_avg_overall)

                try:
                    current_month = pd.to_datetime(date_val).month
                except:
                    current_month = 1
                current_all_hourly = monthly_profiles.get(current_month, global_all_hourly)

                # Recherche gloutonne des N heures les plus rentables a delester
                # en prenant en compte le profil de consommation et le cout du rattrapage
                delest_hours_day = []
                for _ in range(spot_delest_h):
                    best_h = -1
                    best_cost = float('inf')
                    for test_h in range(24):
                        if 2 <= test_h < 12:
                            continue
                        if test_h not in delest_hours_day:
                            test_shed = delest_hours_day + [test_h]
                            e_rem = sum(current_all_hourly[hh] for hh in test_shed)
                            e_rat = e_rem * r * (1.0 - cop) * (1.0 + sec)
                            avail = [hh for hh in range(24) if hh not in test_shed]
                            rat_per_h = e_rat / len(avail) if avail else 0
                            
                            c_test = 0.0
                            for hh in range(24):
                                p_kwh = (prices_day[hh] + spot_margin) / 1000.0 + tarifs['turpe'] + tarifs['taxes']
                                if hh in test_shed:
                                    c_test += 0
                                else:
                                    c_test += (current_all_hourly[hh] + rat_per_h) * p_kwh
                                    
                            if c_test < best_cost:
                                best_cost = c_test
                                best_h = test_h
                    if best_h != -1:
                        delest_hours_day.append(best_h)

                cost_base_day = 0.0
                cost_altileo_day = 0.0
                energy_removed_day = 0.0

                for hh in range(24):
                    price_kwh = (prices_day[hh] + spot_margin) / 1000.0 + tarifs['turpe'] + tarifs['taxes']
                    energy = current_all_hourly[hh]
                    cost_base_day += energy * price_kwh

                    if hh in delest_hours_day:
                        energy_removed_day += energy
                    else:
                        cost_altileo_day += energy * price_kwh

                # Rattrapage thermique
                energy_rattrapage = energy_removed_day * r * (1.0 - cop) * (1.0 + sec)
                available_h = [hh for hh in range(24) if hh not in delest_hours_day]
                ratt_per_h = energy_rattrapage / len(available_h) if available_h else 0
                for hh in available_h:
                    price_kwh = (prices_day[hh] + spot_margin) / 1000.0 + tarifs['turpe'] + tarifs['taxes']
                    cost_altileo_day += ratt_per_h * price_kwh

                # Simulation thermique jour par jour
                current_temp = t_consigne
                max_temp_day = t_consigne
                for hh in range(24):
                    if hh in delest_hours_day:
                        current_temp += pente_rechauffement
                    else:
                        current_temp = max(t_consigne, current_temp + pente_refroidissement)
                    if current_temp > max_temp_day:
                        max_temp_day = current_temp

                # Reference HC/HP sans delestage pour la meme journee
                dt_obj = pd.to_datetime(date_val)
                month_val = dt_obj.month
                saison_val = 'hiver' if month_val in [11, 12, 1, 2, 3] else 'ete'
                cost_hchp_day = 0.0
                for hh in range(24):
                    type_h = 'HC' if (hh >= 22 or hh < 6) else 'HP'
                    if saison_val == 'hiver':
                        t_base = tarifs['hp_hiv'] if type_h == 'HP' else tarifs['hc_hiv']
                    else:
                        t_base = tarifs['hp_ete'] if type_h == 'HP' else tarifs['hc_ete']
                    p_kwh = t_base + tarifs['turpe'] + tarifs['taxes']
                    cost_hchp_day += current_all_hourly[hh] * p_kwh

                daily_results.append({
                    'date': date_val,
                    'cost_hchp': cost_hchp_day,
                    'cost_spot': cost_base_day,
                    'cost_altileo': cost_altileo_day,
                    'avg_price': prices_day.mean(),
                    'max_price': prices_day.max(),
                    'min_price': prices_day.min(),
                    'kwh_saved': energy_removed_day - energy_rattrapage,
                    'max_temp': max_temp_day
                })

            res_df = pd.DataFrame(daily_results)
            res_df['date'] = pd.to_datetime(res_df['date'])
            res_df['gain_jour'] = res_df['cost_spot'] - res_df['cost_altileo']

            # --- Bilan par mois ---
            res_df['mois_str'] = res_df['date'].dt.to_period('M').astype(str)
            monthly = res_df.groupby('mois_str').agg(
                jours=('date', 'count'),
                prix_moyen=('avg_price', 'mean'),
                cout_hchp=('cost_hchp', 'sum'),
                cout_spot=('cost_spot', 'sum'),
                cout_altileo=('cost_altileo', 'sum'),
                kwh_eco=('kwh_saved', 'sum'),
                gain=('gain_jour', 'sum'),
                temp_max=('max_temp', 'max')
            ).reset_index()

            # Application du nb de chambres
            for col in ['cout_hchp', 'cout_spot', 'cout_altileo', 'kwh_eco', 'gain']:
                monthly[col] = monthly[col] * nb
                
            # Totaux de la periode
            total_days = res_df['date'].nunique()
            cost_hchp_tot = res_df['cost_hchp'].sum() * nb
            cost_spot_tot = res_df['cost_spot'].sum() * nb
            cost_altileo_tot = res_df['cost_altileo'].sum() * nb
            kwh_saved_tot = res_df['kwh_saved'].sum() * nb
            gain_spot_brut = cost_spot_tot - cost_altileo_tot
            
            # Prorata SaaS sur la periode analysee
            saas_periode = (saas * nb) * (total_days / 365.0)
            gain_spot_net = gain_spot_brut - saas_periode
            gain_vs_actuel = cost_hchp_tot - (cost_altileo_tot + saas_periode)
            max_temp_spot = res_df['max_temp'].max()

            st.divider()
            st.markdown(f'<p class="section-title">Analyse Mensuelle (Donnees reelles sur {total_days} jours)</p>', unsafe_allow_html=True)
            
            # Affichage d'une carte par mois
            for _, row in monthly.iterrows():
                mois_nom = row['mois_str']
                j = row['jours']
                c_hc = row['cout_hchp']
                c_spot = row['cout_spot']
                c_alt = row['cout_altileo']
                saas_mois = (saas * nb) * (j / 365.0)
                g_net = row['gain'] - saas_mois
                
                st.markdown(f"**Bilan {mois_nom} ({j} jours)**")
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric(label="HC/HP", value=f"{c_hc:,.0f} EUR")
                with k2:
                    d1 = ((c_spot - c_hc)/c_hc*100) if c_hc>0 else 0
                    st.metric(label="Spot SANS Altileo", value=f"{c_spot:,.0f} EUR", delta=f"{d1:+.1f}% vs HC/HP", delta_color="inverse")
                with k3:
                    d2 = ((c_alt + saas_mois - c_hc)/c_hc*100) if c_hc>0 else 0
                    st.metric(label="Spot AVEC Altileo", value=f"{(c_alt + saas_mois):,.0f} EUR", delta=f"{d2:+.1f}% vs HC/HP", delta_color="inverse")
                with k4:
                    st.metric(label="Gain Net Mensuel", value=f"{g_net:,.0f} EUR")
                st.write("") # spacing

            st.divider()
            st.markdown(f'<p class="section-title">Resume de la periode ({spot_start_date.strftime("%d/%m/%Y")} au {spot_end_date.strftime("%d/%m/%Y")})</p>', unsafe_allow_html=True)
            
            c_hc_tot = cost_hchp_tot
            c_spot_tot = cost_spot_tot
            c_alt_tot = cost_altileo_tot + saas_periode
            
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                st.metric("HC/HP Classique", f"{c_hc_tot:,.0f} EUR", f"Sur {total_days} jours", delta_color="off")
            with kpi2:
                d1_tot = ((c_spot_tot - c_hc_tot)/c_hc_tot*100) if c_hc_tot > 0 else 0
                st.metric("Spot SANS Altileo", f"{c_spot_tot:,.0f} EUR", f"{d1_tot:+.1f}% vs HC/HP", delta_color="inverse")
            with kpi3:
                d2_tot = ((c_alt_tot - c_hc_tot)/c_hc_tot*100) if c_hc_tot > 0 else 0
                st.metric("Spot AVEC Altileo", f"{c_alt_tot:,.0f} EUR", f"{d2_tot:+.1f}% vs HC/HP", delta_color="inverse")
            with kpi4:
                st.metric("Gain vs Actuel", f"{gain_vs_actuel:,.0f} EUR", "Spot+Altileo vs HC/HP", delta_color="normal")
            with kpi5:
                st.metric("kWh economises", f"{kwh_saved_tot:,.0f} kWh", f"{(kwh_saved_tot * 0.05) / 1000:,.1f} t CO2 evite", delta_color="normal")

            st.divider()
            if max_temp_spot > t_max:
                st.error(f"⚠️ **RISQUE SANITAIRE** : Sur la periode, la temperature maximale simulee a atteint **{max_temp_spot:.2f} C**, depassant la limite HACCP de **{t_max:.2f} C**.")
            else:
                st.success(f"✅ **CONFORME** : La temperature maximale simulee sur la periode est restee a **{max_temp_spot:.2f} C** (limite: {t_max:.2f} C).")

            # --- Export PDF Spot ---
            def create_spot_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.set_text_color(27, 58, 92)
                pdf.cell(0, 10, "Rapport d'Audit Energetique Altileo (Simulation Spot)", ln=1, align="C")
                
                pdf.set_font("Arial", "", 11)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(5)
                pdf.cell(0, 6, f"Date de l'audit : {date.today().strftime('%d/%m/%Y')}", ln=1)
                pdf.cell(0, 6, f"Nombre de chambres equipees : {nb}", ln=1)
                pdf.cell(0, 6, f"Delestage programme (Spot) : {spot_delest_h} heures/jour", ln=1)
                pdf.cell(0, 6, f"Marge fournisseur Spot : {spot_margin} EUR/MWh", ln=1)
                pdf.cell(0, 6, f"Periode etudiee : {total_days} jours (Moyenne : {spot_avg_overall:.1f} EUR/MWh)", ln=1)
                
                pdf.ln(5)
                pdf.set_font("Arial", "B", 14)
                pdf.set_text_color(27, 58, 92)
                pdf.cell(0, 10, f"Resultats Financiers sur la periode de {total_days} jours (HT)", ln=1)
                
                pdf.set_font("Arial", "", 11)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 6, f"Facture contrat actuel (HC/HP) : {cost_hchp_tot:,.0f} EUR", ln=1)
                pdf.cell(0, 6, f"Facture Spot SANS Altileo : {cost_spot_tot:,.0f} EUR", ln=1)
                pdf.cell(0, 6, f"Facture Spot AVEC Altileo : {(cost_altileo_tot + saas_periode):,.0f} EUR", ln=1)
                pdf.cell(0, 6, f"Gain Altileo sur Spot : {gain_spot_net:,.0f} EUR (Brut : {gain_spot_brut:,.0f} EUR)", ln=1)
                pdf.cell(0, 6, f"Gain total vs contrat actuel : {gain_vs_actuel:,.0f} EUR", ln=1)
                pdf.cell(0, 6, f"Impact carbone : {(kwh_saved_tot * 0.05) / 1000:,.2f} tonnes de CO2 evitees", ln=1)
                pdf.cell(0, 6, f"Validation HACCP : {'ECHEC' if max_temp_spot > t_max else 'CONFORME'} (Temp max: {max_temp_spot:.2f} C)", ln=1)
                
                return bytes(pdf.output())

            st.divider()
            spot_pdf_bytes = create_spot_pdf()
            st.download_button(
                label="📄 Telecharger le rapport PDF Spot",
                data=spot_pdf_bytes,
                file_name=f"Altileo_Audit_Spot_{date.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                type="primary"
            )

            # --- Graphique Barres Mensuel ---
            st.divider()
            st.markdown('<p class="section-title">Evolution Mensuelle de la Facture</p>', unsafe_allow_html=True)
            monthly_table = monthly.copy()
            monthly_table.columns = ['Mois', 'Jours', 'Prix moy (EUR/MWh)', 'HC/HP de base (EUR)', 'Spot seul (EUR)', 'Spot+Altileo (EUR)', 'kWh eco.', 'Gain Altileo (EUR)', 'Temp Max (C)']
            for c in ['Prix moy (EUR/MWh)', 'HC/HP de base (EUR)', 'Spot seul (EUR)', 'Spot+Altileo (EUR)', 'kWh eco.', 'Gain Altileo (EUR)', 'Temp Max (C)']:
                monthly_table[c] = monthly_table[c].round(1)
            
            # --- Graphique Barres Mensuel ---
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_table['Mois'], y=monthly_table['Spot seul (EUR)'],
                name='Spot SANS Altileo', marker_color='#E74C3C'
            ))
            fig_monthly.add_trace(go.Bar(
                x=monthly_table['Mois'], y=monthly_table['Spot+Altileo (EUR)'],
                name='Spot AVEC Altileo', marker_color='#2D8C5A'
            ))
            fig_monthly.update_layout(
                barmode='group',
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=10),
                yaxis=dict(title="Coût Mensuel (EUR)", showgrid=True, gridcolor='rgba(136,152,168,0.2)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_monthly, use_container_width=True, config={'displaylogo': False})
            
            st.dataframe(monthly_table, use_container_width=True, hide_index=True)

            # --- Figures pour l'onglet graphique ---
            fig_evo = go.Figure()
            fig_evo.add_trace(go.Scatter(
                x=res_df['date'], y=res_df['avg_price'],
                mode='lines', name='Prix moyen',
                line=dict(color='#1B3A5C', width=1.5),
                hovertemplate='%{x|%d/%m/%Y}<br>Moy : %{y:.1f} EUR/MWh<extra></extra>'
            ))
            fig_evo.add_trace(go.Scatter(
                x=res_df['date'], y=res_df['max_price'],
                mode='lines', name='Prix max',
                line=dict(color='#E74C3C', width=1, dash='dot'),
                hovertemplate='%{x|%d/%m/%Y}<br>Max : %{y:.1f} EUR/MWh<extra></extra>'
            ))
            fig_evo.add_trace(go.Scatter(
                x=res_df['date'], y=res_df['min_price'],
                mode='lines', name='Prix min',
                line=dict(color='#2D8C5A', width=1, dash='dot'),
                hovertemplate='%{x|%d/%m/%Y}<br>Min : %{y:.1f} EUR/MWh<extra></extra>'
            ))
            fig_evo.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=10),
                yaxis=dict(title="EUR / MWh", showgrid=True, gridcolor='rgba(136,152,168,0.2)'),
                xaxis=dict(showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=350
            )

            fig_compare = go.Figure()
            scenarios = ["Contrat actuel\n(HC/HP)", "Spot seul\n(sans Altileo)", "Spot + Altileo"]
            values = [f_ref_an, cost_spot_annual, cost_altileo_annual + saas * nb]
            colors_bar = ['#8DA0B3', '#C4652B', '#2D8C5A']
            fig_compare.add_trace(go.Bar(
                x=scenarios, y=values, marker_color=colors_bar,
                text=[f"{v:,.0f} EUR" for v in values], textposition='outside',
                hovertemplate='<b>%{x}</b><br>%{y:,.0f} EUR/an<extra></extra>'
            ))
            fig_compare.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=11),
                yaxis=dict(title="EUR / an (HT)", showgrid=True, gridcolor='rgba(136,152,168,0.2)'),
                xaxis=dict(showgrid=False), showlegend=False, height=400
            )

            # --- Textes et hovers pour comparaison mensuelle ---
            text_hchp = []
            text_spot = []
            text_altileo = []
            hover_hchp = []
            hover_spot = []
            hover_altileo = []

            for idx, row in monthly.iterrows():
                c_base = row['HC/HP de base (EUR)']
                c_spot = row['Spot seul (EUR)']
                c_altileo = row['Spot+Altileo (EUR)']
                
                text_hchp.append(f"{c_base:,.0f} €")
                hover_hchp.append(f"<b>{row['Mois']}</b><br>HC/HP de base : {c_base:,.0f} EUR<extra></extra>")
                
                if c_base > 0:
                    pct_spot = (c_spot - c_base) / c_base * 100
                    pct_altileo = (c_altileo - c_base) / c_base * 100
                    text_spot.append(f"{c_spot:,.0f} €<br>({pct_spot:+.1f}%)")
                    text_altileo.append(f"{c_altileo:,.0f} €<br>({pct_altileo:+.1f}%)")
                else:
                    text_spot.append(f"{c_spot:,.0f} €")
                    text_altileo.append(f"{c_altileo:,.0f} €")
                    pct_spot = 0
                    pct_altileo = 0
                
                pct_vs_spot = (c_altileo - c_spot) / c_spot * 100 if c_spot > 0 else 0
                
                hover_spot.append(
                    f"<b>{row['Mois']}</b><br>"
                    f"Spot seul : {c_spot:,.0f} EUR<br>"
                    f"Diff vs HC/HP : {pct_spot:+.1f}%<extra></extra>"
                )
                hover_altileo.append(
                    f"<b>{row['Mois']}</b><br>"
                    f"Spot + Altileo : {c_altileo:,.0f} EUR<br>"
                    f"Gain vs HC/HP : {pct_altileo:+.1f}%<br>"
                    f"Gain vs Spot seul : {pct_vs_spot:+.1f}%<extra></extra>"
                )

            # --- Graphique de comparaison mensuelle ---
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly['Mois'],
                y=monthly['HC/HP de base (EUR)'],
                name='HC/HP de base (sans delestage)',
                marker_color='#8DA0B3',
                text=text_hchp,
                textposition='outside',
                hovertemplate='%{customdata}'
            ))
            fig_monthly.data[-1].customdata = hover_hchp
            
            fig_monthly.add_trace(go.Bar(
                x=monthly['Mois'],
                y=monthly['Spot seul (EUR)'],
                name='Spot seul (sans delestage)',
                marker_color='#C4652B',
                text=text_spot,
                textposition='outside',
                hovertemplate='%{customdata}'
            ))
            fig_monthly.data[-1].customdata = hover_spot
            
            fig_monthly.add_trace(go.Bar(
                x=monthly['Mois'],
                y=monthly['Spot+Altileo (EUR)'],
                name='Spot + Altileo',
                marker_color='#2D8C5A',
                text=text_altileo,
                textposition='outside',
                hovertemplate='%{customdata}'
            ))
            fig_monthly.data[-1].customdata = hover_altileo

            fig_monthly.update_layout(
                barmode='group',
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=11),
                yaxis=dict(title="Cout mensuel (EUR HT)", showgrid=True, gridcolor='rgba(136,152,168,0.2)'),
                xaxis=dict(showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450
            )

# ----------------------------------------------------------
# ONGLET 6 : GRAPHIQUES (PRIX SPOT)
# ----------------------------------------------------------
with tab_spot_charts:
    st.markdown('<p class="section-title">Visualisation de la simulation Spot (Donnees Nord Pool)</p>', unsafe_allow_html=True)
    if not analysis_run:
        st.info("Lancez d'abord l'analyse dans l'onglet Simulation Delestage (Contrat HC/HP).")
    elif spot_df is None or spot_df.empty:
        st.error("Impossible de charger les donnees Nord Pool pour les graphiques.")
    else:
        st.markdown('<p class="section-title">Comparaison mensuelle : Spot seul vs Spot+Altileo (HT)</p>', unsafe_allow_html=True)
        if fig_monthly is not None:
            st.plotly_chart(fig_monthly, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

        st.markdown('<p class="section-title">Comparaison visuelle globale des scenarios (annuel HT)</p>', unsafe_allow_html=True)
        if fig_compare is not None:
            st.plotly_chart(fig_compare, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

        st.markdown('<p class="section-title">Evolution du prix Spot moyen journalier</p>', unsafe_allow_html=True)
        if fig_evo is not None:
            st.plotly_chart(fig_evo, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

# ----------------------------------------------------------
# ONGLET 7 : VALIDATION THERMIQUE
# ----------------------------------------------------------
with tab_thermal:
    st.markdown('<p class="section-title">Validation Thermique HACCP (Simulation Interactive)</p>', unsafe_allow_html=True)
    if not analysis_run:
        st.info("Lancez d'abord l'analyse dans le panneau lateral.")
    else:
        st.markdown("Simulez l'impact du delestage sur la temperature a coeur de la marchandise (ex: carton).")
        
        scenario = st.radio("Scenario de delestage a simuler", ["Optimisation Spot (Moyenne)", "Contrat classique (HC/HP)"], horizontal=True)
        
        if scenario == "Optimisation Spot (Moyenne)":
            if spot_df is None or spot_df.empty:
                st.warning("Veuillez lancer l'analyse avec un historique Spot valide pour voir ce scenario.")
                shed_hours = []
            else:
                shed_hours = top_hours_avg
                st.markdown(f"**Heures coupees (Spot) :** {', '.join([f'{hh}h' for hh in sorted(shed_hours)])}")
        else:
            c1, c2 = st.columns(2)
            with c1:
                start_h = st.number_input("Heure de debut de coupure (HC/HP)", min_value=0, max_value=23, value=18)
            with c2:
                st.markdown(f"<br>**Duree de coupure retenue :** {h} heures", unsafe_allow_html=True)
            shed_hours = [(int(start_h) + i) % 24 for i in range(int(h))]
            st.markdown(f"**Heures coupees (HC/HP) :** {', '.join([f'{hh}h' for hh in sorted(shed_hours)])}")

        # --- Simulation 24h ---
        temp_24h = [t_consigne]
        current_t = t_consigne
        
        for hh in range(24):
            if hh in shed_hours:
                current_t += pente_rechauffement
            else:
                current_t = max(t_consigne, current_t + pente_refroidissement)
            temp_24h.append(current_t)
            
        fig_24h = go.Figure()
        fig_24h.add_trace(go.Scatter(x=list(range(25)), y=temp_24h, mode='lines', name='Temperature Produit', line=dict(color='#C4652B', width=3)))
        fig_24h.add_hline(y=t_consigne, line_dash="dash", line_color="#2D8C5A", annotation_text=f"Consigne ({t_consigne}C)", annotation_position="top left")
        fig_24h.add_hline(y=t_max, line_dash="dash", line_color="#E74C3C", annotation_text=f"Limite HACCP ({t_max}C)", annotation_position="bottom left")
        
        for hh in shed_hours:
            fig_24h.add_vrect(x0=hh, x1=hh+1, fillcolor="rgba(231,76,60,0.1)", layer="below", line_width=0)
            
        fig_24h.update_layout(
            title="Cycle journalier (24 heures) - Zone rouge = compresseur coupe",
            xaxis_title="Heure de la journee", yaxis_title="Temperature (C)",
            margin=dict(l=10, r=10, t=40, b=10), height=350,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#8DA0B3'),
            yaxis=dict(showgrid=True, gridcolor='rgba(136,152,168,0.2)'), xaxis=dict(showgrid=False)
        )
        
        # --- Simulation 168h (7 jours) ---
        temp_168h = [t_consigne]
        current_t = t_consigne
        
        for day in range(7):
            for hh in range(24):
                if hh in shed_hours:
                    current_t += pente_rechauffement
                else:
                    current_t = max(t_consigne, current_t + pente_refroidissement)
                temp_168h.append(current_t)
                
        max_168h = max(temp_168h)
        is_conforme = max_168h <= t_max
        
        fig_168h = go.Figure()
        fig_168h.add_trace(go.Scatter(x=list(range(169)), y=temp_168h, mode='lines', name='Temperature Produit', line=dict(color='#1B3A5C', width=2)))
        fig_168h.add_hline(y=t_consigne, line_dash="dash", line_color="#2D8C5A", annotation_text=f"Consigne ({t_consigne}C)", annotation_position="top left")
        fig_168h.add_hline(y=t_max, line_dash="dash", line_color="#E74C3C", annotation_text=f"Limite HACCP ({t_max}C)", annotation_position="bottom left")
        
        for day in range(7):
            fig_168h.add_vline(x=day*24, line_dash="dot", line_color="rgba(136,152,168,0.5)")
            
        fig_168h.update_layout(
            title="Stress Test inertiel (7 jours / 168 heures)",
            xaxis_title="Heure cumulée", yaxis_title="Temperature (C)",
            margin=dict(l=10, r=10, t=40, b=10), height=350,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#8DA0B3'),
            yaxis=dict(showgrid=True, gridcolor='rgba(136,152,168,0.2)'), xaxis=dict(showgrid=False)
        )
        
        # Affichage
        c_kpi1, c_kpi2 = st.columns(2)
        with c_kpi1:
            if is_conforme:
                st.success(f"✅ **CONFORME** : La temperature reste stable et le compresseur a le temps de rattraper la consigne chaque jour.")
            else:
                st.error(f"⚠️ **RISQUE SANITAIRE** : Derivation thermique detectee. La chaleur s'accumule de jour en jour.")
        with c_kpi2:
            st.metric(label="Pic de temperature sur 7 jours", value=f"{max_168h:.2f} C", delta=f"{max_168h - t_consigne:+.2f} C vs consigne", delta_color="inverse")
            
        st.plotly_chart(fig_24h, use_container_width=True, config={'displaylogo': False})
        st.plotly_chart(fig_168h, use_container_width=True, config={'displaylogo': False})

