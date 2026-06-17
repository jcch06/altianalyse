import os
import math
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
.stApp { background-color: #f4f6f9; }
/* ========== Sidebar ========== */
section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #dfe3e8; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { font-size: 0.65rem !important; text-transform: uppercase; letter-spacing: 2px; color: #8898a8 !important; font-weight: 600 !important; margin-top: 0.6rem !important; margin-bottom: 0.4rem !important; padding-bottom: 0.3rem; border-bottom: 1px solid #eef1f5; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size: 0.85rem !important; font-weight: 600 !important; color: #1B3A5C !important; letter-spacing: 0.3px; margin-bottom: 0.5rem !important; }
/* ========== Header banner ========== */
.app-header { background: linear-gradient(135deg, #1B3A5C 0%, #264d73 60%, #1B3A5C 100%); padding: 1.6rem 2rem; border-radius: 6px; margin-bottom: 1.2rem; }
.app-header h1 { font-size: 1.5rem; font-weight: 700; color: #ffffff; letter-spacing: 3px; margin: 0; }
.app-header p { font-size: 0.8rem; color: #a3bdd4; margin: 0.3rem 0 0 0; font-weight: 400; letter-spacing: 0.4px; }
/* ========== KPI Metrics ========== */
div[data-testid="stMetric"] { background-color: #ffffff; padding: 1rem 1.2rem; border-radius: 6px; border: 1px solid #e2e6ec; border-left: 4px solid #1B3A5C; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }
div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.5px; color: #7a8a9a !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; color: #1a1a2e !important; }
/* ========== Tabs ========== */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #e2e6ec; background-color: transparent; }
.stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.78rem; letter-spacing: 0.2px; padding: 0.7rem 1.4rem; color: #7a8a9a; border-bottom: 2px solid transparent; margin-bottom: -2px; background-color: transparent !important; }
.stTabs [aria-selected="true"] { color: #1B3A5C !important; border-bottom-color: #1B3A5C !important; font-weight: 600; background-color: transparent !important; }
/* ========== Primary Button ========== */
button[data-testid="stBaseButton-primary"] { background-color: #1B3A5C !important; border: none !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.78rem !important; letter-spacing: 0.4px !important; border-radius: 5px !important; color: #ffffff !important; }
button[data-testid="stBaseButton-primary"]:hover { background-color: #264d73 !important; }
/* ========== Secondary Button ========== */
button[data-testid="stBaseButton-secondary"] { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.75rem !important; border-radius: 5px !important; border-color: #c5ced8 !important; color: #1B3A5C !important; }
/* ========== Dataframes ========== */
[data-testid="stDataFrame"] { border: 1px solid #e2e6ec; border-radius: 5px; overflow: hidden; }
/* ========== Alert / Info boxes ========== */
.stAlert { border-radius: 5px; font-size: 0.83rem; }
/* ========== Dividers ========== */
hr { border-color: #eef1f5 !important; }
/* ========== Expander ========== */
[data-testid="stExpander"] details summary p { font-weight: 500 !important; font-size: 0.82rem !important; color: #1a1a2e; }
/* ========== Plotly toolbar cleanup ========== */
.modebar-group { background-color: transparent !important; }
/* ========== Section titles in main area ========== */
.section-title { font-size: 0.95rem; font-weight: 600; color: #1a1a2e; margin: 1rem 0 0.8rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid #e2e6ec; }
/* ========== Stat card for monitoring ========== */
.stat-card { background-color: #ffffff; border: 1px solid #e2e6ec; border-radius: 6px; padding: 0.8rem 1rem; text-align: center; }
.stat-card .stat-label { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1px; color: #8898a8; font-weight: 600; }
.stat-card .stat-value { font-size: 1.1rem; font-weight: 700; color: #1a1a2e; margin-top: 0.2rem; }
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
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
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

            # KPIs Reference (HT)
            f_ref_an = (s_h.get('total_ht', 0) * nb * 5) + (s_e.get('total_ht', 0) * nb * 7)
            k_ref_an = (s_h.get('kwh_mois_total', 0) * nb * 5) + (s_e.get('kwh_mois_total', 0) * nb * 7)

            # Simulation financiere
            def get_sim_metrics(s):
                if not s:
                    return 0.0, 0.0
                k_hp_base = s.get('kwh_mois_hp', 0) * nb
                k_hc_base = s.get('kwh_mois_hc', 0) * nb

                k_effaces = k_hp_base * (h / 16.0)
                k_rattrapes = k_effaces * r * (1.0 - cop) * (1.0 + sec)

                k_hp_sim = k_hp_base - k_effaces
                k_hc_sim = k_hc_base + k_rattrapes

                f_hp_sim = k_hp_sim * s.get('tarif_hp', 0)
                f_hc_sim = k_hc_sim * s.get('tarif_hc', 0)
                turpe_sim = (k_hp_sim + k_hc_sim) * s.get('tarif_turpe', 0)
                taxes_sim = (k_hp_sim + k_hc_sim) * s.get('tarif_taxes', 0)

                total_ht_sim = (f_hp_sim + f_hc_sim + turpe_sim + taxes_sim) - (abo * nb)
                gain = (s.get('total_ht', 0) * nb) - total_ht_sim

                return gain, (k_effaces - k_rattrapes)

            gh, kh = get_sim_metrics(s_h)
            ge, ke = get_sim_metrics(s_e)

            gain_an_brut = (gh * 5) + (ge * 7)
            gain_an_net = gain_an_brut - (saas * nb)
            k_sauves_an = (kh * 5) + (ke * 7)
            pct = (gain_an_brut / f_ref_an * 100) if f_ref_an > 0 else 0

            i_net = max(0, pr * nb - cee)
            roi = (i_net / gain_an_net * 12) if gain_an_net > 0 else 0

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

                ax.set_title(f"Saison {title} (HT)", pad=20, fontweight="bold", fontsize=11, color='#1a1a2e')
                ax.set_ylabel("Euros / mois", fontsize=9, color='#7a8a9a')

                # Annotation gain
                g = c_av - c_ap
                if g > 0:
                    ax.annotate(
                        f"-{g:.0f} EUR\n(-{g / c_av * 100:.1f} %)",
                        xy=(1, c_ap), xytext=(0, c_av),
                        arrowprops=dict(arrowstyle="->", color="#1B3A5C", lw=1.8),
                        color="#1B3A5C", fontweight="bold", ha='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#1B3A5C", lw=1)
                    )

                # Legende
                legend_handles = [Patch(facecolor=c, label=l) for c, l in zip(CHART_COLORS, CHART_LABELS)]
                ax.legend(
                    handles=legend_handles, loc='upper right', fontsize=8,
                    framealpha=0.95, edgecolor='#e2e6ec'
                )

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#e2e6ec')
                ax.spines['bottom'].set_color('#e2e6ec')
                ax.tick_params(colors='#7a8a9a')
                fig.tight_layout()

                return fig

            fig_h = draw_chart(s_h, "HIVER")
            fig_e = draw_chart(s_e, "ETE")

# ============================================================
# 10. ONGLETS
# ============================================================

tab_dashboard, tab_details, tab_charts, tab_monitoring = st.tabs([
    "Indicateurs financiers",
    "Audit detaille",
    "Modelisation graphique",
    "Monitoring capteurs"
])

# ----------------------------------------------------------
# ONGLET 1 : INDICATEURS FINANCIERS
# ----------------------------------------------------------
with tab_dashboard:
    if not analysis_run:
        st.info("Configurez les parametres dans le panneau lateral puis lancez l'analyse.")
    else:
        st.markdown('<p class="section-title">Synthese annuelle du projet</p>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                label="Facture de reference",
                value=f"{f_ref_an:,.0f} EUR",
                delta=f"{k_ref_an:,.0f} kWh",
                delta_color="off"
            )
        with col2:
            st.metric(
                label="Gain Brut d'exploit.",
                value=f"{gain_an_brut:,.0f} EUR",
                delta="Avant abo SaaS",
                delta_color="off"
            )
        with col3:
            st.metric(
                label="Gain Net d'exploit.",
                value=f"{gain_an_net:,.0f} EUR",
                delta=f"-{pct:.1f} % (brut)"
            )
        with col4:
            st.metric(
                label="Impact environnemental",
                value=f"{(k_sauves_an * 0.05) / 1000:,.1f} t",
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
            
            return pdf.output(dest="S").encode("latin-1")
            
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
                template='plotly_white',
                height=480,
                margin=dict(l=50, r=30, t=30, b=50),
                font=dict(family='Inter, Arial, sans-serif', size=12, color='#1a1a2e'),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='left',
                    x=0,
                    font=dict(size=11)
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#f0f2f5',
                    gridwidth=1,
                    rangeslider=dict(visible=True, thickness=0.04),
                    title=''
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#f0f2f5',
                    gridwidth=1,
                    title='Valeur',
                    title_font=dict(size=11, color='#7a8a9a')
                ),
                hovermode='x unified',
                plot_bgcolor='#ffffff'
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
