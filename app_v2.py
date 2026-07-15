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
_FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon.png")

st.set_page_config(
    page_title="Altileo PRO",
    page_icon=_FAVICON_PATH if os.path.exists(_FAVICON_PATH) else "*",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Facteur d'émission moyen du reseau électrique français (source : ADEME Base
# Carbone / RTE, moyenne annuelle ~50 gCO2/kWh). A ajuster si une source plus
# recente ou spécifique au contrat du client est disponible.
CO2_FACTOR_KG_PER_KWH = 0.05
CO2_FACTOR_SOURCE = "ADEME Base Carbone / RTE, moyenne annuelle FR ~50 gCO2/kWh"


def fmt_fr(value, decimals: int = 0) -> str:
    """Formate un nombre en convention française (espace = separateur de milliers, virgule = decimale)."""
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", " ")


def fmt_fr_signed(value, decimals: int = 0) -> str:
    """Comme fmt_fr, avec un signe + explicite pour les valeurs positives ou nulles."""
    formatted = fmt_fr(value, decimals)
    return f"+{formatted}" if value >= 0 else formatted


# ============================================================
# 1b. HABILLAGE PDF (charte "Frost & Carbon", coherent avec le CSS de l'app)
# ============================================================
PDF_CARBON = (17, 17, 17)      # carbon -- bandeaux de preuve
PDF_TEAL = (0, 164, 180)       # teal glacier -- accent
PDF_MUTED_DARK = (153, 153, 153)  # muted-dark -- legendes sur fond sombre


def pdf_add_header(pdf: FPDF, title: str) -> None:
    """Bandeau de titre carbon avec logotype altileo* (asterisque teal), en tete de chaque rapport PDF."""
    pdf.set_fill_color(*PDF_CARBON)
    pdf.rect(0, 0, 210, 26, style="F")
    pdf.set_xy(10, 8)
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.write(10, "altileo")
    pdf.set_text_color(*PDF_TEAL)
    pdf.write(10, "*")
    pdf.set_text_color(255, 255, 255)
    pdf.write(10, f"  {title}")
    pdf.ln(18)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 11)


def pdf_add_section_title(pdf: FPDF, text: str) -> None:
    """Titre de section avec lisere teal, coherent avec le design de l'application."""
    pdf.ln(4)
    y = pdf.get_y()
    pdf.set_fill_color(*PDF_TEAL)
    pdf.rect(10, y + 1, 2.2, 6, style="F")
    pdf.set_xy(14, y)
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(*PDF_CARBON)
    pdf.cell(0, 8, text, ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 11)


def pdf_add_footer(pdf: FPDF) -> None:
    """Pied de page discret, coherent sur tous les rapports Altileo."""
    pdf.set_auto_page_break(False)
    pdf.set_y(-18)
    pdf.set_draw_color(*PDF_MUTED_DARK)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(*PDF_MUTED_DARK)
    pdf.cell(0, 5, f"Altileo -- Rapport généré automatiquement le {date.today().strftime('%d/%m/%Y')}", ln=1, align="C")

# ============================================================
# 2. DESIGN SYSTEM (CSS) -- Charte "Frost & Carbon"
# ============================================================
st.markdown("""<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
    --frost: #FFFFFF;
    --carbon: #111111;
    --surface: #F5F5F5;
    --ink: #111111;
    --muted: #555555;
    --muted-dark: #999999;
    --brand: #00A4B4;
    --brand-deep: #0E7C8C;
    --brand-dark: #0A6470;
    --frost-tint: #D4F0F0;
    --highlight: #FFD66B;
    --success: #2A9D6E;
    --warning: #E0A83D;
    --danger: #D14343;
    --line: #E5E5E5;
}
/* ========== Global ========== */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp { background-color: var(--frost); }
.num, div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"],
[data-testid="stDataFrame"] { font-family: 'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }
/* ========== Sidebar ========== */
section[data-testid="stSidebar"] { background-color: var(--surface); }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { font-family: 'Inter', sans-serif; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted) !important; font-weight: 500 !important; margin-top: 0.6rem !important; margin-bottom: 0.4rem !important; padding-bottom: 0.3rem; border-bottom: 0.5px solid var(--line); }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size: 0.85rem !important; font-weight: 700 !important; color: var(--ink) !important; letter-spacing: -0.01em; margin-bottom: 0.5rem !important; }
/* ========== Header banner (bandeau carbon) ========== */
.app-header { background: var(--carbon); padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1.2rem; }
.app-header .logo { font-size: 1.5rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; margin: 0; }
.app-header .logo .asterisk { color: var(--brand); }
.app-header .logo .pro-tag { font-family: 'Inter', sans-serif; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.08em; color: var(--brand); background: rgba(0,164,180,0.15); border: 0.5px solid rgba(0,164,180,0.5); border-radius: 6px; padding: 0.15rem 0.5rem; margin-left: 0.7rem; vertical-align: middle; text-transform: uppercase; }
.app-header p { font-size: 0.85rem; color: var(--muted-dark); margin: 0.35rem 0 0 0; font-weight: 400; letter-spacing: 0; }
/* ========== KPI Metrics (carte-metrique) ========== */
div[data-testid="stMetric"] { background-color: var(--frost); padding: 1rem 1.2rem; border-radius: 12px; border: 0.5px solid var(--line); }
div[data-testid="stMetricLabel"] { font-family: 'Inter', sans-serif; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted) !important; font-weight: 500 !important; }
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; color: var(--ink) !important; letter-spacing: -0.02em; }
/* ========== Tabs ========== */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 0.5px solid var(--line); background-color: transparent; }
.stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.78rem; letter-spacing: 0; padding: 0.7rem 1.4rem; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -1px; background-color: transparent !important; }
.stTabs [aria-selected="true"] { color: var(--brand-deep) !important; border-bottom-color: var(--brand) !important; font-weight: 700; }
/* ========== Buttons ========== */
button[data-testid="stBaseButton-primary"] { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.78rem !important; letter-spacing: 0 !important; border-radius: 6px !important; background-color: var(--brand) !important; border-color: var(--brand) !important; }
button[data-testid="stBaseButton-primary"]:hover { background-color: var(--brand-deep) !important; border-color: var(--brand-deep) !important; }
button[data-testid="stBaseButton-secondary"] { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.75rem !important; border-radius: 6px !important; border-color: var(--ink) !important; color: var(--ink) !important; }
/* ========== Dataframes ========== */
[data-testid="stDataFrame"] { border: 0.5px solid var(--line); border-radius: 12px; overflow: hidden; }
/* ========== Alert / Info boxes ========== */
.stAlert { border-radius: 6px; font-size: 0.83rem; font-family: 'Inter', sans-serif; }
/* ========== Dividers ========== */
hr { border-color: var(--line) !important; }
/* ========== Expander ========== */
[data-testid="stExpander"] details summary p { font-weight: 500 !important; font-size: 0.82rem !important; color: var(--ink); }
/* ========== Plotly toolbar cleanup ========== */
.modebar-group { background-color: transparent !important; }
/* ========== Section titles in main area ========== */
.section-title { font-size: 0.95rem; font-weight: 700; color: var(--ink); margin: 1rem 0 0.8rem 0; padding-bottom: 0.4rem; border-bottom: 0.5px solid var(--line); letter-spacing: -0.01em; }
/* ========== Stat card for monitoring ========== */
.stat-card { background-color: var(--frost); border: 0.5px solid var(--line); border-radius: 12px; padding: 0.8rem 1rem; text-align: center; }
.stat-card .stat-label { font-family: 'Inter', sans-serif; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 500; }
.stat-card .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--ink); margin-top: 0.2rem; }
/* ========== Accessibilite : focus visible ========== */
*:focus-visible { outline: 2px solid var(--brand-deep) !important; outline-offset: 2px; }
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# Matplotlib global style (charte "Frost & Carbon" : fond clair, texte muted/ink)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'text.color': '#555555',
    'axes.labelcolor': '#555555',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
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

try:
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
except Exception:
    APP_PASSWORD = os.getenv("APP_PASSWORD")

# ============================================================
# 3b. AUTHENTIFICATION
# ============================================================


def check_password() -> bool:
    """Gate simple par mot de passe partage. Pas de gate si APP_PASSWORD n'est pas configure."""
    if not APP_PASSWORD:
        return True
    if st.session_state.get("authenticated", False):
        return True

    st.markdown("""
    <div class="app-header">
        <p class="logo">altileo<span class="asterisk">*</span><span class="pro-tag">PRO</span></p>
        <p>Acces reserve</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        pwd = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", type="primary")
    if submitted:
        if pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    return False


if not check_password():
    st.stop()

# ============================================================
# 4. FONCTIONS : AUTO-DETECTION
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_available_tables() -> list:
    """Détecte les tables disponibles via le schema PostgREST."""
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
    """Détecte les capteurs uniques dans une table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Échantillonner début et fin de la table pour couvrir tous les capteurs
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
    """Détecte la première et la dernière date disponibles dans la table."""
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
# 5. FONCTIONS : DONNÉES ANALYSE (Logique Métier)
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_supabase_data(start_date: str, end_date: str, table_name: str, sensors: tuple):
    """Récupère les données brutes depuis Supabase avec pagination."""
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
    """Calcule puissance, énergie, coûts par catégorie et plage tarifaire."""
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

        # Gestion des gaps (clipping à 30 min max)
        df_sensor['delta_hours'] = df_sensor['timestamp'].diff().dt.total_seconds() / 3600.0
        df_sensor['delta_hours'] = df_sensor['delta_hours'].fillna(0).clip(upper=30.0 / 60.0)

        if capteur == sensor_comp:
            df_sensor['puissance_kw'] = (math.sqrt(3) * 400 * df_sensor['valeur'] * cos_phi) / 1000.0
            df_sensor['categorie_conso'] = 'compresseur'
        else:
            df_sensor['puissance_kw'] = (230 * df_sensor['valeur'] * 1.0) / 1000.0
            df_sensor['categorie_conso'] = 'hvac'
            df_sensor.loc[df_sensor['valeur'] > 6.0, 'categorie_conso'] = 'resistance'

        # Intégration trapezoidale
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
    """Génère les bilans journalier et hebdomadaire."""
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
    """Calcule la synthèse mensuelle pour une saison (HT)."""
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
    """Récupère les données brutes pour le monitoring (cache court)."""
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
    """Récupère les prix Day-Ahead France depuis l'API Nord Pool.
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


@st.cache_data(show_spinner=False)
def load_excel_prices_2025():
    """Charge l'historique annuel complet des prix Spot 2025 depuis le fichier Excel."""
    excel_path = "Prix Électricité France 2025 - Spot + Tempo.xlsx"
    local_path = os.path.join(os.path.dirname(__file__), excel_path)
    if os.path.exists(local_path):
        excel_path = local_path
    elif not os.path.exists(excel_path):
        excel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), excel_path)
    
    try:
        # Charger la feuille Spot 2025 Horaire
        df = pd.read_excel(excel_path, sheet_name='Spot 2025 Horaire')
        df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)
        df['hour'] = df['Heure'].astype(str).str.split(':').str[0].astype(int)
        
        # Trouver la colonne Spot (peut contenir des caractères accentués comme €)
        spot_col = [c for c in df.columns if 'Spot' in c][0]
        df['price_eur_mwh'] = df[spot_col].astype(float)
        
        # Tri et renommage des colonnes
        res = df.groupby(['Date', 'hour'])['price_eur_mwh'].mean().reset_index()
        res = res.rename(columns={'Date': 'date'})
        return res
    except Exception as e:
        st.error(f"Erreur lors du chargement des prix depuis Excel : {e}")
        return pd.DataFrame()


# ============================================================
# 6c. FONCTIONS : THERMIQUE (MODÈLE ASYMPTOTIQUE)
# ============================================================
# Les pentes mesurées à Rungis (+0.21 C/h de réchauffement, -0.70 C/h de
# refroidissement) sont des taux INITIAUX, mesurés au voisinage de la
# consigne. Physiquement, une chambre froide se rapproche exponentiellement
# de son point d'équilibre (loi de refroidissement de Newton) : la dérive
# ralentit à mesure qu'elle s'éloigne de la consigne (réchauffement) ou
# qu'elle s'en rapproche à nouveau (refroidissement). Le modèle linéaire
# simple restait correct sur la fenêtre testée (~4h) mais surestime la
# dérive sur des délestages plus longs (MCP, nuit complète).


def temp_rechauffement(t_start: float, dt_h: float, t_ambiante: float,
                        pente_rechauffement: float, t_consigne: float) -> float:
    """Rechauffement asymptotique vers la température ambiante d'équilibre.
    Conserve le taux initial mesure (pente_rechauffement) au voisinage de la consigne."""
    if t_ambiante <= t_consigne or pente_rechauffement <= 0:
        return t_start + pente_rechauffement * dt_h
    tau = (t_ambiante - t_consigne) / pente_rechauffement
    return t_ambiante - (t_ambiante - t_start) * math.exp(-dt_h / tau)


def temp_refroidissement(t_start: float, dt_h: float, t_consigne: float,
                          pente_refroidissement: float) -> float:
    """Refroidissement asymptotique vers la consigne : la recharge ralentit
    à mesure que l'écart avec la consigne se réduit (taux de référence calibré à 1 C d'écart)."""
    if t_start <= t_consigne or pente_refroidissement >= 0:
        return t_consigne
    tau = 1.0 / abs(pente_refroidissement)
    return t_consigne + (t_start - t_consigne) * math.exp(-dt_h / tau)


# ============================================================
# 7. INTERFACE : EN-TETE
# ============================================================

st.markdown("""
<div class="app-header">
    <p class="logo">altileo<span class="asterisk">*</span><span class="pro-tag">PRO</span></p>
    <p>Plateforme d'audit énergétique et de modélisation financière</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 8. INTERFACE : BARRE LATÉRALE
# ============================================================

with st.sidebar:
    st.markdown("## Altileo PRO")

    with st.expander("Glossaire des abréviations"):
        st.markdown("""
- **HP / HC** : Heures Pleines / Heures Creuses (tarif réglementé classique)
- **CVC** : Chauffage, Ventilation, Climatisation
- **MCP** : Matériau à Changement de Phase (stockage thermique, alternative aux batteries lithium)
- **COP** : Coefficient de Performance (rendement thermodynamique du compresseur)
- **TURPE** : Tarif d'Utilisation des Réseaux Publics d'Électricité (frais d'acheminement)
- **HACCP** : Hazard Analysis Critical Control Point (norme sécurité sanitaire des denrées)
- **Spot** : Prix de gros de l'électricité, fixé heure par heure la veille pour le lendemain (Day-Ahead)
- **Inverter / VFD** : Variateur de fréquence, permet de moduler en continu la vitesse du compresseur
- **ROI** : Retour sur investissement
        """)

    # --- Source de données ---
    st.markdown("### Source de données")
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
        s_cvc = st.selectbox("Capteur CVC / résistance", detected_sensors, index=cvc_idx)
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            s_comp = st.text_input("Capteur froid", "courant_1")
        with col_c2:
            s_cvc = st.text_input("Capteur CVC", "courant_2")

    # --- Période d'analyse ---
    st.markdown("### Période d'analyse")
    min_date, max_date = fetch_table_date_range(t_name)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        d_start = st.date_input("Début", min_date)
    with col_d2:
        d_end = st.date_input("Fin", max_date)

    # --- Tarifs ---
    with st.expander("Paramètres physiques et tarifs"):
        tarifs = {
            'cos_phi': st.number_input("Cos Phi (Froid)", value=0.92, format="%.2f"),
            'hp_hiv': st.number_input("HP Hiver (EUR/kWh)", value=0.12618, format="%.5f"),
            'hc_hiv': st.number_input("HC Hiver (EUR/kWh)", value=0.09779, format="%.5f"),
            'hp_ete': st.number_input("HP Été (EUR/kWh)", value=0.07869, format="%.5f"),
            'hc_ete': st.number_input("HC Été (EUR/kWh)", value=0.07424, format="%.5f"),
            'turpe': st.number_input("TURPE (EUR/kWh)", value=0.04000, format="%.5f"),
            'taxes': st.number_input("Taxes CSPE (EUR/kWh)", value=0.02100, format="%.5f"),
        }

    with st.expander("Paramètres thermiques (HACCP)"):
        t_consigne = st.number_input("Température consigne (°C)", value=-18.0, step=0.5)
        t_max = st.number_input("Limite HACCP (°C)", value=-15.0, step=0.5)
        pente_rechauffement = st.number_input(
            "Réchauffement initial (°C/h)", value=0.21, step=0.01,
            help="Taux mesuré au voisinage de la consigne (test Rungis). Le modèle ralentit ensuite cette dérive de façon asymptotique."
        )
        pente_refroidissement = st.number_input(
            "Refroidissement initial (°C/h)", value=-0.70, step=0.05,
            help="Taux de recharge mesuré à 1°C d'écart de la consigne. Le modèle ralentit la recharge à mesure que la consigne est approchée."
        )
        t_ambiante = st.number_input(
            "Température ambiante d'équilibre (°C)", value=12.0, step=1.0,
            help="Température vers laquelle la chambre dériverait si le compresseur restait coupé indéfiniment (charge thermique du bâtiment). Détermine la courbure du réchauffement."
        )

    st.divider()

    # --- Modélisation ---
    st.markdown("### Modélisation Altileo")
    h = st.slider("Délestage HP (heures)", 0.0, 16.0, 5.0, 0.5)
    r = st.slider("Rattrapage HC (%)", 0, 100, 10, 1) / 100.0
    sec = st.slider("Marge de sécurité (%)", 0, 50, 10, 1) / 100.0
    cop = st.slider("Bonus COP nuit (%)", 0, 50, 5, 1) / 100.0

    abo = st.number_input("Gain abonnement kVA (EUR/mois/ch)", value=0.0)
    saas = st.number_input("Abonnement SaaS (EUR/an/ch)", value=240.0)

    st.divider()

    # --- Perspective Inverter / VFD ---
    st.markdown("### Perspective Inverter / VFD")
    equipe_inverter = st.checkbox("Compresseur équipé Inverter / VFD")
    if equipe_inverter:
        gain_usure_an = st.number_input(
            "Gain usure mécanique estimé (EUR/an/chambre)", value=50.0, step=10.0,
            help="Estimation qualitative de l'usure évitée (moins de cycles marche/arrêt, pics de courant réduits "
                 "au démarrage — cf. section 6 de la note technique). Aucun modèle physique validé à ce jour : "
                 "valeur à ajuster manuellement selon le retour terrain."
        )
    else:
        gain_usure_an = 0.0

    st.divider()

    # --- Déploiement ---
    st.markdown("### Déploiement")
    nb = st.number_input("Nombre de chambres", min_value=1, value=1)
    pr = st.number_input("Prix installation / chambre (EUR)", value=1500.0)
    cee = st.number_input("Prime CEE globale (EUR)", value=0.0)

    st.divider()

    # --- Simulation Spot ---
    st.markdown("### Simulation Spot")
    spot_source = st.radio("Source des prix Spot", ["Historique annuel 2025 (Sobry)", "Derniers jours réels (API Nord Pool)"])
    spot_delest_h = st.slider("Heures délestées / jour", 0, 12, 5)
    spot_margin = st.number_input("Marge fournisseur fixe (EUR/MWh)", value=0.0, step=1.0)
    spot_margin_pct = st.number_input("Marge proportionnelle fournisseur (ex: Sobry %)", value=8.0, step=0.5)
    
    if spot_source == "Derniers jours réels (API Nord Pool)":
        spot_days = st.slider("Jours d'historique Nord Pool", 30, 180, 90, step=10)
    else:
        spot_days = 365

    st.divider()

    # --- Bouton d'analyse ---
    if st.button("Lancer l'analyse", type="primary", use_container_width=True):
        st.session_state['run_analysis'] = True

# ============================================================
# 9. MOTEUR DE CALCUL
# ============================================================

analysis_run = st.session_state.get('run_analysis', False)

# Variables d'analyse (initialisées à None)
df_proc = None
f_ref_an = k_ref_an = gain_an_net = gain_an_brut = 0
k_sauves_an = pct = roi = i_net = 0
s_h = s_e = None

# Variables Spot (initialisées à None ou valeurs par défaut)
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
            st.warning("Aucune donnée trouvée pour cette période et cette table.")
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

            # --- Génération des profils mensuels ---
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
            noms_mois = {1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"}
            
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
                
                # Séparation HP/HC de base (HP de 6h à 22h)
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
                
                # Calcul précis de l'effacement basé sur le profil horaire réel (limité à la plage hors travail 12h-22h)
                hp_sorted_consos = sorted([prof[hh] for hh in range(12, 22)], reverse=True)
                h_int = int(h)
                h_frac = h - h_int
                kwh_efface_jour = sum(hp_sorted_consos[:h_int])
                if h_int < 16 and h_frac > 0:
                    kwh_efface_jour += hp_sorted_consos[h_int] * h_frac
                
                k_effaces = kwh_efface_jour * jours * nb
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
                usure_mois = (gain_usure_an * nb) * (jours / 365.0)

                hchp_monthly_results.append({
                    'mois_num': m,
                    'mois_nom': noms_mois[m],
                    'saison': saison,
                    'jours': jours,
                    'cout_base': total_ht_base,
                    'cout_sim': total_ht_sim + saas_mois - usure_mois,
                    'gain_net': gain_mois - saas_mois + usure_mois
                })

            jours_totaux = sum(jours_par_mois[m] for m in mois_dispos)
            saas_tot = (saas * nb) * (jours_totaux / 365.0)
            usure_tot = (gain_usure_an * nb) * (jours_totaux / 365.0)
            gain_tot_net = gain_tot_brut - saas_tot + usure_tot
            pct = (gain_tot_brut / f_ref_tot * 100) if f_ref_tot > 0 else 0
            
            i_net = max(0, pr * nb - cee)
            gain_moyen_mensuel = (gain_tot_net / jours_totaux) * (365.0 / 12.0) if jours_totaux > 0 else 0
            roi = (i_net / gain_moyen_mensuel) if gain_moyen_mensuel > 0 else 0

            # Extrapolation annuelle pour le PDF et les graphiques
            f_ref_an = f_ref_tot * (365.0 / jours_totaux) if jours_totaux > 0 else 0.0
            gain_an_brut = gain_tot_brut * (365.0 / jours_totaux) if jours_totaux > 0 else 0.0
            gain_an_net = gain_tot_net * (365.0 / jours_totaux) if jours_totaux > 0 else 0.0
            k_sauves_an = k_sauves_tot * (365.0 / jours_totaux) if jours_totaux > 0 else 0.0

            # Calcul thermique théorique (rechauffement asymptotique sur h heures continues)
            temp_max_hchp = temp_rechauffement(t_consigne, h, t_ambiante, pente_rechauffement, t_consigne)
            derive_max_hchp = temp_max_hchp - t_consigne

            # --- Génération des graphiques (en mémoire) ---
            CHART_COLORS = ['#111111', '#00A4B4', '#555555', '#0E7C8C']
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

                ax.set_title(f"Saison {title} (HT)", pad=20, fontweight="bold", fontsize=11, color='#555555')
                ax.set_ylabel("Euros / mois", fontsize=9, color='#555555')

                # Ajuster l'axe Y pour éviter que l'annotation soit coupée
                ax.set_ylim(0, max(c_av, c_ap) * 1.25)

                # Annotation gain
                g = c_av - c_ap
                if g > 0:
                    ax.annotate(
                        f"-{g:.0f} EUR\n(-{g / c_av * 100:.1f} %)",
                        xy=(1, c_ap), xytext=(0, c_av * 1.02),
                        arrowprops=dict(arrowstyle="->", color="#00A4B4", lw=1.8),
                        color="#FFFFFF", fontweight="bold", ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#111111", ec="#00A4B4", lw=1)
                    )

                # Légende
                legend_handles = [Patch(facecolor=c, label=l) for c, l in zip(CHART_COLORS, CHART_LABELS)]
                ax.legend(
                    handles=legend_handles, loc='upper right', fontsize=8,
                    framealpha=0.0, edgecolor='none', labelcolor='#555555'
                )

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#E5E5E5')
                ax.spines['bottom'].set_color('#E5E5E5')
                ax.tick_params(colors='#555555')
                fig.tight_layout()

                return fig

            fig_h = draw_chart(s_h, "HIVER")
            fig_e = draw_chart(s_e, "ÉTÉ")

# ============================================================
# 10. ONGLETS
# ============================================================

tab_dashboard, tab_details, tab_charts, tab_spot, tab_spot_charts, tab_thermal, tab_monitoring = st.tabs([
    "Simulation Délestage (Contrat HC/HP)",
    "Audit détaillé (HC/HP)",
    "Graphiques (HC/HP)",
    "Simulation Délestage (Prix SPOT)",
    "Graphiques (Prix SPOT)",
    "Validation Thermique",
    "Monitoring capteurs"
])

# ----------------------------------------------------------
# ONGLET 1 : INDICATEURS FINANCIERS
# ----------------------------------------------------------
with tab_dashboard:
    if not analysis_run:
        st.info("Configurez les paramètres dans le panneau latéral puis lancez l'analyse.")
    else:
        nb_mois = len(mois_dispos)
        st.markdown(f'<p class="section-title">Détail Mensuel ({nb_mois} mois étudiés)</p>', unsafe_allow_html=True)
        
        for res in hchp_monthly_results:
            saison_label = {'ete': 'Été', 'hiver': 'Hiver'}.get(res['saison'], res['saison'].capitalize())
            st.markdown(f"**Bilan {res['mois_nom']} ({res['jours']} jours - Tarif {saison_label})**")
            c1, c2, c3, c4 = st.columns(4)
            c_base = res['cout_base']
            c_sim = res['cout_sim']
            g_net = res['gain_net']
            
            with c1:
                st.metric("HC/HP Classique", f"{fmt_fr(c_base, 0)} EUR")
            with c2:
                # empty column to keep alignment with Spot
                st.write("")
            with c3:
                d = ((c_sim - c_base)/c_base*100) if c_base > 0 else 0
                saving = c_sim - c_base
                st.metric("HC/HP AVEC Altileo", f"{fmt_fr(c_sim, 0)} EUR", f"{d:+.1f}% ({fmt_fr_signed(saving, 0)} EUR) vs Classique", delta_color="inverse")
            with c4:
                st.metric("Gain Net Mensuel", f"{fmt_fr(g_net, 0)} EUR")
            st.write("")

        st.divider()
        st.markdown(f'<p class="section-title">Synthèse globale sur la période étudiée ({jours_totaux} jours)</p>', unsafe_allow_html=True)

        if jours_totaux < 30:
            st.warning(
                f"**Période d'analyse courte ({jours_totaux} jours)** : l'extrapolation annuelle ci-dessous "
                "repose sur peu de données et peut être peu fiable (saisonnalité, jours atypiques). "
                "À considérer comme une estimation indicative, à confirmer sur une période plus longue."
            )

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                label="Facture de référence",
                value=f"{fmt_fr(f_ref_tot, 0)} EUR",
                delta=f"{fmt_fr(k_ref_tot, 0)} kWh",
                delta_color="off"
            )
        with col2:
            st.metric(
                label="Gain Brut d'exploit.",
                value=f"{fmt_fr(gain_tot_brut, 0)} EUR",
                delta="Avant abo SaaS",
                delta_color="off"
            )
        with col3:
            st.metric(
                label="Gain Net d'exploit.",
                value=f"{fmt_fr(gain_tot_net, 0)} EUR",
                delta=f"+{fmt_fr(gain_tot_net, 0)} EUR (-{pct:.1f}% brut)"
            )
        with col4:
            st.metric(
                label="Impact environnemental",
                value=f"{fmt_fr((k_sauves_tot * CO2_FACTOR_KG_PER_KWH) / 1000, 1)} t",
                delta="CO2 évité",
                help=f"Calculé avec un facteur de {CO2_FACTOR_KG_PER_KWH * 1000:.0f} gCO2/kWh ({CO2_FACTOR_SOURCE})"
            )
        with col5:
            st.metric(
                label="Retour investissement",
                value=f"{roi:.1f} mois",
                delta=f"soit {roi/12:.1f} ans",
                delta_color="off"
            )

        st.divider()
        st.markdown('<p class="section-title">Paramètres de modélisation retenus</p>', unsafe_allow_html=True)

        param_col1, param_col2 = st.columns(2)
        with param_col1:
            st.markdown(f"**Chambres équipées :** {nb}")
            st.markdown(f"**Délestage journalier :** {h} heures")
        with param_col2:
            st.markdown(f"**Rattrapage thermique :** {r * 100:.0f} % (marge sécurité {sec * 100:.0f} %)")
            st.markdown(f"**Abonnement SaaS :** {saas * nb:.0f} EUR / an")
            if equipe_inverter:
                st.markdown(f"**Gain usure mécanique (Inverter, estimation) :** {gain_usure_an * nb:.0f} EUR / an")

        st.divider()
        st.markdown('<p class="section-title">Validation Thermique HACCP</p>', unsafe_allow_html=True)
        if temp_max_hchp > t_max:
            st.error(f"**RISQUE SANITAIRE** : La température maximale simulée atteint **{temp_max_hchp:.2f} C**, dépassant la limite HACCP de **{t_max:.2f} C**.")
        else:
            st.success(f"**CONFORME** : La température maximale simulée est de **{temp_max_hchp:.2f} C** (limite : {t_max:.2f} C).")

        st.divider()
        
        # --- Export PDF ---
        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf_add_header(pdf, "Rapport d'Audit Énergétique")

            pdf.cell(0, 6, f"Date de l'audit : {date.today().strftime('%d/%m/%Y')}", ln=1)
            try:
                pdf.cell(0, 6, f"Période analysée : du {d_start.strftime('%d/%m/%Y')} au {d_end.strftime('%d/%m/%Y')}", ln=1)
            except:
                pdf.cell(0, 6, f"Période analysée : du {d_start} au {d_end}", ln=1)
            pdf.cell(0, 6, f"Nombre de chambres équipées : {nb}", ln=1)
            pdf.cell(0, 6, f"Délestage programmé : {h} heures/jour", ln=1)

            pdf_add_section_title(pdf, "Résultats Financiers (HT)")
            pdf.cell(0, 6, f"Facture de référence : {fmt_fr(f_ref_an, 0)} EUR/an", ln=1)
            pdf.cell(0, 6, f"Gain d'exploitation brut : {fmt_fr(gain_an_brut, 0)} EUR/an", ln=1)
            pdf.cell(0, 6, f"Abonnement SaaS : {fmt_fr(saas * nb, 0)} EUR/an", ln=1)
            if equipe_inverter:
                pdf.cell(0, 6, f"Gain usure mécanique (Inverter, estimation) : {fmt_fr(gain_usure_an * nb, 0)} EUR/an", ln=1)
            pdf.cell(0, 6, f"Gain d'exploitation NET : {fmt_fr(gain_an_net, 0)} EUR/an", ln=1)
            pdf.cell(0, 6, f"Retour sur investissement (ROI) : {roi:.1f} mois (soit {roi/12:.1f} ans)", ln=1)
            pdf.cell(0, 6, f"Impact carbone : {fmt_fr((k_sauves_an * CO2_FACTOR_KG_PER_KWH) / 1000, 1)} tonnes de CO2 évitées/an", ln=1)
            pdf.set_font("Arial", "I", 8)
            pdf.cell(0, 5, f"(facteur retenu : {CO2_FACTOR_KG_PER_KWH * 1000:.0f} gCO2/kWh -- {CO2_FACTOR_SOURCE})", ln=1)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"Validation HACCP : {'ÉCHEC' if temp_max_hchp > t_max else 'CONFORME'} (Temp max : {temp_max_hchp:.2f} C)", ln=1)

            pdf_add_section_title(pdf, "Modélisation Mensuelle")

            img_h = io.BytesIO()
            fig_h.savefig(img_h, format='png', bbox_inches='tight')
            pdf.image(img_h, x=10, y=pdf.get_y(), w=90)

            img_e = io.BytesIO()
            fig_e.savefig(img_e, format='png', bbox_inches='tight')
            pdf.image(img_e, x=110, y=pdf.get_y(), w=90)

            pdf_add_footer(pdf)
            return bytes(pdf.output())
            
        pdf_bytes = create_pdf()
        st.download_button(
            label="Télécharger le rapport PDF",
            data=pdf_bytes,
            file_name=f"Altileo_Audit_{date.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary"
        )

# ----------------------------------------------------------
# ONGLET 2 : AUDIT DÉTAILLÉ
# ----------------------------------------------------------
with tab_details:
    if not analysis_run or df_proc is None:
        st.info("Lancez l'analyse pour consulter le détail des consommations.")
    else:
        df_daily, df_weekly = generate_detailed_dataframes(df_proc)

        st.markdown('<p class="section-title">Bilan hebdomadaire</p>', unsafe_allow_html=True)
        st.dataframe(df_weekly, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-title">Bilan journalier</p>', unsafe_allow_html=True)
        st.dataframe(df_daily, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# ONGLET 3 : MODÉLISATION GRAPHIQUE
# ----------------------------------------------------------
with tab_charts:
    if not analysis_run or s_h is None:
        st.info("Lancez l'analyse pour afficher les graphiques de modélisation.")
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

    # --- Contrôles ---
    mon_col1, mon_col2, mon_col3 = st.columns([3, 2, 1])

    with mon_col1:
        all_sensors = fetch_available_sensors(t_name)
        default_sensors = all_sensors[:2] if len(all_sensors) >= 2 else all_sensors
        mon_sensors = st.multiselect(
            "Capteurs à afficher",
            all_sensors,
            default=default_sensors,
            key="mon_sensors"
        )

    with mon_col2:
        period_options = [
            "Dernière heure",
            "6 dernières heures",
            "24 dernières heures",
            "48 dernières heures",
            "7 derniers jours",
            "30 derniers jours",
            "Personnalisée"
        ]
        period_choice = st.selectbox("Période", period_options, index=2, key="mon_period")

    with mon_col3:
        st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button("Actualiser", key="mon_refresh", type="secondary")

    # Période personnalisée
    if period_choice == "Personnalisée":
        mon_date_col1, mon_date_col2 = st.columns(2)
        with mon_date_col1:
            mon_start_date = st.date_input("Du", date.today() - timedelta(days=7), key="mon_start")
        with mon_date_col2:
            mon_end_date = st.date_input("Au", date.today(), key="mon_end")
        mon_start_ts = f"{mon_start_date}T00:00:00"
        mon_end_ts = f"{mon_end_date}T23:59:59"
    else:
        period_deltas = {
            "Dernière heure": timedelta(hours=1),
            "6 dernières heures": timedelta(hours=6),
            "24 dernières heures": timedelta(hours=24),
            "48 dernières heures": timedelta(hours=48),
            "7 derniers jours": timedelta(days=7),
            "30 derniers jours": timedelta(days=30),
        }
        delta = period_deltas[period_choice]
        # Arrondir à la minute pour stabiliser le cache
        now_rounded = datetime.now().replace(second=0, microsecond=0)
        mon_start_ts = (now_rounded - delta).strftime('%Y-%m-%dT%H:%M:%S')
        mon_end_ts = now_rounded.strftime('%Y-%m-%dT%H:%M:%S')

    # Forcer le rafraîchissement
    if refresh_clicked:
        fetch_monitoring_data.clear()
        st.rerun()

    # --- Affichage des données ---
    if not mon_sensors:
        st.info("Sélectionnez au moins un capteur pour afficher les données.")
    else:
        df_mon = fetch_monitoring_data(t_name, tuple(mon_sensors), mon_start_ts, mon_end_ts)

        if df_mon.empty:
            st.info("Aucune donnée disponible pour la période et les capteurs sélectionnés.")
        else:
            df_mon['timestamp'] = pd.to_datetime(df_mon['timestamp'])
            df_mon['valeur'] = pd.to_numeric(df_mon['valeur'], errors='coerce')
            df_mon = df_mon.sort_values('timestamp')

            # --- Graphique Plotly ---
            # Palette technique dérivée des deux teintes de marque (teal, carbon),
            # par variations de nuance -- suffisamment de contraste pour plusieurs capteurs
            # sans sortir de la famille de couleurs Frost & Carbon.
            PLOTLY_COLORS = [
                '#00A4B4', '#111111', '#0E7C8C', '#555555',
                '#0A6470', '#999999', '#5FC4D0', '#333333'
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
                font=dict(family="Inter, sans-serif", size=10, color='#555555'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(229,229,229,0.7)',
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(153,153,153,0.5)',
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
                    unit = " A" if "courant" in sensor.lower() else " °C" if "temp" in sensor.lower() else ""
                    st.metric("Dernière valeur", f"{last_val:.2f}{unit}")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Min", f"{df_s['valeur'].min():.2f}{unit}")
                    sc2.metric("Moy", f"{df_s['valeur'].mean():.2f}{unit}")
                    sc3.metric("Max", f"{df_s['valeur'].max():.2f}{unit}")

            # Horodatage
            st.caption(f"Dernière actualisation : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# ----------------------------------------------------------
# ONGLET 5 : SIMULATION SPOT (DONNÉES NORD POOL)
# ----------------------------------------------------------
with tab_spot:
    st.markdown('<p class="section-title">Simulation contrat Spot -- Données Nord Pool France</p>', unsafe_allow_html=True)

    if not analysis_run:
        st.info("Lancez d'abord l'analyse dans l'onglet Simulation Délestage (Contrat HC/HP).")
    else:
        # --- Chargement des prix Nord Pool ---
        if spot_source == "Historique annuel 2025 (Sobry)":
            with st.spinner("Chargement des prix Spot 2025 depuis le fichier Excel..."):
                spot_df = load_excel_prices_2025()
            if spot_df.empty:
                st.error("Impossible de charger les données du fichier Excel 2025. Assurez-vous que le fichier est présent dans le répertoire racine.")
            else:
                dates_dt = pd.to_datetime(spot_df['date'])
                spot_start_date = dates_dt.min().date()
                spot_end_date = dates_dt.max().date()
        else:
            spot_end_date = date.today() - timedelta(days=1)
            spot_start_date = spot_end_date - timedelta(days=spot_days)
            with st.spinner(f"Chargement des prix Spot Nord Pool ({spot_days} jours)..."):
                spot_df = fetch_nordpool_prices(spot_start_date.isoformat(), spot_end_date.isoformat())
            if spot_df.empty:
                st.error("Impossible de charger les données Nord Pool. Vérifiez votre connexion.")

        if not spot_df.empty:
            n_days_loaded = spot_df['date'].nunique()
            spot_avg_overall = spot_df['price_eur_mwh'].mean()

            if spot_source == "Historique annuel 2025 (Sobry)":
                st.success(f"Année 2025 complète chargée ({n_days_loaded} jours) depuis le fichier Sobry -- Prix moyen annuel : **{spot_avg_overall:.1f} EUR/MWh**")
            else:
                st.success(f"{n_days_loaded} jours chargés ({spot_start_date.strftime('%d/%m/%Y')} au {spot_end_date.strftime('%d/%m/%Y')}) -- Prix moyen : **{spot_avg_overall:.1f} EUR/MWh**")

            # --- Profil horaire moyen (depuis les vraies données) ---
            avg_profile = spot_df.groupby('hour')['price_eur_mwh'].mean().reindex(range(24), fill_value=0)

            # Identifier les heures les plus chères en moyenne pour le graphique (hors 2h-12h)
            valid_hours = [h for h in range(24) if not (2 <= h < 12)]
            top_hours_avg = avg_profile.loc[valid_hours].nlargest(spot_delest_h).index.tolist()

            bar_colors = ['#00A4B4' if hh in top_hours_avg else '#111111' for hh in range(24)]

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
                font=dict(family="Inter", size=10, color='#555555'),
                yaxis=dict(title="EUR / MWh", showgrid=True, gridcolor='#E5E5E5'),
                xaxis=dict(showgrid=False, showline=True, linecolor='#E5E5E5'),
                title=dict(
                    text=f"Profil horaire moyen Nord Pool FR -- {spot_delest_h}h délestées/jour (en teal)",
                    font=dict(size=13, color='#111111'), x=0.5
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

                # Recherche gloutonne des N heures les plus rentables à délester
                # en prenant en compte le profil de consommation et le coût du rattrapage
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
                                p_kwh = ((prices_day[hh] + spot_margin) / 1000.0) * (1.0 + spot_margin_pct / 100.0) + tarifs['turpe'] + tarifs['taxes']
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
                    price_kwh = ((prices_day[hh] + spot_margin) / 1000.0) * (1.0 + spot_margin_pct / 100.0) + tarifs['turpe'] + tarifs['taxes']
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
                    price_kwh = ((prices_day[hh] + spot_margin) / 1000.0) * (1.0 + spot_margin_pct / 100.0) + tarifs['turpe'] + tarifs['taxes']
                    cost_altileo_day += ratt_per_h * price_kwh

                # Simulation thermique jour par jour
                current_temp = t_consigne
                max_temp_day = t_consigne
                for hh in range(24):
                    if hh in delest_hours_day:
                        current_temp = temp_rechauffement(current_temp, 1.0, t_ambiante, pente_rechauffement, t_consigne)
                    else:
                        current_temp = temp_refroidissement(current_temp, 1.0, t_consigne, pente_refroidissement)
                    if current_temp > max_temp_day:
                        max_temp_day = current_temp

                # Référence HC/HP sans délestage pour la même journée
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

                # HC/HP AVEC délestage pour la même journée
                hp_consos = [current_all_hourly[hh] for hh in range(6, 22)]
                hp_sorted = sorted(hp_consos, reverse=True)
                h_int = int(h)
                h_frac = h - h_int
                kwh_efface = sum(hp_sorted[:h_int])
                if h_int < 16 and h_frac > 0:
                    kwh_efface += hp_sorted[h_int] * h_frac
                
                k_effaces_day = kwh_efface
                k_rattrapes_day = k_effaces_day * r * (1.0 - cop) * (1.0 + sec)
                
                k_hp_base = sum(current_all_hourly[hh] for hh in range(6, 22))
                k_hc_base = sum(current_all_hourly[hh] for hh in [hh for hh in range(24) if hh < 6 or hh >= 22])
                
                k_hp_sim = k_hp_base - k_effaces_day
                k_hc_sim = k_hc_base + k_rattrapes_day
                
                if saison_val == 'hiver':
                    t_hp = tarifs['hp_hiv']
                    t_hc = tarifs['hc_hiv']
                else:
                    t_hp = tarifs['hp_ete']
                    t_hc = tarifs['hc_ete']
                    
                f_hp_sim = k_hp_sim * t_hp
                f_hc_sim = k_hc_sim * t_hc
                turpe_sim = (k_hp_sim + k_hc_sim) * tarifs['turpe']
                taxes_sim = (k_hp_sim + k_hc_sim) * tarifs['taxes']
                cost_hchp_sim_day = f_hp_sim + f_hc_sim + turpe_sim + taxes_sim

                daily_results.append({
                    'date': date_val,
                    'cost_hchp': cost_hchp_day,
                    'cost_hchp_sim': cost_hchp_sim_day,
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
                cout_hchp_sim=('cost_hchp_sim', 'sum'),
                cout_spot=('cost_spot', 'sum'),
                cout_altileo=('cost_altileo', 'sum'),
                kwh_eco=('kwh_saved', 'sum'),
                gain=('gain_jour', 'sum'),
                temp_max=('max_temp', 'max')
            ).reset_index()

            # Application du nb de chambres
            for col in ['cout_hchp', 'cout_spot', 'cout_altileo', 'kwh_eco', 'gain']:
                monthly[col] = monthly[col] * nb
                
            # Totaux de la période
            total_days = res_df['date'].nunique()
            cost_hchp_tot = res_df['cost_hchp'].sum() * nb
            cost_spot_tot = res_df['cost_spot'].sum() * nb
            cost_altileo_tot = res_df['cost_altileo'].sum() * nb
            kwh_saved_tot = res_df['kwh_saved'].sum() * nb
            gain_spot_brut = cost_spot_tot - cost_altileo_tot
            
            # Prorata SaaS et gain usure mécanique sur la période analysée
            saas_periode = (saas * nb) * (total_days / 365.0)
            usure_periode = (gain_usure_an * nb) * (total_days / 365.0)
            gain_spot_net = gain_spot_brut - saas_periode + usure_periode
            gain_vs_actuel = cost_hchp_tot - (cost_altileo_tot + saas_periode - usure_periode)
            max_temp_spot = res_df['max_temp'].max()

            # Extrapolation annuelle pour le graphique de comparaison
            f_ref_an = cost_hchp_tot * (365.0 / total_days) if total_days > 0 else 0.0
            cost_spot_annual = cost_spot_tot * (365.0 / total_days) if total_days > 0 else 0.0
            cost_altileo_annual = cost_altileo_tot * (365.0 / total_days) if total_days > 0 else 0.0

            st.divider()
            st.markdown(f'<p class="section-title">Analyse Mensuelle (Données réelles sur {total_days} jours)</p>', unsafe_allow_html=True)
            
            # Affichage d'une carte par mois
            for _, row in monthly.iterrows():
                mois_nom = row['mois_str']
                j = row['jours']
                c_hc = row['cout_hchp']
                c_spot = row['cout_spot']
                c_alt = row['cout_altileo']
                saas_mois = (saas * nb) * (j / 365.0)
                usure_mois = (gain_usure_an * nb) * (j / 365.0)
                c_alt_net_mois = c_alt + saas_mois - usure_mois
                g_net = row['gain'] - saas_mois + usure_mois

                st.markdown(f"**Bilan {mois_nom} ({j} jours)**")
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric(label="HC/HP", value=f"{fmt_fr(c_hc, 0)} EUR")
                with k2:
                    d1 = ((c_spot - c_hc)/c_hc*100) if c_hc>0 else 0
                    diff1 = c_spot - c_hc
                    st.metric(label="Spot SANS Altileo", value=f"{fmt_fr(c_spot, 0)} EUR", delta=f"{d1:+.1f}% ({fmt_fr_signed(diff1, 0)} EUR) vs HC/HP", delta_color="inverse")
                with k3:
                    d2 = ((c_alt_net_mois - c_hc)/c_hc*100) if c_hc>0 else 0
                    diff2 = c_alt_net_mois - c_hc
                    st.metric(label="Spot AVEC Altileo", value=f"{fmt_fr(c_alt_net_mois, 0)} EUR", delta=f"{d2:+.1f}% ({fmt_fr_signed(diff2, 0)} EUR) vs HC/HP", delta_color="inverse")
                with k4:
                    st.metric(label="Gain Net Mensuel", value=f"{fmt_fr(g_net, 0)} EUR")
                st.write("") # spacing

            st.divider()
            st.markdown(f'<p class="section-title">Résumé de la période ({spot_start_date.strftime("%d/%m/%Y")} au {spot_end_date.strftime("%d/%m/%Y")})</p>', unsafe_allow_html=True)

            if total_days < 30:
                st.warning(
                    f"**Période d'analyse courte ({total_days} jours)** : l'extrapolation annuelle peut être peu "
                    "fiable (saisonnalité, jours atypiques). À considérer comme une estimation indicative."
                )

            c_hc_tot = cost_hchp_tot
            c_spot_tot = cost_spot_tot
            c_alt_tot = cost_altileo_tot + saas_periode - usure_periode
            
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                st.metric("HC/HP Classique", f"{fmt_fr(c_hc_tot, 0)} EUR", f"Sur {total_days} jours", delta_color="off")
            with kpi2:
                d1_tot = ((c_spot_tot - c_hc_tot)/c_hc_tot*100) if c_hc_tot > 0 else 0
                diff1_tot = c_spot_tot - c_hc_tot
                st.metric("Spot SANS Altileo", f"{fmt_fr(c_spot_tot, 0)} EUR", f"{d1_tot:+.1f}% ({fmt_fr_signed(diff1_tot, 0)} EUR) vs HC/HP", delta_color="inverse")
            with kpi3:
                d2_tot = ((c_alt_tot - c_hc_tot)/c_hc_tot*100) if c_hc_tot > 0 else 0
                diff2_tot = c_alt_tot - c_hc_tot
                st.metric("Spot AVEC Altileo", f"{fmt_fr(c_alt_tot, 0)} EUR", f"{d2_tot:+.1f}% ({fmt_fr_signed(diff2_tot, 0)} EUR) vs HC/HP", delta_color="inverse")
            with kpi4:
                st.metric("Gain vs Actuel", f"{fmt_fr(gain_vs_actuel, 0)} EUR", "Spot+Altileo vs HC/HP", delta_color="normal")
            with kpi5:
                st.metric(
                    "kWh économisés", f"{fmt_fr(kwh_saved_tot, 0)} kWh",
                    f"{fmt_fr((kwh_saved_tot * CO2_FACTOR_KG_PER_KWH) / 1000, 1)} t CO2 évité", delta_color="normal",
                    help=f"Calculé avec un facteur de {CO2_FACTOR_KG_PER_KWH * 1000:.0f} gCO2/kWh ({CO2_FACTOR_SOURCE})"
                )

            st.divider()
            if max_temp_spot > t_max:
                st.error(f"**RISQUE SANITAIRE** : Sur la période, la température maximale simulée a atteint **{max_temp_spot:.2f} C**, dépassant la limite HACCP de **{t_max:.2f} C**.")
            else:
                st.success(f"**CONFORME** : La température maximale simulée sur la période est restée à **{max_temp_spot:.2f} C** (limite : {t_max:.2f} C).")

            # --- Export PDF Spot ---
            def create_spot_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf_add_header(pdf, "Rapport d'Audit (Simulation Spot)")

                pdf.cell(0, 6, f"Date de l'audit : {date.today().strftime('%d/%m/%Y')}", ln=1)
                pdf.cell(0, 6, f"Nombre de chambres équipées : {nb}", ln=1)
                pdf.cell(0, 6, f"Délestage programmé (Spot) : {spot_delest_h} heures/jour", ln=1)
                pdf.cell(0, 6, f"Marge fixe Spot : {spot_margin} EUR/MWh", ln=1)
                pdf.cell(0, 6, f"Marge proportionnelle Spot (Sobry) : {spot_margin_pct} %", ln=1)
                pdf.cell(0, 6, f"Période étudiée : {total_days} jours (Moyenne : {spot_avg_overall:.1f} EUR/MWh)", ln=1)

                pdf_add_section_title(pdf, f"Résultats Financiers sur la période de {total_days} jours (HT)")
                pdf.cell(0, 6, f"Facture contrat actuel (HC/HP) : {fmt_fr(cost_hchp_tot, 0)} EUR", ln=1)
                pdf.cell(0, 6, f"Facture Spot SANS Altileo : {fmt_fr(cost_spot_tot, 0)} EUR", ln=1)
                pdf.cell(0, 6, f"Facture Spot AVEC Altileo : {fmt_fr(c_alt_tot, 0)} EUR", ln=1)
                if equipe_inverter:
                    pdf.cell(0, 6, f"Gain usure mécanique (Inverter, estimation) : {fmt_fr(usure_periode, 0)} EUR", ln=1)
                pdf.cell(0, 6, f"Gain Altileo sur Spot : {fmt_fr(gain_spot_net, 0)} EUR (Brut : {fmt_fr(gain_spot_brut, 0)} EUR)", ln=1)
                pdf.cell(0, 6, f"Gain total vs contrat actuel : {fmt_fr(gain_vs_actuel, 0)} EUR", ln=1)
                pdf.cell(0, 6, f"Impact carbone : {fmt_fr((kwh_saved_tot * CO2_FACTOR_KG_PER_KWH) / 1000, 2)} tonnes de CO2 évitées", ln=1)
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 5, f"(facteur retenu : {CO2_FACTOR_KG_PER_KWH * 1000:.0f} gCO2/kWh -- {CO2_FACTOR_SOURCE})", ln=1)
                pdf.set_font("Arial", "", 11)
                pdf.cell(0, 6, f"Validation HACCP : {'ÉCHEC' if max_temp_spot > t_max else 'CONFORME'} (Temp max : {max_temp_spot:.2f} C)", ln=1)

                pdf_add_footer(pdf)
                return bytes(pdf.output())

            st.divider()
            spot_pdf_bytes = create_spot_pdf()
            st.download_button(
                label="Télécharger le rapport PDF Spot",
                data=spot_pdf_bytes,
                file_name=f"Altileo_Audit_Spot_{date.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                type="primary"
            )

            # --- Graphique Barres Mensuel ---
            st.divider()
            st.markdown('<p class="section-title">Évolution Mensuelle de la Facture</p>', unsafe_allow_html=True)
            monthly_table = monthly.copy()
            monthly_table.columns = ['Mois', 'Jours', 'Prix moy (EUR/MWh)', 'HC/HP de base (EUR)', 'HC/HP avec Altileo (EUR)', 'Spot seul (EUR)', 'Spot+Altileo (EUR)', 'kWh eco.', 'Gain Altileo (EUR)', 'Temp Max (C)']
            for c in ['Prix moy (EUR/MWh)', 'HC/HP de base (EUR)', 'HC/HP avec Altileo (EUR)', 'Spot seul (EUR)', 'Spot+Altileo (EUR)', 'kWh eco.', 'Gain Altileo (EUR)', 'Temp Max (C)']:
                monthly_table[c] = monthly_table[c].round(1)
            
            # --- Graphique Barres Mensuel ---
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_table['Mois'], y=monthly_table['Spot seul (EUR)'],
                name='Spot SANS Altileo', marker_color='#D14343'
            ))
            fig_monthly.add_trace(go.Bar(
                x=monthly_table['Mois'], y=monthly_table['Spot+Altileo (EUR)'],
                name='Spot AVEC Altileo', marker_color='#2A9D6E'
            ))
            fig_monthly.update_layout(
                barmode='group',
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=10, color='#555555'),
                yaxis=dict(title="Coût Mensuel (EUR)", showgrid=True, gridcolor='rgba(229,229,229,0.7)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_monthly, use_container_width=True, config={'displaylogo': False})
            
            st.dataframe(monthly_table, use_container_width=True, hide_index=True)

            # --- Figures pour l'onglet graphique ---
            fig_evo = go.Figure()
            fig_evo.add_trace(go.Scatter(
                x=res_df['date'], y=res_df['avg_price'],
                mode='lines', name='Prix moyen',
                line=dict(color='#111111', width=1.5),
                hovertemplate='%{x|%d/%m/%Y}<br>Moy : %{y:.1f} EUR/MWh<extra></extra>'
            ))
            fig_evo.add_trace(go.Scatter(
                x=res_df['date'], y=res_df['max_price'],
                mode='lines', name='Prix max',
                line=dict(color='#0E7C8C', width=1, dash='dot'),
                hovertemplate='%{x|%d/%m/%Y}<br>Max : %{y:.1f} EUR/MWh<extra></extra>'
            ))
            fig_evo.add_trace(go.Scatter(
                x=res_df['date'], y=res_df['min_price'],
                mode='lines', name='Prix min',
                line=dict(color='#999999', width=1, dash='dot'),
                hovertemplate='%{x|%d/%m/%Y}<br>Min : %{y:.1f} EUR/MWh<extra></extra>'
            ))
            fig_evo.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=10, color='#555555'),
                yaxis=dict(title="EUR / MWh", showgrid=True, gridcolor='rgba(229,229,229,0.7)'),
                xaxis=dict(showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=350
            )

            fig_compare = go.Figure()
            scenarios = ["Contrat actuel\n(HC/HP)", "Spot seul\n(sans Altileo)", "Spot + Altileo"]
            values = [f_ref_an, cost_spot_annual, cost_altileo_annual + saas * nb]
            colors_bar = ['#111111', '#999999', '#2A9D6E']
            fig_compare.add_trace(go.Bar(
                x=scenarios, y=values, marker_color=colors_bar,
                text=[f"{fmt_fr(v, 0)} EUR" for v in values], textposition='outside',
                hovertemplate='<b>%{x}</b><br>%{y:,.0f} EUR/an<extra></extra>'
            ))
            fig_compare.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=11, color='#555555'),
                yaxis=dict(title="EUR / an (HT)", showgrid=True, gridcolor='rgba(229,229,229,0.7)'),
                xaxis=dict(showgrid=False), showlegend=False, height=400
            )

            # --- Textes et hovers pour comparaison mensuelle ---
            text_hchp = []
            text_hchp_sim = []
            text_spot = []
            text_altileo = []
            hover_hchp = []
            hover_hchp_sim = []
            hover_spot = []
            hover_altileo = []

            # Calculer la série "HC/HP avec Altileo (SaaS inc)" pour Plotly
            monthly_table['HC/HP avec Altileo (SaaS inc) (EUR)'] = monthly_table['HC/HP avec Altileo (EUR)'] + (saas * nb) * (monthly_table['Jours'] / 365.0)

            for idx, row in monthly_table.iterrows():
                c_base = row['HC/HP de base (EUR)']
                c_hchp_sim = row['HC/HP avec Altileo (SaaS inc) (EUR)']
                c_spot = row['Spot seul (EUR)']
                c_altileo = row['Spot+Altileo (EUR)']
                saas_mois = (saas * nb) * (row['Jours'] / 365.0)
                
                text_hchp.append(f"{fmt_fr(c_base, 0)} €")
                hover_hchp.append(f"<b>{row['Mois']}</b><br>HC/HP de base : {fmt_fr(c_base, 0)} EUR<extra></extra>")
                
                if c_base > 0:
                    pct_hchp_sim = (c_hchp_sim - c_base) / c_base * 100
                    pct_spot = (c_spot - c_base) / c_base * 100
                    pct_altileo = (c_altileo - c_base) / c_base * 100
                    text_hchp_sim.append(f"{fmt_fr(c_hchp_sim, 0)} €<br>({pct_hchp_sim:+.1f}%)")
                    text_spot.append(f"{fmt_fr(c_spot, 0)} €<br>({pct_spot:+.1f}%)")
                    text_altileo.append(f"{fmt_fr(c_altileo, 0)} €<br>({pct_altileo:+.1f}%)")
                else:
                    text_hchp_sim.append(f"{fmt_fr(c_hchp_sim, 0)} €")
                    text_spot.append(f"{fmt_fr(c_spot, 0)} €")
                    text_altileo.append(f"{fmt_fr(c_altileo, 0)} €")
                    pct_hchp_sim = 0
                    pct_spot = 0
                    pct_altileo = 0
                
                pct_vs_spot = (c_altileo - c_spot) / c_spot * 100 if c_spot > 0 else 0
                
                hover_hchp_sim.append(
                    f"<b>{row['Mois']}</b><br>"
                    f"HC/HP avec Altileo : {fmt_fr(c_hchp_sim, 0)} EUR<br>"
                    f"Gain vs Base : {pct_hchp_sim:+.1f}%<extra></extra>"
                )
                hover_spot.append(
                    f"<b>{row['Mois']}</b><br>"
                    f"Spot seul : {fmt_fr(c_spot, 0)} EUR<br>"
                    f"Diff vs HC/HP : {pct_spot:+.1f}%<extra></extra>"
                )
                hover_altileo.append(
                    f"<b>{row['Mois']}</b><br>"
                    f"Spot + Altileo : {fmt_fr(c_altileo, 0)} EUR<br>"
                    f"Gain vs HC/HP : {pct_altileo:+.1f}%<br>"
                    f"Gain vs Spot seul : {pct_vs_spot:+.1f}%<extra></extra>"
                )

            # --- Graphique de comparaison mensuelle ---
            fig_monthly_compare = go.Figure()
            fig_monthly_compare.add_trace(go.Bar(
                x=monthly_table['Mois'],
                y=monthly_table['HC/HP de base (EUR)'],
                name='HC/HP de base (sans délestage)',
                marker_color='#999999',
                text=text_hchp,
                textposition='outside',
                hovertemplate='%{customdata}'
            ))
            fig_monthly_compare.data[-1].customdata = hover_hchp

            fig_monthly_compare.add_trace(go.Bar(
                x=monthly_table['Mois'],
                y=monthly_table['HC/HP avec Altileo (SaaS inc) (EUR)'],
                name='HC/HP avec Altileo',
                marker_color='#0E7C8C',
                text=text_hchp_sim,
                textposition='outside',
                hovertemplate='%{customdata}'
            ))
            fig_monthly_compare.data[-1].customdata = hover_hchp_sim
            
            fig_monthly_compare.add_trace(go.Bar(
                x=monthly_table['Mois'],
                y=monthly_table['Spot seul (EUR)'],
                name='Spot seul (sans délestage)',
                marker_color='#111111',
                text=text_spot,
                textposition='outside',
                hovertemplate='%{customdata}'
            ))
            fig_monthly_compare.data[-1].customdata = hover_spot
            
            fig_monthly_compare.add_trace(go.Bar(
                x=monthly_table['Mois'],
                y=monthly_table['Spot+Altileo (EUR)'],
                name='Spot + Altileo',
                marker_color='#2A9D6E',
                text=text_altileo,
                textposition='outside',
                hovertemplate='%{customdata}'
            ))
            fig_monthly_compare.data[-1].customdata = hover_altileo

            fig_monthly_compare.update_layout(
                barmode='group',
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=11, color='#555555'),
                yaxis=dict(title="Coût mensuel (EUR HT)", showgrid=True, gridcolor='rgba(229,229,229,0.7)'),
                xaxis=dict(showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450
            )

# ----------------------------------------------------------
# ONGLET 6 : GRAPHIQUES (PRIX SPOT)
# ----------------------------------------------------------
with tab_spot_charts:
    st.markdown('<p class="section-title">Visualisation de la simulation Spot (Données Nord Pool)</p>', unsafe_allow_html=True)
    if not analysis_run:
        st.info("Lancez d'abord l'analyse dans l'onglet Simulation Délestage (Contrat HC/HP).")
    elif spot_df is None or spot_df.empty:
        st.error("Impossible de charger les données Nord Pool pour les graphiques.")
    else:
        st.markdown('<p class="section-title">Comparaison mensuelle : Spot seul vs Spot+Altileo (HT)</p>', unsafe_allow_html=True)
        if fig_monthly_compare is not None:
            st.plotly_chart(fig_monthly_compare, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

        st.markdown('<p class="section-title">Comparaison visuelle globale des scénarios (annuel HT)</p>', unsafe_allow_html=True)
        if fig_compare is not None:
            st.plotly_chart(fig_compare, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

        st.markdown('<p class="section-title">Évolution du prix Spot moyen journalier</p>', unsafe_allow_html=True)
        if fig_evo is not None:
            st.plotly_chart(fig_evo, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

# ----------------------------------------------------------
# ONGLET 7 : VALIDATION THERMIQUE
# ----------------------------------------------------------
with tab_thermal:
    st.markdown('<p class="section-title">Validation Thermique HACCP (Simulation Interactive)</p>', unsafe_allow_html=True)
    if not analysis_run:
        st.info("Lancez d'abord l'analyse dans le panneau latéral.")
    else:
        st.markdown("Simulez l'impact du délestage sur la température à cœur de la marchandise (ex : carton).")
        
        scenario = st.radio("Scénario de délestage à simuler", ["Optimisation Spot (Moyenne)", "Contrat classique (HC/HP)"], horizontal=True)
        
        if scenario == "Optimisation Spot (Moyenne)":
            if spot_df is None or spot_df.empty:
                st.warning("Veuillez lancer l'analyse avec un historique Spot valide pour voir ce scénario.")
                shed_hours = []
            else:
                shed_hours = top_hours_avg
                st.markdown(f"**Heures coupées (Spot) :** {', '.join([f'{hh}h' for hh in sorted(shed_hours)])}")
        else:
            c1, c2 = st.columns(2)
            with c1:
                start_h = st.number_input("Heure de début de coupure (HC/HP)", min_value=0, max_value=23, value=18)
            with c2:
                st.markdown(f"<br>**Durée de coupure retenue :** {h} heures", unsafe_allow_html=True)
            shed_hours = [(int(start_h) + i) % 24 for i in range(int(h))]
            st.markdown(f"**Heures coupées (HC/HP) :** {', '.join([f'{hh}h' for hh in sorted(shed_hours)])}")

        # --- Simulation 24h ---
        temp_24h = [t_consigne]
        current_t = t_consigne
        
        for hh in range(24):
            if hh in shed_hours:
                current_t = temp_rechauffement(current_t, 1.0, t_ambiante, pente_rechauffement, t_consigne)
            else:
                current_t = temp_refroidissement(current_t, 1.0, t_consigne, pente_refroidissement)
            temp_24h.append(current_t)
            
        fig_24h = go.Figure()
        fig_24h.add_trace(go.Scatter(x=list(range(25)), y=temp_24h, mode='lines', name='Température Produit', line=dict(color='#111111', width=3)))
        fig_24h.add_hline(y=t_consigne, line_dash="dash", line_color="#2A9D6E", annotation_text=f"Consigne ({t_consigne}°C)", annotation_position="top left")
        fig_24h.add_hline(y=t_max, line_dash="dash", line_color="#D14343", annotation_text=f"Limite HACCP ({t_max}°C)", annotation_position="bottom left")

        for hh in shed_hours:
            fig_24h.add_vrect(x0=hh, x1=hh+1, fillcolor="rgba(0,164,180,0.1)", layer="below", line_width=0)

        fig_24h.update_layout(
            title="Cycle journalier (24 heures) - Zone teal = compresseur coupé",
            xaxis_title="Heure de la journée", yaxis_title="Température (°C)",
            margin=dict(l=10, r=10, t=40, b=10), height=350,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#555555'),
            yaxis=dict(showgrid=True, gridcolor='rgba(229,229,229,0.7)'), xaxis=dict(showgrid=False)
        )
        
        # --- Simulation 168h (7 jours) ---
        temp_168h = [t_consigne]
        current_t = t_consigne
        
        for day in range(7):
            for hh in range(24):
                if hh in shed_hours:
                    current_t = temp_rechauffement(current_t, 1.0, t_ambiante, pente_rechauffement, t_consigne)
                else:
                    current_t = temp_refroidissement(current_t, 1.0, t_consigne, pente_refroidissement)
                temp_168h.append(current_t)
                
        max_168h = max(temp_168h)
        is_conforme = max_168h <= t_max
        
        fig_168h = go.Figure()
        fig_168h.add_trace(go.Scatter(x=list(range(169)), y=temp_168h, mode='lines', name='Température Produit', line=dict(color='#111111', width=2)))
        fig_168h.add_hline(y=t_consigne, line_dash="dash", line_color="#2A9D6E", annotation_text=f"Consigne ({t_consigne}°C)", annotation_position="top left")
        fig_168h.add_hline(y=t_max, line_dash="dash", line_color="#D14343", annotation_text=f"Limite HACCP ({t_max}°C)", annotation_position="bottom left")
        
        for day in range(7):
            fig_168h.add_vline(x=day*24, line_dash="dot", line_color="rgba(153,153,153,0.5)")
            
        fig_168h.update_layout(
            title="Stress Test inertiel (7 jours / 168 heures)",
            xaxis_title="Heure cumulée", yaxis_title="Température (°C)",
            margin=dict(l=10, r=10, t=40, b=10), height=350,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#555555'),
            yaxis=dict(showgrid=True, gridcolor='rgba(229,229,229,0.7)'), xaxis=dict(showgrid=False)
        )
        
        # Affichage
        c_kpi1, c_kpi2 = st.columns(2)
        with c_kpi1:
            if is_conforme:
                st.success(f"**CONFORME** : La température reste stable et le compresseur a le temps de rattraper la consigne chaque jour.")
            else:
                st.error(f"**RISQUE SANITAIRE** : Dérive thermique détectée. La chaleur s'accumule de jour en jour.")
        with c_kpi2:
            st.metric(label="Pic de température sur 7 jours", value=f"{max_168h:.2f} °C", delta=f"{max_168h - t_consigne:+.2f} °C vs consigne", delta_color="inverse")
            
        st.plotly_chart(fig_24h, use_container_width=True, config={'displaylogo': False})
        st.plotly_chart(fig_168h, use_container_width=True, config={'displaylogo': False})
