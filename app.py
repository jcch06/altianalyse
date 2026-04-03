import os
import math
import pandas as pd
import streamlit as st
from datetime import datetime, date
from supabase import create_client, Client

# --- FIX ANTI-PLANTAGE GRAPHIQUE ---
import matplotlib
matplotlib.use('Agg') # Force Matplotlib à dessiner sans écran
import matplotlib.pyplot as plt
# ----------------------------------

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Altileo PRO - Audit Energetique", layout="wide")

# Custom CSS pour un rendu épuré et professionnel
st.markdown("""
    <style>
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .main-header {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GESTION DES IDENTIFIANTS (Local vs Web)
# ==========================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if "SUPABASE_URL" in st.secrets:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
else:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ==========================================
# 3. FONCTIONS DE CALCUL & CACHE
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_supabase_data(start_date: str, end_date: str, table_name: str, sensors: list):
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Identifiants Supabase introuvables. Configurez vos secrets Streamlit.")
        return pd.DataFrame()
        
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    start_ts, end_ts = f"{start_date}T00:00:00", f"{end_date}T23:59:59"
    all_data, page_size, start_range = [], 1000, 0
    
    while True:
        response = supabase.table(table_name).select('timestamp, capteur, valeur')\
            .in_('capteur', sensors).gte('timestamp', start_ts)\
            .lte('timestamp', end_ts).order('timestamp', desc=False)\
            .range(start_range, start_range + page_size - 1).execute()
        data = response.data
        if not data: break 
        all_data.extend(data)
        start_range += len(data)
        if len(data) < page_size: break
        
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

@st.cache_data(show_spinner=False)
def process_data(df: pd.DataFrame, sensor_comp: str, sensor_cvc: str, tarifs: dict) -> pd.DataFrame:
    if df.empty: return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['capteur', 'timestamp']).reset_index(drop=True)
    processed_dfs = []
    
    cos_phi = tarifs['cos_phi']
    
    for capteur in [sensor_comp, sensor_cvc]:
        df_sensor = df[df['capteur'] == capteur].copy()
        if df_sensor.empty: continue
        
        # Gestion des Gaps et clipping
        df_sensor['delta_hours'] = df_sensor['timestamp'].diff().dt.total_seconds() / 3600.0
        df_sensor['delta_hours'] = df_sensor['delta_hours'].fillna(0).clip(upper=30.0/60.0)
        
        if capteur == sensor_comp:
            df_sensor['puissance_kw'] = (math.sqrt(3) * 400 * df_sensor['valeur'] * cos_phi) / 1000.0
            df_sensor['categorie_conso'] = 'compresseur'
        else:
            df_sensor['puissance_kw'] = (230 * df_sensor['valeur'] * 1.0) / 1000.0
            df_sensor['categorie_conso'] = 'hvac'
            df_sensor.loc[df_sensor['valeur'] > 6.0, 'categorie_conso'] = 'resistance'
        
        # Integration trapézoïdale
        df_sensor['puissance_moyenne'] = (df_sensor['puissance_kw'] + df_sensor['puissance_kw'].shift(1).fillna(df_sensor['puissance_kw'])) / 2.0
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
        
        df_sensor['cout_euros'] = df_sensor['energie_kwh'] * (tarif + tarifs['turpe'] + tarifs['taxes']) * 1.20
        processed_dfs.append(df_sensor)
        
    return pd.concat(processed_dfs, ignore_index=True)

@st.cache_data(show_spinner=False)
def generate_detailed_dataframes(df: pd.DataFrame):
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    daily_records = []
    for d in sorted(df['date'].unique()):
        df_d = df[df['date'] == d]
        def get_val(cat, th):
            return df_d[(df_d['categorie_conso']==cat) & (df_d['type_heure']==th)]['energie_kwh'].sum()
        
        c_hp, c_hc = get_val('compresseur', 'HP'), get_val('compresseur', 'HC')
        r_hp, r_hc = get_val('resistance', 'HP'), get_val('resistance', 'HC')
        v_hp, v_hc = get_val('hvac', 'HP'), get_val('hvac', 'HC')
        tot = c_hp + c_hc + r_hp + r_hc + v_hp + v_hc
        
        daily_records.append({
            'Date': str(d), 'Comp. HP (kWh)': round(c_hp, 2), 'Comp. HC (kWh)': round(c_hc, 2),
            'Résist. HP (kWh)': round(r_hp, 2), 'Résist. HC (kWh)': round(r_hc, 2),
            'CVC HP (kWh)': round(v_hp, 2), 'CVC HC (kWh)': round(v_hc, 2), 'Total (kWh)': round(tot, 2)
        })
        
    df_w = df.copy()
    df_w['timestamp'] = pd.to_datetime(df_w['timestamp'])
    df_w['week'] = df_w['timestamp'].dt.to_period('W-SUN').apply(lambda r: f"{r.start_time.strftime('%d/%m/%Y')} - {r.end_time.strftime('%d/%m/%Y')}")
    weekly_records = []
    
    for w in sorted(df_w['week'].unique()):
        df_w_data = df_w[df_w['week'] == w]
        def get_val_w(cat, th):
            return df_w_data[(df_w_data['categorie_conso']==cat) & (df_w_data['type_heure']==th)]['energie_kwh'].sum()
            
        c_hp, c_hc = get_val_w('compresseur', 'HP'), get_val_w('compresseur', 'HC')
        r_hp, r_hc = get_val_w('resistance', 'HP'), get_val_w('resistance', 'HC')
        v_hp, v_hc = get_val_w('hvac', 'HP'), get_val_w('hvac', 'HC')
        tot = c_hp + c_hc + r_hp + r_hc + v_hp + v_hc
        
        weekly_records.append({
            'Semaine': w, 'Comp. HP (kWh)': round(c_hp, 2), 'Comp. HC (kWh)': round(c_hc, 2),
            'Résist. HP (kWh)': round(r_hp, 2), 'Résist. HC (kWh)': round(r_hc, 2),
            'CVC HP (kWh)': round(v_hp, 2), 'CVC HC (kWh)': round(v_hc, 2), 'Total (kWh)': round(tot, 2)
        })

    return pd.DataFrame(daily_records), pd.DataFrame(weekly_records)

def calculate_period_summary(df: pd.DataFrame, saison: str, tarifs: dict) -> dict:
    if df.empty: return {}
    
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
    total_ttc = total_ht * 1.20
    
    return {
        'kwh_mois_hp': kwh_mois_hp, 'kwh_mois_hc': kwh_mois_hc, 'kwh_mois_total': kwh_mois_hp + kwh_mois_hc,
        'tarif_hp': tarif_hp, 'tarif_hc': tarif_hc, 'tarif_turpe': tarifs['turpe'], 'tarif_taxes': tarifs['taxes'],
        'f_hp': f_hp, 'f_hc': f_hc, 'turpe': turpe, 'taxes': taxes,
        'tva': total_ht * 0.20, 'total_ttc': total_ttc
    }

# ==========================================
# 4. INTERFACE UTILISATEUR
# ==========================================
st.markdown("<h2 class='main-header'>Plateforme Altileo Analyse</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration Données")
    t_name = st.text_input("Table Supabase", "mesures_fady")
    
    col_cap1, col_cap2 = st.columns(2)
    with col_cap1: s_comp = st.text_input("Capteur Froid", "courant_1")
    with col_cap2: s_cvc = st.text_input("Capteur CVC", "courant_2")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1: d_start = st.date_input("Début", date(2025, 11, 1))
    with col_d2: d_end = st.date_input("Fin", date(2026, 3, 31))

    with st.expander("Paramètres Physiques & Tarifs"):
        tarifs = {
            'cos_phi': st.number_input("Cos Phi (Froid)", value=0.92, format="%.2f"),
            'hp_hiv': st.number_input("HP Hiver (€)", value=0.12618, format="%.5f"),
            'hc_hiv': st.number_input("HC Hiver (€)", value=0.09779, format="%.5f"),
            'hp_ete': st.number_input("HP Été (€)", value=0.07869, format="%.5f"),
            'hc_ete': st.number_input("HC Été (€)", value=0.07424, format="%.5f"),
            'turpe': st.number_input("TURPE (€)", value=0.04000, format="%.5f"),
            'taxes': st.number_input("Taxes (CSPE) (€)", value=0.02100, format="%.5f")
        }

    st.markdown("---")
    st.header("Modélisation Altileo")
    h = st.slider("Délestage HP (heures)", 0.0, 16.0, 8.0, 0.5)
    r = st.slider("Rattrapage HC (%)", 0, 100, 20, 1) / 100.0
    sec = st.slider("Marge Sécurité (%)", 0, 50, 15, 1) / 100.0
    cop = st.slider("Bonus COP Nuit (%)", 0, 50, 5, 1) / 100.0
    
    abo = st.number_input("Gain Abo kVA (€/mois/ch)", value=50.0)
    maint = st.number_input("Maintenance (€/an/ch)", value=150.0)
    
    st.markdown("---")
    st.header("Déploiement")
    nb = st.number_input("Nombre de Chambres", min_value=1, value=1)
    pr = st.number_input("Prix Install/ch (€)", value=2500.0)
    cee = st.number_input("Prime CEE Globale (€)", value=1500.0)

# ==========================================
# 5. MOTEUR D'EXÉCUTION (MODIFIÉ)
# ==========================================

if st.button("🚀 Lancer l'Analyse Altileo", type="primary", use_container_width=True):
    st.session_state['run_analysis'] = True

if not st.session_state.get('run_analysis', False):
    st.info("👈 Configurez vos paramètres sur la gauche puis cliquez sur le bouton 'Lancer l'Analyse Altileo' ci-dessus.")
else:
    with st.spinner("Analyse thermodynamique et financière en cours..."):
        df_raw = fetch_supabase_data(str(d_start), str(d_end), t_name, [s_comp, s_cvc])
        
        if df_raw.empty:
            st.warning("Aucune donnée disponible pour cette période ou table.")
            st.stop()
            
        df_proc = process_data(df_raw, s_comp, s_cvc, tarifs)
        
        sum_h = calculate_period_summary(df_proc[df_proc['saison']=='hiver'], 'hiver', tarifs)
        sum_e = calculate_period_summary(df_proc[df_proc['saison']=='ete'], 'ete', tarifs)
        
        s_h = sum_h if sum_h else sum_e
        s_e = sum_e if sum_e else sum_h
        
        # KPIs Référence
        f_ref_an = (s_h.get('total_ttc', 0) * nb * 5) + (s_e.get('total_ttc', 0) * nb * 7)
        k_ref_an = (s_h.get('kwh_mois_total', 0) * nb * 5) + (s_e.get('kwh_mois_total', 0) * nb * 7)

        # Fonction de Simulation Financière
        def get_sim_metrics(s):
            if not s: return 0.0, 0.0
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
            
            total_ttc_sim = (f_hp_sim + f_hc_sim + turpe_sim + taxes_sim) * 1.20 - (abo * nb)
            gain = (s.get('total_ttc', 0) * nb) - total_ttc_sim
            
            return gain, (k_effaces - k_rattrapes)

        gh, kh = get_sim_metrics(s_h)
        ge, ke = get_sim_metrics(s_e)
        
        gain_an_brut = (gh * 5) + (ge * 7)
        gain_an_net = gain_an_brut - (maint * nb)
        k_sauves_an = (kh * 5) + (ke * 7)
        pct = (gain_an_brut / f_ref_an * 100) if f_ref_an > 0 else 0
        
        i_net = max(0, pr*nb - cee)
        roi = (i_net / gain_an_net * 12) if gain_an_net > 0 else 0

    # ==========================================
    # 6. AFFICHAGE DES RÉSULTATS
    # ==========================================
    tab_dashboard, tab_details, tab_charts = st.tabs(["Indicateurs Financiers", "Audit Détaillé (Jours/Semaines)", "Modélisation Graphique"])

    with tab_dashboard:
        st.markdown("### Synthèse Annuelle du Projet")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Facture de Référence", value=f"{f_ref_an:,.0f} €", delta=f"{k_ref_an:,.0f} kWh", delta_color="off")
        with col2:
            st.metric(label="Gain Net d'Exploitation", value=f"{gain_an_net:,.0f} € / an", delta=f"-{pct:.1f} % (Brut)")
        with col3:
            st.metric(label="Impact Environnemental", value=f"{(k_sauves_an*0.05)/1000:,.1f} t", delta="CO2 Évité")
        with col4:
            st.metric(label="Retour sur Investissement", value=f"{roi:.1f} mois", delta=f"CAPEX Net: {i_net:,.0f} €", delta_color="inverse")

        st.markdown("---")
        st.markdown("### Paramètres de Modélisation Retenus")
        st.write(f"- **Nombre de chambres équipées :** {nb}")
        st.write(f"- **Temps de délestage journalier :** {h} heures")
        st.write(f"- **Taux de rattrapage thermique :** {r*100:.0f} % (avec marge de sécurité de {sec*100:.0f} %)")
        st.write(f"- **Coût de maintenance provisionné :** {maint*nb:.0f} € / an")

    with tab_details:
        df_daily, df_weekly = generate_detailed_dataframes(df_proc)
        
        st.markdown("### Bilan Hebdomadaire (Sommes en kWh)")
        st.dataframe(df_weekly, use_container_width=True, hide_index=True)
        
        st.markdown("### Bilan Journalier (Détail des cycles HP/HC)")
        st.dataframe(df_daily, use_container_width=True, hide_index=True)

    with tab_charts:
        st.markdown("### Évolution Mensuelle de la Facture")
        
        def draw_chart(s, title):
            fig, ax = plt.subplots(figsize=(6, 5))
            if not s: ax.axis('off'); return fig
            
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
            
            tot_ttc_ap = (f_hp_ap + f_hc_ap + t_ap + tax_ap) * 1.20 - (abo * nb)
            
            b_av = [s.get('f_hc', 0)*nb, s.get('f_hp', 0)*nb, s.get('turpe', 0)*nb, s.get('taxes', 0)*nb, s.get('tva', 0)*nb]
            b_ap = [f_hc_ap, f_hp_ap, t_ap, tax_ap, max(0, tot_ttc_ap - (tot_ttc_ap/1.2))]
            
            clrs = ['#3498db', '#fd7e14', '#95a5a6', '#adb5bd', '#198754']
            c_av, c_ap = 0, 0
            for i in range(5):
                ax.bar('Actuel', b_av[i], bottom=c_av, color=clrs[i], width=0.5)
                ax.bar('Altileo', b_ap[i], bottom=c_ap, color=clrs[i], width=0.5)
                if b_av[i]>15: ax.text(0, c_av+b_av[i]/2, f"{b_av[i]:.0f}€", ha='center', color='w', fontweight='bold', fontsize=9)
                if b_ap[i]>15: ax.text(1, c_ap+b_ap[i]/2, f"{b_ap[i]:.0f}€", ha='center', color='w', fontweight='bold', fontsize=9)
                c_av += b_av[i]
                c_ap += b_ap[i]
                
            ax.set_title(f"Saison {title} (TTC)", pad=20, fontweight="bold")
            ax.set_ylabel("Euros")
            
            g = c_av - c_ap
            if g > 0:
                ax.annotate(f"-{g:.0f}€\n(-{g/c_av*100:.1f}%)", xy=(1, c_ap), xytext=(0, c_av), 
                            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2), 
                            color="#2c3e50", fontweight="bold", ha='center', 
                            bbox=dict(boxstyle="round", fc="w", ec="#2c3e50"))
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            return fig

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.pyplot(draw_chart(s_h, "HIVER"), use_container_width=True)
        with col_chart2:
            st.pyplot(draw_chart(s_e, "ETE"), use_container_width=True)