import os
import math
import pandas as pd
import threading
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ==========================================
# CONFIGURATION & CONNEXION SUPABASE
# ==========================================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not os.path.exists('cache'):
    os.makedirs('cache')

class EnergyCalculator:
    def __init__(self, supabase_url: str, supabase_key: str):
        if not supabase_url or not supabase_key:
            raise ValueError("Credentials Supabase manquants.")
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
    def fetch_data(self, start_date: str, end_date: str, table_name: str, sensors: list) -> pd.DataFrame:
        cache_file = f"cache/{table_name}_{start_date}_{end_date}.pkl"
        if os.path.exists(cache_file):
            print(f"\n[⚡ CACHE] Données trouvées en local ! Chargement instantané...")
            return pd.read_pickle(cache_file)

        print(f"\n[☁️ API] Téléchargement Supabase ({table_name}) du {start_date} au {end_date}...")
        start_ts, end_ts = f"{start_date}T00:00:00", f"{end_date}T23:59:59"
        all_data, page_size, start_range = [], 1000, 0
        
        while True:
            response = self.supabase.table(table_name).select('timestamp, capteur, valeur')\
                .in_('capteur', sensors).gte('timestamp', start_ts)\
                .lte('timestamp', end_ts).order('timestamp', desc=False)\
                .range(start_range, start_range + page_size - 1).execute()
            data = response.data
            if not data: break 
            all_data.extend(data)
            start_range += len(data)
            if len(data) < page_size: break
            
        df = pd.DataFrame(all_data) if all_data else pd.DataFrame()
        if not df.empty:
            df.to_pickle(cache_file)
            print(f"[⚡ CACHE] Sauvegarde locale OK.")
        return df

    def process_data(self, df: pd.DataFrame, sensor_comp: str, sensor_cvc: str, tarifs: dict) -> pd.DataFrame:
        if df.empty: return df
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['capteur', 'timestamp']).reset_index(drop=True)
        processed_dfs = []
        
        cos_phi = tarifs['cos_phi'] # Récupéré de l'UI
        
        for capteur in [sensor_comp, sensor_cvc]:
            df_sensor = df[df['capteur'] == capteur].copy()
            if df_sensor.empty: continue
            
            # CORRECTION 2: Gaps de 30 min max et clipping (Conservation de l'énergie)
            df_sensor['delta_hours'] = df_sensor['timestamp'].diff().dt.total_seconds() / 3600.0
            df_sensor['delta_hours'] = df_sensor['delta_hours'].fillna(0).clip(upper=30.0/60.0)
            
            # CORRECTION 1 & 3: Cos Phi dynamique et Résistance sans shift()
            if capteur == sensor_comp:
                df_sensor['puissance_kw'] = (math.sqrt(3) * 400 * df_sensor['valeur'] * cos_phi) / 1000.0
                df_sensor['categorie_conso'] = 'compresseur'
            else:
                df_sensor['puissance_kw'] = (230 * df_sensor['valeur'] * 1.0) / 1000.0
                df_sensor['categorie_conso'] = 'hvac'
                df_sensor.loc[df_sensor['valeur'] > 6.0, 'categorie_conso'] = 'resistance'
            
            # CORRECTION 4: Méthode trapézoïdale pour l'énergie
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

    def generate_detailed_log(self, df: pd.DataFrame) -> str:
        if df.empty: return "Aucune donnée traitée."
        lines = []
        lines.append("=" * 105)
        lines.append(f"{'DATE':<12} | {'COMPRESSEUR (HP / HC)':<24} | {'RÉSISTANCE (HP / HC)':<24} | {'CVC (HP / HC)':<18} | TOTAL JOUR")
        lines.append("=" * 105)
        dates = sorted(df['date'].unique())
        for d in dates:
            df_d = df[df['date'] == d]
            def get_val(cat, th): return df_d[(df_d['categorie_conso']==cat) & (df_d['type_heure']==th)]['energie_kwh'].sum()
            c_hp, c_hc = get_val('compresseur', 'HP'), get_val('compresseur', 'HC')
            r_hp, r_hc = get_val('resistance', 'HP'), get_val('resistance', 'HC')
            v_hp, v_hc = get_val('hvac', 'HP'), get_val('hvac', 'HC')
            tot = c_hp + c_hc + r_hp + r_hc + v_hp + v_hc
            lines.append(f"{str(d):<12} | {c_hp:>6.1f} / {c_hc:>6.1f} kWh{'':<4} | {r_hp:>6.1f} / {r_hc:>6.1f} kWh{'':<4} | {v_hp:>5.1f} / {v_hc:>5.1f} kWh{'':<2} | {tot:>7.1f} kWh")
            
        lines.append("\n" + "=" * 105)
        lines.append(" "*35 + "BILAN HEBDOMADAIRE (Totaux par semaine)")
        lines.append("=" * 105)
        df_w = df.copy()
        df_w['week'] = df_w['timestamp'].dt.to_period('W-SUN').apply(lambda r: f"Du {r.start_time.strftime('%d/%m')} au {r.end_time.strftime('%d/%m/%Y')}")
        for w in sorted(df_w['week'].unique()):
            df_w_data = df_w[df_w['week'] == w]
            def get_val_w(cat, th): return df_w_data[(df_w_data['categorie_conso']==cat) & (df_w_data['type_heure']==th)]['energie_kwh'].sum()
            c_hp, c_hc = get_val_w('compresseur', 'HP'), get_val_w('compresseur', 'HC')
            r_hp, r_hc = get_val_w('resistance', 'HP'), get_val_w('resistance', 'HC')
            v_hp, v_hc = get_val_w('hvac', 'HP'), get_val_w('hvac', 'HC')
            tot = c_hp + c_hc + r_hp + r_hc + v_hp + v_hc
            lines.append(f"{w:<24} | Comp: {c_hp:>5.1f} / {c_hc:>5.1f} | Rés: {r_hp:>5.1f} / {r_hc:>5.1f} | CVC: {v_hp:>4.1f} / {v_hc:>4.1f} | Tot: {tot:>7.1f} kWh")
        return "\n".join(lines)

    def calculate_period_summary(self, df: pd.DataFrame, saison: str, tarifs: dict) -> dict:
        if df.empty: return {}
        df['heure'] = df['timestamp'].dt.hour
        jour_kw = df.groupby(['type_heure', 'heure'])['puissance_kw'].mean().reset_index()
        kwh_jour_hp = jour_kw[jour_kw['type_heure'] == 'HP']['puissance_kw'].sum()
        kwh_jour_hc = jour_kw[jour_kw['type_heure'] == 'HC']['puissance_kw'].sum()
        kwh_mois_hp, kwh_mois_hc = kwh_jour_hp * 30.0, kwh_jour_hc * 30.0
        tarif_hp = tarifs['hp_hiv'] if saison == 'hiver' else tarifs['hp_ete']
        tarif_hc = tarifs['hc_hiv'] if saison == 'hiver' else tarifs['hc_ete']
        f_hp, f_hc = kwh_mois_hp * tarif_hp, kwh_mois_hc * tarif_hc
        turpe, taxes = (kwh_mois_hp + kwh_mois_hc) * tarifs['turpe'], (kwh_mois_hp + kwh_mois_hc) * tarifs['taxes']
        total_ht = f_hp + f_hc + turpe + taxes
        
        return {
            'kwh_mois_hp': kwh_mois_hp, 'kwh_mois_hc': kwh_mois_hc, 'kwh_mois_total': kwh_mois_hp + kwh_mois_hc,
            'tarif_hp': tarif_hp, 'tarif_hc': tarif_hc, 'tarif_turpe': tarifs['turpe'], 'tarif_taxes': tarifs['taxes'],
            'f_hp': f_hp, 'f_hc': f_hc, 'turpe': turpe, 'taxes': taxes,
            'tva': total_ht * 0.20, 'total_ttc': total_ht * 1.20
        }

# ==========================================
# INTERFACE MODERNE
# ==========================================
class EnergyApp(tb.Window):
    def __init__(self, calculator):
        super().__init__(themename="cosmo")
        self.calculator = calculator
        self.title("Altileo PRO - Audit & Modélisation SaaS v3 (Thermodynamique Exacte)")
        self.geometry("1750x980")
        self.create_widgets()
        
    def create_widgets(self):
        main = tb.Frame(self)
        main.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        left = tb.Frame(main, width=680)
        left.pack(side=LEFT, fill=Y, expand=False, padx=(0, 10))
        right = tb.Frame(main)
        right.pack(side=RIGHT, fill=BOTH, expand=True)
        
        self.fig = Figure(figsize=(10, 6), dpi=95)
        self.ax1, self.ax2 = self.fig.add_subplot(121), self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self.progress = tb.Progressbar(left, mode='indeterminate', bootstyle="success")
        self.progress.pack(fill=X, pady=(0, 10))

        nb = tb.Notebook(left)
        nb.pack(fill=BOTH, expand=True, pady=5)
        
        tab_data = tb.Frame(nb); tab_roi = tb.Frame(nb); tab_details = tb.Frame(nb)
        nb.add(tab_data, text="📊 Données & Tarifs")
        nb.add(tab_roi, text="🚀 Simulateur Altileo")
        nb.add(tab_details, text="🔍 Audit Détaillé")

        # --- TAB 1 : DATA ---
        db_box = tb.LabelFrame(tab_data, text="Base de Données")
        db_box.pack(fill=X, pady=10, padx=10, ipadx=10, ipady=10)
        tb.Label(db_box, text="Table:").grid(row=0, column=0, padx=2)
        self.cbo_t = tb.Combobox(db_box, values=["mesures_fady"], width=12); self.cbo_t.set("mesures_fady"); self.cbo_t.grid(row=0, column=1)
        tb.Label(db_box, text="Froid:").grid(row=0, column=2, padx=2)
        self.cbo_c = tb.Combobox(db_box, values=["courant_1"], width=10); self.cbo_c.set("courant_1"); self.cbo_c.grid(row=0, column=3)
        tb.Label(db_box, text="CVC:").grid(row=0, column=4, padx=2)
        self.cbo_v = tb.Combobox(db_box, values=["courant_2"], width=10); self.cbo_v.set("courant_2"); self.cbo_v.grid(row=0, column=5)
        
        date_box = tb.Frame(tab_data); date_box.pack(fill=X, pady=10, padx=10)
        tb.Label(date_box, text="Du:").pack(side=LEFT, padx=2)
        self.dt_start = tb.DateEntry(date_box, dateformat='%Y-%m-%d', startdate=datetime(2025, 11, 1)); self.dt_start.pack(side=LEFT, padx=5)
        tb.Label(date_box, text="Au:").pack(side=LEFT, padx=2)
        self.dt_end = tb.DateEntry(date_box, dateformat='%Y-%m-%d', startdate=datetime(2026, 3, 31)); self.dt_end.pack(side=LEFT, padx=5)
        
        tar_box = tb.LabelFrame(tab_data, text="Paramètres Physiques & Électriques")
        tar_box.pack(fill=X, pady=10, padx=10, ipadx=10, ipady=10)
        
        t_vars = [("Cos Phi (Froid):", "0.92"), ("HP Hiver (€):", "0.12618"), ("HC Hiver (€):", "0.09779"), 
                  ("HP Été (€):", "0.07869"), ("HC Été (€):", "0.07424"), ("TURPE (€):", "0.040"), ("Taxes (€):", "0.021")]
        self.t_entries = {}
        for i, (lbl, val) in enumerate(t_vars):
            tb.Label(tar_box, text=lbl).grid(row=i//2, column=(i%2)*2, sticky=W, padx=5, pady=2)
            e = tb.Entry(tar_box, width=10); e.insert(0, val); e.grid(row=i//2, column=(i%2)*2+1, pady=2)
            self.t_entries[lbl] = e

        self.btn_load = tb.Button(tab_data, text="1. GÉNÉRER LA BASELINE EXACTE", bootstyle="primary", command=self.load_data_threaded)
        self.btn_load.pack(fill=X, pady=10, padx=10)
        self.txt_audit = tb.Text(tab_data, height=3, font=("Consolas", 9), state=DISABLED, bg="#f8f9fa", relief="flat")
        self.txt_audit.pack(fill=X, pady=5, padx=10)

        # --- TAB 2 : ROI ---
        roi_box = tb.Frame(tab_roi); roi_box.pack(fill=X, pady=10, padx=10)
        # Ajout des corrections de Marge de Sécurité et de Maintenance
        r_vars = [("Délestage (h):", "8"), ("Rattrapage (%):", "20"), ("Marge Sécurité (%):", "15"), 
                  ("Bonus COP (%):", "5"), ("Gain Abo (€):", "50"), ("Nb Chambres:", "1"), 
                  ("Prix/ch (€):", "2500"), ("Prime CEE (€):", "1500"), ("Maintenance (€/an/ch):", "150")]
        self.r_entries = {}
        for i, (lbl, val) in enumerate(r_vars):
            tb.Label(roi_box, text=lbl).grid(row=i//2, column=(i%2)*2, sticky=W, padx=5, pady=5)
            e = tb.Entry(roi_box, width=10); e.insert(0, val); e.grid(row=i//2, column=(i%2)*2+1, pady=5)
            self.r_entries[lbl] = e
            
        btn_f = tb.Frame(tab_roi); btn_f.pack(fill=X, pady=10, padx=10)
        tb.Button(btn_f, text="2. SIMULER LE BUSINESS PLAN", bootstyle="success", command=self.simulate).pack(side=LEFT, expand=True, fill=X, padx=5)
        self.btn_export = tb.Button(btn_f, text="🖨️ EXPORT", bootstyle="outline-secondary", command=self.export_report, state=DISABLED)
        self.btn_export.pack(side=LEFT, padx=5)

        self.txt_roi = tb.Text(tab_roi, height=18, font=("Consolas", 10, "bold"), bg="#e8f8ec", fg="#198754", state=DISABLED, relief="flat", padx=10, pady=10)
        self.txt_roi.pack(fill=X, pady=5, padx=10)

        # --- TAB 3 : DÉTAILS ---
        scroll = tb.Scrollbar(tab_details); scroll.pack(side=RIGHT, fill=Y)
        self.txt_details = tb.Text(tab_details, font=("Consolas", 9), yscrollcommand=scroll.set, wrap="none")
        self.txt_details.pack(fill=BOTH, expand=True, padx=5, pady=5)
        scroll.config(command=self.txt_details.yview)

    def get_tarifs(self):
        return {
            'cos_phi': float(self.t_entries["Cos Phi (Froid):"].get()),
            'hp_hiv': float(self.t_entries["HP Hiver (€):"].get()), 'hc_hiv': float(self.t_entries["HC Hiver (€):"].get()),
            'hp_ete': float(self.t_entries["HP Été (€):"].get()), 'hc_ete': float(self.t_entries["HC Été (€):"].get()),
            'turpe': float(self.t_entries["TURPE (€):"].get()), 'taxes': float(self.t_entries["Taxes (€):"].get())
        }

    def load_data_threaded(self):
        self.btn_load.config(state=DISABLED)
        self.progress.start()
        self._upd_txt(self.txt_audit, "Extraction et calculs (Thermodynamique & Gaps)...")
        self._upd_txt(self.txt_details, "Calculs en cours...")
        
        t, sc, sv = self.cbo_t.get(), self.cbo_c.get(), self.cbo_v.get()
        d_start, d_end = self.dt_start.entry.get(), self.dt_end.entry.get()
        tarifs = self.get_tarifs()
        
        thread = threading.Thread(target=self._process_data_backend, args=(t, sc, sv, d_start, d_end, tarifs))
        thread.daemon = True
        thread.start()

    def _process_data_backend(self, t, sc, sv, d_start, d_end, tarifs):
        try:
            df = self.calculator.fetch_data(d_start, d_end, t, [sc, sv])
            dfp = self.calculator.process_data(df, sc, sv, tarifs)
            detailed_log_str = self.calculator.generate_detailed_log(dfp)
            sum_h = self.calculator.calculate_period_summary(dfp[dfp['saison']=='hiver'], 'hiver', tarifs)
            sum_e = self.calculator.calculate_period_summary(dfp[dfp['saison']=='ete'], 'ete', tarifs)
            
            self.after(0, self._load_complete, sum_h, sum_e, detailed_log_str, True, "✅ Modèle validé. Passez au Simulateur.")
        except Exception as e:
            self.after(0, self._load_complete, None, None, "", False, str(e))

    def _load_complete(self, sum_h, sum_e, detailed_log, success, msg):
        self.progress.stop()
        self.btn_load.config(state=NORMAL)
        if success:
            self.sum_h = sum_h
            self.sum_e = sum_e
            self._upd_txt(self.txt_audit, msg)
            self._upd_txt(self.txt_details, detailed_log)
            self.draw(0, 0, 0, 0, 1, 0)
        else:
            Messagebox.show_error(f"Erreur d'extraction : {msg}", "Erreur Système")

    def simulate(self):
        if not hasattr(self, 'sum_h'):
            Messagebox.show_warning("Générez d'abord la Baseline !", "Attention")
            return
            
        h = float(self.r_entries["Délestage (h):"].get())
        r = float(self.r_entries["Rattrapage (%):"].get()) / 100
        sec = float(self.r_entries["Marge Sécurité (%):"].get()) / 100
        cop = float(self.r_entries["Bonus COP (%):"].get()) / 100
        abo = float(self.r_entries["Gain Abo (€):"].get())
        nb = int(self.r_entries["Nb Chambres:"].get())
        pr = float(self.r_entries["Prix/ch (€):"].get())
        cee = float(self.r_entries["Prime CEE (€):"].get())
        maint = float(self.r_entries["Maintenance (€/an/ch):"].get())
        
        s_h = self.sum_h if self.sum_h else self.sum_e
        s_e = self.sum_e if self.sum_e else self.sum_h
        
        f_ref_an = (s_h['total_ttc'] * nb * 5) + (s_e['total_ttc'] * nb * 7)
        k_ref_an = (s_h['kwh_mois_total'] * nb * 5) + (s_e['kwh_mois_total'] * nb * 7)
        
        def get_sim_metrics(s):
            if not s: return 0.0, 0.0
            k_hp_base, k_hc_base = s['kwh_mois_hp'] * nb, s['kwh_mois_hc'] * nb
            k_effaces = k_hp_base * (h / 16.0)
            
            # CORRECTION 5: Application du Facteur de Sécurité
            k_rattrapes = k_effaces * r * (1.0 - cop) * (1.0 + sec)
            
            k_hp_sim, k_hc_sim = k_hp_base - k_effaces, k_hc_base + k_rattrapes
            f_hp_sim, f_hc_sim = k_hp_sim * s['tarif_hp'], k_hc_sim * s['tarif_hc']
            turpe_sim, taxes_sim = (k_hp_sim + k_hc_sim) * s['tarif_turpe'], (k_hp_sim + k_hc_sim) * s['tarif_taxes']
            
            total_ttc_sim = (f_hp_sim + f_hc_sim + turpe_sim + taxes_sim) * 1.20 - (abo * nb)
            return (s['total_ttc'] * nb) - total_ttc_sim, (k_effaces - k_rattrapes)

        gh, kh = get_sim_metrics(s_h)
        ge, ke = get_sim_metrics(s_e)
        
        gain_an_brut = (gh * 5) + (ge * 7)
        
        # CORRECTION 6: Calcul du Gain Net (déduction maintenance)
        gain_an_net = gain_an_brut - (maint * nb)
        
        pct = (gain_an_brut / f_ref_an * 100) if f_ref_an > 0 else 0
        i_net = max(0, pr*nb - cee)
        roi = f"{(i_net/gain_an_net*12):.1f} mois" if gain_an_net > 0 else "Jamais (Gain Négatif)"

        res = (f"--- BUSINESS PLAN ALTILEO ({nb} CHAMBRE(S)) ---\n"
               f"📉 Conso annuelle Réf : {k_ref_an:,.0f} kWh/an\n"
               f"📊 Facture annuelle Réf: {f_ref_an:,.2f} €/an\n\n"
               f"💰 ÉCONOMIE BRUTE ÉNERGIE : {gain_an_brut:,.2f} €/an\n"
               f"🔧 Maintenance Annuelle   : -{maint*nb:,.2f} €/an\n"
               f"💶 GAIN NET EXPLOITATION  : {gain_an_net:,.2f} €/an\n"
               f"✨ OPTIMISATION GLOBALE   : -{pct:.1f} %\n"
               f"----------------------------------------\n"
               f"🛠️ Investissement CAPEX   : {i_net:,.2f} € (CEE inclus)\n"
               f"🚀 TEMPS DE RETOUR ROI    : {roi}\n"
               f"🌿 IMPACT RSE (CO2 Évité) : -{(kh*5+ke*7)*0.05:,.0f} kg/an")
        
        self._upd_txt(self.txt_roi, res)
        self.draw(h, r, abo, cop, nb, sec)
        self.btn_export.config(state=NORMAL)

    def draw(self, h, r, abo, cop, nb, sec):
        self.ax1.clear(); self.ax2.clear()
        
        def plot(ax, s, title):
            if not s: ax.axis('off'); return
            k_hp_base, k_hc_base = s['kwh_mois_hp'] * nb, s['kwh_mois_hc'] * nb
            k_eff = k_hp_base * (h / 16.0)
            k_rat = k_eff * r * (1.0 - cop) * (1.0 + sec)
            k_hp_sim, k_hc_sim = k_hp_base - k_eff, k_hc_base + k_rat
            
            f_hp_ap, f_hc_ap = k_hp_sim * s['tarif_hp'], k_hc_sim * s['tarif_hc']
            t_ap, tax_ap = (k_hp_sim + k_hc_sim) * s['tarif_turpe'], (k_hp_sim + k_hc_sim) * s['tarif_taxes']
            tot_ttc_ap = (f_hp_ap + f_hc_ap + t_ap + tax_ap) * 1.20 - (abo * nb)
            
            b_av = [s['f_hc']*nb, s['f_hp']*nb, s['turpe']*nb, s['taxes']*nb, s['tva']*nb]
            b_ap = [f_hc_ap, f_hp_ap, t_ap, tax_ap, max(0, tot_ttc_ap - (tot_ttc_ap/1.2))]
            
            clrs = ['#3498db', '#fd7e14', '#95a5a6', '#adb5bd', '#198754']
            c_av, c_ap = 0, 0
            for i in range(5):
                ax.bar('Actuel', b_av[i], bottom=c_av, color=clrs[i], width=0.5)
                ax.bar('Altileo', b_ap[i], bottom=c_ap, color=clrs[i], width=0.5)
                if b_av[i]>15: ax.text(0, c_av+b_av[i]/2, f"{b_av[i]:.0f}€", ha='center', color='w', fontweight='bold', fontsize=8)
                if b_ap[i]>15: ax.text(1, c_ap+b_ap[i]/2, f"{b_ap[i]:.0f}€", ha='center', color='w', fontweight='bold', fontsize=8)
                c_av += b_av[i]; c_ap += b_ap[i]
                
            ax.set_title(f"Facture {title} (TTC)"); ax.set_ylabel("Euros / Mois")
            g = c_av - c_ap
            if g>0: ax.annotate(f"-{g:.0f}€\n(-{g/c_av*100:.1f}%)", xy=(1, c_ap), xytext=(0, c_av), arrowprops=dict(arrowstyle="->", color="#198754", lw=1.5), color="#198754", fontweight="bold", ha='center', bbox=dict(boxstyle="round", fc="w", ec="#198754"))
            
        plot(self.ax1, self.sum_h if self.sum_h else self.sum_e, "HIVER")
        plot(self.ax2, self.sum_e if self.sum_e else self.sum_h, "ÉTÉ")
        self.fig.tight_layout(); self.canvas.draw()

    def _upd_txt(self, w, t): 
        w.config(state=NORMAL); w.delete(1.0, END); w.insert(END, t); w.config(state=DISABLED)
    
    def export_report(self):
        f = filedialog.asksaveasfilename(defaultextension=".txt", title="Sauvegarder l'Audit")
        if f:
            with open(f, 'w', encoding='utf-8') as doc:
                doc.write("BUSINESS PLAN ALTILEO B2B\n============================\n\n")
                doc.write(self.txt_roi.get(1.0, END))
            self.fig.savefig(f.replace('.txt', '.png'))
            Messagebox.show_info("Rapport et Graphiques exportés !", "Succès")

if __name__ == "__main__":
    app = EnergyApp(EnergyCalculator(SUPABASE_URL, SUPABASE_KEY))
    app.mainloop()