"""Prototype haute fidelite du Dashboard de supervision client Altileo (mode demo).

Ce module ne contient AUCUNE logique metier : c'est un template HTML/CSS/JS autonome
(Tailwind CDN, Google Fonts, Lucide Icons, Chart.js) destine a etre injecte via
st.components.v1.html() dans app_v2.py. Toutes les valeurs affichees sont statiques
et illustratives -- ce prototype n'est pas connecte a Supabase.

Design aligne sur la charte graphique Altileo "Frost & Carbon" (registre carbon :
fond #111111, cartes #181818, bordures #333333, accent teal #00A4B4, highlight
#FFD66B reserve au chiffre-hero, succes/alerte/danger fonctionnels).
"""

# ============================================================
# Donnees statiques du prototype (illustratives, non connectees a Supabase)
# ============================================================

_HOURS = [f"{h:02d}:00" for h in range(25)]

_CONSO_KWH = [
    195, 190, 188, 185, 182, 188, 200, 210,
    95, 80, 75, 85, 190,
    205, 210, 215, 212,
    90, 78, 82, 88,
    200, 195, 190, 195,
]

_STOCKAGE_PCT = [
    65, 63, 62, 64, 68, 75, 85, 92,
    88, 78, 68, 60, 55,
    60, 68, 76, 88,
    84, 74, 64, 57,
    62, 68, 72, 65,
]

_SPARKLINES = {
    "A1": [-18.1, -18.3, -18.6, -18.4, -18.5, -18.7, -18.5, -18.4, -18.5, -18.6, -18.5, -18.5],
    "A2": [-17.6, -17.9, -17.7, -17.8, -18.0, -17.8, -17.7, -17.9, -17.8, -17.8, -17.7, -17.8],
    "B1": [-19.0, -19.3, -19.1, -19.2, -19.4, -19.2, -19.1, -19.3, -19.2, -19.2, -19.1, -19.2],
    "B2": [-17.2, -17.0, -16.8, -16.9, -16.6, -16.9, -17.1, -16.9, -16.7, -16.9, -17.0, -16.9],
}

_ALERTS = [
    ("10:42", "Délestage démarré -- Chambre A1", "info"),
    ("08:15", "Capteur B2 : dérive détectée +0.3 °C", "warning"),
    ("07:58", "Recharge nocturne terminée -- stockage 92 %", "success"),
    ("Hier 23:10", "Fin de cycle -- objectif journalier atteint", "success"),
]

_EQUIPMENT = [
    {"nom": "Chambre A1", "temp": "-18.5 °C", "puissance": "6.2 kW", "statut": "conforme"},
    {"nom": "Chambre A2", "temp": "-17.8 °C", "puissance": "5.9 kW", "statut": "conforme"},
    {"nom": "Chambre B1", "temp": "-19.2 °C", "puissance": "6.5 kW", "statut": "conforme"},
    {"nom": "Chambre B2", "temp": "-16.9 °C", "puissance": "6.1 kW", "statut": "attention"},
]

_STATUT_COLORS = {"conforme": "#2A9D6E", "attention": "#E0A83D", "danger": "#D14343"}


def _equipment_cards_html() -> str:
    cards = []
    for i, eq in enumerate(_EQUIPMENT):
        color = _STATUT_COLORS[eq["statut"]]
        label = "Conforme" if eq["statut"] == "conforme" else "Attention"
        cards.append(f"""
        <div class="bg-[#181818] border border-[#333333] rounded-[12px] p-4 flex flex-col gap-3">
            <div class="flex items-center justify-between">
                <span class="text-sm font-medium text-white">{eq['nom']}</span>
                <span class="flex items-center gap-1.5 text-xs font-medium" style="color:{color}">
                    <span class="w-1.5 h-1.5 rounded-full" style="background:{color}"></span>{label}
                </span>
            </div>
            <div class="flex items-baseline gap-2">
                <span class="font-mono text-2xl font-bold text-white tabular-nums">{eq['temp']}</span>
            </div>
            <div class="flex items-center justify-between text-xs text-[#999999]">
                <span>Puissance</span>
                <span class="font-mono tabular-nums text-[#cccccc]">{eq['puissance']}</span>
            </div>
            <canvas id="spark-{i}" height="40"></canvas>
        </div>""")
    return "".join(cards)


def _alerts_html() -> str:
    dot_colors = {"info": "#00A4B4", "warning": "#E0A83D", "success": "#2A9D6E"}
    items = []
    for ts, text, kind in _ALERTS:
        items.append(f"""
        <li class="flex items-start gap-2.5 py-2 border-b border-[#262626] last:border-0">
            <span class="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0" style="background:{dot_colors[kind]}"></span>
            <div class="flex flex-col">
                <span class="text-xs text-[#999999] font-mono">{ts}</span>
                <span class="text-sm text-[#e5e5e5]">{text}</span>
            </div>
        </li>""")
    return "".join(items)


def render_client_dashboard_html() -> str:
    """Retourne le document HTML autonome du prototype de dashboard client."""
    equipment_html = _equipment_cards_html()
    alerts_html = _alerts_html()
    sparkline_data = {k: v for k, v in _SPARKLINES.items()}

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  /* Fallback independant de Tailwind : reste lisible meme si le CDN Tailwind est bloque (reseau d'entreprise restrictif). */
  html, body {{ margin:0; padding:0; background:#111111; color:#e5e5e5; font-family:'Inter', -apple-system, sans-serif; }}
  .mono {{ font-family:'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; letter-spacing:-0.02em; }}
  ::-webkit-scrollbar {{ width:8px; height:8px; }}
  ::-webkit-scrollbar-thumb {{ background:#333333; border-radius:6px; }}
  .nav-item {{ display:flex; align-items:center; gap:10px; padding:9px 12px; border-radius:8px; font-size:13px; color:#999999; cursor:pointer; }}
  .nav-item:hover {{ background:#1c1c1c; color:#e5e5e5; }}
  .nav-item.active {{ background:rgba(0,164,180,0.12); color:#00A4B4; font-weight:500; }}
  .nav-item svg {{ width:16px; height:16px; }}
</style>
</head>
<body class="bg-[#111111] text-[#e5e5e5]">

<div class="flex min-h-screen">

  <!-- ============ SIDEBAR ============ -->
  <aside class="w-[220px] flex-shrink-0 bg-[#111111] border-r border-[#262626] flex flex-col justify-between p-4">
    <div>
      <div class="flex items-center gap-1 px-1 mb-6">
        <span class="font-bold text-lg text-white">altileo<span class="text-[#00A4B4]">*</span></span>
        <span class="ml-1 text-[10px] font-medium tracking-wide text-[#00A4B4] bg-[rgba(0,164,180,0.15)] border border-[rgba(0,164,180,0.5)] rounded px-1.5 py-0.5 uppercase">Live</span>
      </div>
      <nav class="flex flex-col gap-1">
        <div class="nav-item active"><i data-lucide="layout-dashboard"></i>Dashboard</div>
        <div class="nav-item"><i data-lucide="thermometer"></i>Température</div>
        <div class="nav-item"><i data-lucide="zap"></i>Consommation</div>
        <div class="nav-item"><i data-lucide="battery-charging"></i>Stockage</div>
        <div class="nav-item"><i data-lucide="snowflake"></i>Chaîne du Froid</div>
        <div class="nav-item"><i data-lucide="cpu"></i>Équipements</div>
        <div class="nav-item"><i data-lucide="bell"></i>Alertes</div>
        <div class="nav-item"><i data-lucide="file-text"></i>Rapports</div>
        <div class="nav-item"><i data-lucide="settings"></i>Paramètres</div>
      </nav>
    </div>
    <div class="bg-[#181818] border border-[#333333] rounded-[12px] p-3 text-xs text-[#999999] flex flex-col gap-1.5">
      <div class="flex justify-between"><span>Site</span><span class="text-[#e5e5e5]">Rungis - Entrepôt A</span></div>
      <div class="flex justify-between"><span>Mode</span><span class="text-[#00A4B4]">Optimisation</span></div>
      <div class="flex justify-between items-center"><span>Statut</span>
        <span class="flex items-center gap-1.5 text-[#2A9D6E]"><span class="w-1.5 h-1.5 rounded-full bg-[#2A9D6E]"></span>Opérationnel</span>
      </div>
    </div>
  </aside>

  <!-- ============ MAIN ============ -->
  <main class="flex-1 p-6 flex flex-col gap-6 min-w-0">

    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-bold text-white">Tableau de bord</h1>
        <p class="text-sm text-[#999999] mt-0.5">Pilotage énergétique intelligent</p>
      </div>
      <div class="text-right">
        <div id="live-date" class="text-sm text-[#e5e5e5]"></div>
        <div id="live-time" class="mono text-lg font-bold text-white"></div>
      </div>
    </div>

    <!-- KPI cards -->
    <div class="grid grid-cols-4 gap-4">
      <div class="bg-[#181818] border border-[#333333] rounded-[12px] p-4 flex flex-col gap-2">
        <div class="flex items-center justify-between text-xs text-[#999999] uppercase tracking-wide"><span>Consommation Actuelle</span><i data-lucide="zap" class="w-4 h-4 text-[#00A4B4]"></i></div>
        <div class="mono text-3xl font-bold text-white">245<span class="text-base font-medium text-[#999999] ml-1">kWh</span></div>
        <div class="text-xs font-medium text-[#2A9D6E] flex items-center gap-1"><i data-lucide="arrow-down" class="w-3 h-3"></i>-12.5 % vs hier</div>
      </div>
      <div class="bg-[#181818] border border-[#333333] rounded-[12px] p-4 flex flex-col gap-2">
        <div class="flex items-center justify-between text-xs text-[#999999] uppercase tracking-wide"><span>Stockage Énergétique</span><i data-lucide="battery-charging" class="w-4 h-4 text-[#00A4B4]"></i></div>
        <div class="mono text-3xl font-bold text-white">72,4<span class="text-base font-medium text-[#999999] ml-1">%</span></div>
        <div class="text-xs font-medium text-[#2A9D6E] flex items-center gap-1"><i data-lucide="arrow-up" class="w-3 h-3"></i>+18,2 % vs hier</div>
      </div>
      <div class="bg-[#181818] border border-[#333333] rounded-[12px] p-4 flex flex-col gap-2">
        <div class="flex items-center justify-between text-xs text-[#999999] uppercase tracking-wide"><span>Température Moyenne</span><i data-lucide="thermometer" class="w-4 h-4 text-[#00A4B4]"></i></div>
        <div class="mono text-3xl font-bold text-white">-18,2<span class="text-base font-medium text-[#999999] ml-1">°C</span></div>
        <div class="text-xs font-medium text-[#E0A83D] flex items-center gap-1"><i data-lucide="arrow-up" class="w-3 h-3"></i>+0,8 °C vs hier</div>
      </div>
      <div class="bg-[#181818] border border-[#333333] rounded-[12px] p-4 flex flex-col gap-2">
        <div class="flex items-center justify-between text-xs text-[#999999] uppercase tracking-wide"><span>Économies Réalisées</span><i data-lucide="euro" class="w-4 h-4 text-[#FFD66B]"></i></div>
        <div class="mono text-3xl font-bold text-[#FFD66B]">1 847<span class="text-base font-medium text-[#999999] ml-1">€</span></div>
        <div class="text-xs font-medium text-[#2A9D6E] flex items-center gap-1"><i data-lucide="arrow-up" class="w-3 h-3"></i>+15,3 % vs mois dernier</div>
      </div>
    </div>

    <!-- Chart + status -->
    <div class="grid grid-cols-3 gap-6 flex-1 min-h-0">

      <div class="col-span-2 bg-[#181818] border border-[#333333] rounded-[12px] p-5 flex flex-col gap-4">
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium text-white">Consommation &amp; stockage thermique -- 24 h</span>
          <div class="flex items-center gap-4 text-xs text-[#999999]">
            <span class="flex items-center gap-1.5"><span class="w-2.5 h-0.5 bg-[#00A4B4]"></span>Consommation (kWh)</span>
            <span class="flex items-center gap-1.5"><span class="w-2.5 h-0.5 bg-[#FFD66B]"></span>Stockage thermique (%)</span>
          </div>
        </div>
        <div class="flex-1 min-h-[260px]"><canvas id="mainChart"></canvas></div>
        <div class="grid grid-cols-4 gap-3 border-t border-[#262626] pt-4">
          <div class="flex flex-col gap-0.5"><span class="text-[11px] text-[#999999] uppercase tracking-wide">Consom. Moyenne</span><span class="mono text-sm font-bold text-white">186 kWh</span></div>
          <div class="flex flex-col gap-0.5"><span class="text-[11px] text-[#999999] uppercase tracking-wide">Consom. Prévue</span><span class="mono text-sm font-bold text-white">210 kWh</span></div>
          <div class="flex flex-col gap-0.5"><span class="text-[11px] text-[#999999] uppercase tracking-wide">Objectif Journalier</span><span class="mono text-sm font-bold text-white">200 kWh</span></div>
          <div class="flex flex-col gap-0.5"><span class="text-[11px] text-[#999999] uppercase tracking-wide">CO₂ Évité</span><span class="mono text-sm font-bold text-[#2A9D6E]">384 kg</span></div>
        </div>
      </div>

      <div class="flex flex-col gap-4 min-h-0">
        <div class="bg-[#181818] border border-[#333333] rounded-[12px] p-4 flex flex-col gap-3">
          <span class="text-sm font-medium text-white">Statut système</span>
          <ul class="flex flex-col gap-2.5 text-sm">
            <li class="flex items-center justify-between"><span class="text-[#cccccc]">IA Optimisation</span><span class="flex items-center gap-1.5 text-[#2A9D6E] text-xs font-medium"><span class="w-1.5 h-1.5 rounded-full bg-[#2A9D6E]"></span>Actif</span></li>
            <li class="flex items-center justify-between"><span class="text-[#cccccc]">Capteurs</span><span class="flex items-center gap-1.5 text-[#2A9D6E] text-xs font-medium"><span class="w-1.5 h-1.5 rounded-full bg-[#2A9D6E]"></span>OK</span></li>
            <li class="flex items-center justify-between"><span class="text-[#cccccc]">Réseau</span><span class="flex items-center gap-1.5 text-[#2A9D6E] text-xs font-medium"><span class="w-1.5 h-1.5 rounded-full bg-[#2A9D6E]"></span>Stable</span></li>
            <li class="flex items-center justify-between"><span class="text-[#cccccc]">Stockage</span><span class="flex items-center gap-1.5 text-[#E0A83D] text-xs font-medium"><span class="w-1.5 h-1.5 rounded-full bg-[#E0A83D]"></span>Maintenance</span></li>
          </ul>
          <button class="mt-1 w-full bg-[#00A4B4] hover:bg-[#0E7C8C] text-white text-xs font-medium rounded-[6px] py-2 transition-colors">Voir le diagnostic</button>
        </div>

        <div class="bg-[#181818] border border-[#333333] rounded-[12px] p-4 flex flex-col gap-1 flex-1 min-h-0 overflow-y-auto">
          <span class="text-sm font-medium text-white mb-1">Alertes récentes</span>
          <ul>{alerts_html}</ul>
        </div>
      </div>
    </div>

    <!-- Equipment grid -->
    <div>
      <span class="text-sm font-medium text-white block mb-3">Chambres froides</span>
      <div class="grid grid-cols-4 gap-4">{equipment_html}</div>
    </div>

  </main>
</div>

<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script src="https://unpkg.com/lucide@latest"></script>
<script>
  lucide.createIcons();

  function updateClock() {{
    const now = new Date();
    const dateOpts = {{ weekday:'long', day:'numeric', month:'long', year:'numeric' }};
    document.getElementById('live-date').textContent = now.toLocaleDateString('fr-FR', dateOpts);
    document.getElementById('live-time').textContent = now.toLocaleTimeString('fr-FR');
  }}
  updateClock();
  setInterval(updateClock, 1000);

  const ctx = document.getElementById('mainChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: {_HOURS},
      datasets: [
        {{
          label: 'Consommation (kWh)',
          data: {_CONSO_KWH},
          borderColor: '#00A4B4',
          backgroundColor: 'rgba(0,164,180,0.08)',
          borderWidth: 2,
          tension: 0.35,
          fill: true,
          pointRadius: 0,
          yAxisID: 'y',
        }},
        {{
          label: 'Stockage thermique (%)',
          data: {_STOCKAGE_PCT},
          borderColor: '#FFD66B',
          backgroundColor: 'rgba(255,214,107,0.06)',
          borderWidth: 2,
          borderDash: [4,3],
          tension: 0.35,
          fill: false,
          pointRadius: 0,
          yAxisID: 'y1',
        }},
      ],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#111111',
          borderColor: '#333333',
          borderWidth: 1,
          titleColor: '#ffffff',
          bodyColor: '#cccccc',
          bodyFont: {{ family: "'JetBrains Mono', monospace" }},
        }},
      }},
      scales: {{
        x: {{
          grid: {{ color: '#262626' }},
          ticks: {{ color: '#999999', maxTicksLimit: 12, font: {{ family: "'JetBrains Mono', monospace", size: 10 }} }},
        }},
        y: {{
          position: 'left',
          grid: {{ color: '#262626' }},
          ticks: {{ color: '#00A4B4', font: {{ family: "'JetBrains Mono', monospace", size: 10 }} }},
          title: {{ display: true, text: 'kWh', color: '#999999' }},
        }},
        y1: {{
          position: 'right',
          grid: {{ display: false }},
          ticks: {{ color: '#FFD66B', font: {{ family: "'JetBrains Mono', monospace", size: 10 }} }},
          title: {{ display: true, text: '%', color: '#999999' }},
          min: 0, max: 100,
        }},
      }},
    }},
  }});

  const sparklineData = {sparkline_data};
  const sparklineColors = {{0:'#2A9D6E',1:'#2A9D6E',2:'#2A9D6E',3:'#E0A83D'}};
  Object.keys(sparklineData).forEach((key, i) => {{
    const el = document.getElementById('spark-' + i);
    if (!el) return;
    new Chart(el.getContext('2d'), {{
      type: 'line',
      data: {{
        labels: sparklineData[key].map((_, idx) => idx),
        datasets: [{{
          data: sparklineData[key],
          borderColor: sparklineColors[i] || '#00A4B4',
          borderWidth: 1.5,
          tension: 0.4,
          fill: false,
          pointRadius: 0,
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display:false }}, tooltip: {{ enabled:false }} }},
        scales: {{ x: {{ display:false }}, y: {{ display:false }} }},
        elements: {{ line: {{ borderJoinStyle:'round' }} }},
      }},
    }});
  }});
</script>
</body>
</html>"""
