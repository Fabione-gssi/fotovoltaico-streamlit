import math
from dataclasses import dataclass
import streamlit as st
import numpy as np

# -------------------------------------------------------------
# App: Calcolatore fotovoltaico avanzato (Terlizzi, BA)
# Obiettivo: esperienza da "ingegnere virtuale" con 4 aree + simulazione oraria
# -------------------------------------------------------------

st.set_page_config(page_title="Progettazione FV 3 kWp - Ingegnerizzato", layout="wide")
st.title("⚙️ Progettazione impianto fotovoltaico – Terlizzi (BA)")
st.caption("Toolkit interattivo per pre-dimensionamento tecnico con confronto Off-grid vs Grid-connected")

# ---------- Utils ----------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def orientation_factor(tilt_deg: float, azimuth_deg: float) -> float:
    """
    Fattore molto semplificato vs Sud 30°. Azimuth: 0° = Sud, +90° = Ovest, -90° = Est.
    Ritorna ~1.0 vicino a 30°/Sud, ~0.85 Est/Ovest, mai sotto 0.75.
    """
    # Tilt penalty (parabola blanda centrata a 30°)
    f_tilt = 1.0 - 0.0015 * (tilt_deg - 30.0) ** 2
    f_tilt = clamp(f_tilt, 0.88, 1.0)
    # Azimuth penalty (curva morbida)
    az = abs(azimuth_deg)
    f_az = 0.95 - 0.15 * (az / 90.0) ** 1.3
    f_az = clamp(f_az, 0.75, 1.0)
    return f_tilt * f_az

# Month metadata for Puglia-like conditions (approssimazioni)
MONTHS = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# Ripartizione % della produzione annua (somma = 1.0)
MONTH_SHARE = np.array([0.05, 0.06, 0.09, 0.11, 0.12, 0.13, 0.13, 0.11, 0.09, 0.06, 0.04, 0.04])
# Ore di luce medie (41°N, indicativo)
DAYLIGHT_H = [9.6, 10.8, 12.0, 13.2, 14.3, 15.0, 14.6, 13.5, 12.4, 11.2, 10.1, 9.5]

# ---------- Sidebar: Parametri globali ----------
st.sidebar.header("Parametri globali")

localita = st.sidebar.text_input("Località", value="Terlizzi (BA)")
potenza_kwp = st.sidebar.number_input("Potenza impianto (kWp)", 1.0, 20.0, 3.0, 0.1)
# Baseline Puglia a tilt ottimale sud ~1500 kWh/kWp/anno
baseline_kwh_per_kwp = st.sidebar.number_input("Irraggiamento base (kWh/kWp·anno)", 900, 2000, 1500)
system_losses_pct = st.sidebar.slider("Perdite di sistema complessive (%)", 5, 25, 14,
                                      help="Cavi, mismatch, inverter, polvere, temperatura (valore unico semplificato)")
shading_pct = st.sidebar.slider("Fattore ombreggiamento (%)", 0, 30, 5)

# Geometria
st.sidebar.subheader("Geometria tetto")
tilt = st.sidebar.slider("Inclinazione falda (°)", 0, 60, 30)
azimuth = st.sidebar.slider("Azimuth (° da Sud; Est=-90, Ovest=+90)", -180, 180, 0)
superficie_m2 = st.sidebar.number_input("Superficie utile disponibile (m²)", 5.0, 200.0, 20.0, 0.5)

# Economici generali
st.sidebar.subheader("Economia")
costo_moduli_inverter_eur_per_wp = st.sidebar.number_input("Costo moduli+inverter (€/Wp)", 0.5, 3.0, 1.2, 0.05)
costo_installazione_misc = st.sidebar.number_input("Installazione & minuteria (€)", 0, 10000, 1500, 100)

# Scenari
scenario = st.sidebar.radio("Scenario principale", ["Off-grid", "Grid-connected"], index=0)

# Battery global settings
st.sidebar.subheader("Batteria – impostazioni globali")
roundtrip_eff = st.sidebar.slider("Efficienza round-trip batteria (%)", 80, 98, 92)
usable_dod_pct = st.sidebar.slider("Profondità di scarica utilizzabile DOD (%)", 50, 100, 90)

# -------------------------------------------------------------
# Tab principali
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1) Impianto & Geometria",
    "2) Componenti",
    "3) Simulazione oraria",
    "4) Economia",
    "5) Burocrazia & BOM",
    "6) Report"
])

# -------------------------------------------------------------
# 1) Impianto & Geometria
# -------------------------------------------------------------
with tab1:
    st.subheader("Geometria e resa attesa")
    f_orient = orientation_factor(tilt, azimuth)
    losses = (1 - system_losses_pct/100.0) * (1 - shading_pct/100.0)
    produzione_annua_kwh = potenza_kwp * baseline_kwh_per_kwp * f_orient * losses

    st.metric("Fattore orientamento", f"{f_orient:.2f}×")
    st.metric("Produzione annua stimata", f"{produzione_annua_kwh:,.0f} kWh/anno")

    st.markdown("---")
    st.markdown("### Area e densità di potenza")
    eff_impianto_pct = st.slider("Efficienza moduli (indicativa, %)", 16, 23, 20,
                                 help="Solo per stima area: 1 kWp richiede circa 5–6 m² a queste efficienze")
    area_per_kwp = 10.0 / eff_impianto_pct  # m² per kWp (~ 10 / eff%)
    area_richiesta = potenza_kwp * area_per_kwp

    st.write(f"**Area stimata necessaria:** {area_richiesta:.1f} m²  ")
    if area_richiesta > superficie_m2:
        st.warning("L'area disponibile NON è sufficiente per la potenza impostata. Riduci kWp o scegli moduli più efficienti.")
    else:
        st.success("L'area disponibile è sufficiente.")

    st.info("Nota: i calcoli qui sono semplificati ma conservativi. La produzione reale dipende anche da temperatura cella, soiling, albedo.")

# -------------------------------------------------------------
# 2) Componenti – libreria semplificata + dimensionamento stringhe e cavi
# -------------------------------------------------------------
@dataclass
class Module:
    model: str
    Wp: float
    Vmpp: float
    Impp: float
    Voc: float
    Isc: float
    length_m: float
    width_m: float
    price_per_W: float

@dataclass
class Inverter:
    model: str
    ac_kw: float
    type: str  # "hybrid" | "string"
    mppt_count: int
    mppt_v_min: float
    mppt_v_max: float
    mppt_i_max: float
    pv_kw_max: float

@dataclass
class Battery:
    model: str
    chemistry: str
    kwh: float
    max_c_rate: float  # in C (continua)
    price_per_kwh: float

MODULES = [
    Module("Mono 400", 400, 34.0, 11.8, 41.0, 12.6, 1.72, 1.13, 0.35),
    Module("Mono 430", 430, 33.5, 12.8, 41.5, 13.3, 1.75, 1.14, 0.38),
    Module("Mono 450 HJT", 450, 34.5, 13.0, 42.5, 13.6, 1.76, 1.14, 0.42),
]

INVERTERS = [
    Inverter("Ibrido 3 kW 2xMPPT", 3.0, "hybrid", 2, 120, 500, 14.0, 4.0),
    Inverter("Ibrido 5 kW 2xMPPT", 5.0, "hybrid", 2, 120, 550, 14.0, 7.0),
    Inverter("String 3 kW 2xMPPT", 3.0, "string", 2, 125, 500, 12.0, 4.0),
]

BATTERIES = [
    Battery("LiFePO₄ rack 5 kWh", "LiFePO4", 5.0, 0.5, 450),
    Battery("LiFePO₄ rack 10 kWh", "LiFePO4", 10.0, 0.5, 430),
    Battery("LFP Premium 5 kWh", "LiFePO4", 5.0, 1.0, 520),
]

with tab2:
    st.subheader("Selezione componenti reali (esempio semplificato)")

    colA, colB, colC = st.columns(3)
    with colA:
        mod_sel = st.selectbox("Modulo fotovoltaico", MODULES, format_func=lambda m: f"{m.model} – {m.Wp:.0f} Wp")
    with colB:
        inv_sel = st.selectbox("Inverter", INVERTERS, format_func=lambda i: f"{i.model} ({i.type}) – {i.ac_kw:.1f} kW AC")
    with colC:
        bat_sel = st.selectbox("Batteria (unità base)", BATTERIES, format_func=lambda b: f"{b.model} – {b.kwh:.0f} kWh")

    # Numero moduli richiesti
    n_mod_tot = math.ceil(potenza_kwp * 1000.0 / mod_sel.Wp)

    # String sizing – assumo Tmin = -10°C a Terlizzi
    t_min = st.number_input("Temperatura minima progetto (°C)", -30.0, 20.0, -10.0, 1.0)
    beta_voc = -0.003  # -0.3%/°C
    voc_cold = mod_sel.Voc * (1 - beta_voc * (t_min - 25.0))  # maggiore al freddo

    # Range ammesso in MPPT per N moduli in serie
    def series_ok(n_series: int) -> bool:
        vmp = n_series * mod_sel.Vmpp
        voc = n_series * voc_cold
        return (inv_sel.mppt_v_min <= vmp <= inv_sel.mppt_v_max) and (voc < inv_sel.mppt_v_max * 1.05)

    # Cerca numero ottimale moduli per stringa
    candidates = [n for n in range(4, 20) if series_ok(n)]
    if not candidates:
        st.error("Nessuna combinazione di moduli per stringa rientra nella finestra MPPT/Inverter. Modifica modello o inverter.")
        n_series = 0
    else:
        n_series = st.select_slider("Moduli per stringa (validi)", options=candidates, value=candidates[len(candidates)//2])

    n_strings = math.ceil(n_mod_tot / max(1, n_series))
    if n_strings > inv_sel.mppt_count:
        st.warning(f"Servono {n_strings} stringhe; l'inverter ha {inv_sel.mppt_count} MPPT. Possibile parallelo su stesso MPPT se correnti lo consentono.")

    st.write(f"**Moduli totali:** {n_mod_tot}  ")
    st.write(f"**Stringhe:** {n_strings} × {n_series} moduli (tot {n_strings*n_series} – potrebbe esserci 1 modulo in più/meno per chiudere)")

    # Correnti e potenze
    string_power_wp = n_series * mod_sel.Wp
    string_impp = mod_sel.Impp  # in serie la corrente resta Impp del modulo
    string_isc = mod_sel.Isc
    st.write(f"**Potenza per stringa:** ~{string_power_wp:.0f} Wp  | **I_mpp:** {string_impp:.1f} A  | **I_sc:** {string_isc:.1f} A")

    # Cavi DC – scelta sezione basata su caduta di tensione
    st.markdown("### Cablaggi DC")
    dc_len = st.number_input("Lunghezza tratta DC (solo andata, m)", 1.0, 100.0, 15.0, 0.5)
    vmp_string = n_series * mod_sel.Vmpp
    drop_pct = st.slider("Caduta di tensione ammessa DC (%)", 1.0, 5.0, 2.0, 0.5)
    delta_v = vmp_string * drop_pct/100.0
    resistivity_cu = 0.0175  # ohm·mm²/m
    # S(mm²) = (2 * L * I * rho) / ΔV
    S_mm2 = (2 * dc_len * string_impp * resistivity_cu) / max(0.1, delta_v)
    std_sizes = [4, 6, 10, 16, 25]
    dc_section = next(s for s in std_sizes if s >= S_mm2)
    st.write(f"Sezione minima calcolata: {S_mm2:.1f} mm² → **Consigliata standard: {dc_section} mm²**")

    st.markdown("### Cablaggi AC")
    ac_len = st.number_input("Lunghezza tratta AC (m)", 1.0, 100.0, 20.0, 0.5)
    vac = 230.0
    i_ac = (inv_sel.ac_kw * 1000.0) / vac
    drop_ac_pct = st.slider("Caduta di tensione ammessa AC (%)", 1.0, 5.0, 2.0, 0.5)
    delta_v_ac = vac * drop_ac_pct/100.0
    S_ac = (2 * ac_len * i_ac * resistivity_cu) / max(0.1, delta_v_ac)
    ac_section = next(s for s in std_sizes if s >= S_ac)
    st.write(f"Corrente AC: {i_ac:.1f} A – Sezione minima: {S_ac:.1f} mm² → **Consigliata: {ac_section} mm²**")

    # Costi componenti
    costo_mod_inv = potenza_kwp * 1000.0 * costo_moduli_inverter_eur_per_wp

    st.markdown("### Stima costi parziali")
    st.write(f"**Moduli + inverter:** {costo_mod_inv:,.0f} €  ")

# -------------------------------------------------------------
# 3) Simulazione oraria – profilo PV vs carichi + batteria
# -------------------------------------------------------------
with tab3:
    st.subheader("Simulazione oraria di una giornata tipo")
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        month_idx = st.select_slider("Mese rappresentativo", options=list(range(12)), value=6, format_func=lambda i: MONTHS[i])
        daylight = DAYLIGHT_H[month_idx]
        share = MONTH_SHARE[month_idx]
        daily_pv_kwh = produzione_annua_kwh * share / DAYS_IN_MONTH[month_idx]

        # Profilo PV idealizzato a campana
        hours = np.arange(24)
        pv_kw = np.zeros(24)
        start = max(0, int(round(12 - daylight/2)))
        end = min(24, int(round(12 + daylight/2)))
        if end > start:
            x = np.linspace(0, math.pi, end - start)
            shape = np.sin(x)  # 0..pi → campana
            # Scala a energia giornaliera
            energy_norm = shape.sum()  # somma valori unitari
            # potenza media per step da 1h: daily_kwh ≈ sum(pv_kw)
            scale = daily_pv_kwh / max(0.1, energy_norm)
            pv_kw[start:end] = shape * scale

        st.markdown("### Profilo consumi")
        preset = st.selectbox("Preset consumi", ["Residenziale (serale)", "Residenziale (diurno)", "Ufficio", "Personalizzato"])
        base_kw = st.slider("Carico base (kW)", 0.05, 1.5, 0.2, 0.05)
        load_kw = np.full(24, base_kw)
        if preset == "Residenziale (serale)":
            load_kw[7:9] += 0.5
            load_kw[19:22] += 1.2
        elif preset == "Residenziale (diurno)":
            load_kw[10:16] += 0.8
            load_kw[19:21] += 0.4
        elif preset == "Ufficio":
            load_kw[9:18] += 1.5
        else:
            # Personalizzato: tre blocchi
            st.info("Configura 3 blocchi opzionali di carico extra")
            for i in range(1, 4):
                c = st.slider(f"Blocco {i} – potenza (kW)", 0.0, 3.0, 0.0, 0.1)
                start_h = st.number_input(f"Blocco {i} – inizio (h)", 0, 23, 18)
                durata = st.number_input(f"Blocco {i} – durata (h)", 0, 24, 2)
                load_kw[start_h:start_h+durata] += c

        daily_load_kwh = float(load_kw.sum())

    with col_right:
        st.markdown("### Batteria e limiti di potenza")
        bat_dimensioning = st.radio("Modalità dimensionamento batteria", ["Per giorni di autonomia", "Manuale"], index=0)
        if bat_dimensioning == "Per giorni di autonomia":
            days_autonomy = st.slider("Giorni di autonomia target", 1, 5, 2)
            bat_kwh_nominal = days_autonomy * daily_load_kwh
        else:
            bat_kwh_nominal = st.number_input("Capacità nominale batteria (kWh)", 1.0, 40.0, 10.0, 0.5)

        usable_frac = usable_dod_pct/100.0
        bat_kwh_usable = bat_kwh_nominal * usable_frac
        inv_ac_limit_kw = st.slider("Limite potenza inverter AC (kW)", 1.0, 15.0, value=float(INVERTERS[0].ac_kw), step=0.5)
        bat_c_rate = st.slider("C-rate max batteria (C)", 0.2, 1.5, 0.5, 0.1)
        bat_max_chg_kw = bat_kwh_nominal * bat_c_rate
        bat_max_dchg_kw = bat_kwh_nominal * bat_c_rate

        st.write(f"**Batteria utilizzabile:** {bat_kwh_usable:.1f} kWh  ")
        st.write(f"**Limiti potenza:** carica {bat_max_chg_kw:.1f} kW / scarica {bat_max_dchg_kw:.1f} kW / inverter {inv_ac_limit_kw:.1f} kW")

    # Simulazione SOC oraria
    eta = roundtrip_eff/100.0
    soc = np.zeros(24)
    soc[0] = 0.5 * bat_kwh_usable  # partenza a 50% utilizzabile
    grid_import_kwh = 0.0
    unmet_kwh = 0.0
    curtailed_kwh = 0.0

    for h in range(24):
        gen = min(pv_kw[h], inv_ac_limit_kw)  # limite inverter AC
        load = load_kw[h]
        surplus = gen - load
        if surplus >= 0:
            # carica batteria con efficienza
            can_charge = min(bat_max_chg_kw, surplus)
            energy_to_store = can_charge * eta
            room = bat_kwh_usable - soc[h]
            charge = min(energy_to_store, room)
            actual_input_from_pv = charge / max(1e-6, eta)
            curtail = surplus - actual_input_from_pv
            curtailed_kwh += max(0.0, curtail)
            next_soc = soc[h] + charge
        else:
            # deficit: scarico batteria
            need = -surplus
            discharge = min(bat_max_dchg_kw, need)
            discharge_available = min(discharge, soc[h])
            # energia utile al carico tenendo conto efficienza
            served = discharge_available * eta
            residual_need = need - served
            if scenario == "Grid-connected":
                grid_import_kwh += max(0.0, residual_need)
                residual_need = 0.0
            else:
                unmet_kwh += max(0.0, residual_need)
            next_soc = soc[h] - discharge_available
        if h < 23:
            soc[h+1] = clamp(next_soc, 0.0, bat_kwh_usable)

    served_load_kwh = float(load_kw.sum() - unmet_kwh)

    # Grafici
    import matplotlib.pyplot as plt

    fig1 = plt.figure(figsize=(8,3))
    plt.plot(hours, pv_kw, label="FV (kW)")
    plt.plot(hours, load_kw, label="Carico (kW)")
    plt.xlabel("Ora")
    plt.ylabel("Potenza [kW]")
    plt.title("Produzione FV vs Carico – giornata tipo")
    plt.legend()
    st.pyplot(fig1)

    fig2 = plt.figure(figsize=(8,3))
    plt.step(hours, soc, where='mid')
    plt.xlabel("Ora")
    plt.ylabel("Energia in batteria [kWh]")
    plt.title("Stato di carica (SOC) – energia utilizzabile")
    st.pyplot(fig2)

    st.markdown("### Bilancio giornaliero")
    st.write(f"Produzione FV: **{pv_kw.sum():.1f} kWh**  | Carico: **{daily_load_kwh:.1f} kWh**  | Energia servita: **{served_load_kwh:.1f} kWh**")
    if scenario == "Grid-connected":
        st.write(f"Import da rete: **{grid_import_kwh:.1f} kWh**  | Curtailment: **{curtailed_kwh:.1f} kWh**")
    else:
        st.write(f"Non servito (blackout potenziale): **{unmet_kwh:.1f} kWh**  | Curtailment: **{curtailed_kwh:.1f} kWh**")

# -------------------------------------------------------------
# 4) Economia – LCOE e payback
# -------------------------------------------------------------
with tab4:
    st.subheader("Economia del sistema")

    # CAPEX
    capex_pv = potenza_kwp * 1000.0 * costo_moduli_inverter_eur_per_wp
    bat_units = st.number_input("Numero unità batteria selezionata (tab Componenti)", 0, 10, 2)
    bat_total_kwh = bat_units * bat_sel.kwh
    capex_batt = bat_units * bat_sel.kwh * bat_sel.price_per_kwh
    capex_total = capex_pv + capex_batt + costo_installazione_misc

    st.write(f"**CAPEX PV:** {capex_pv:,.0f} €  | **CAPEX Batterie:** {capex_batt:,.0f} €  | **Installazione:** {costo_installazione_misc:,.0f} €")
    st.write(f"**CAPEX Totale:** {capex_total:,.0f} €")

    # Parametri finanziari
    years = st.slider("Anni di analisi", 5, 30, 20)
    rate = st.slider("Tasso di sconto (%)", 0.0, 12.0, 5.0, 0.5)/100.0
    o_and_m_pct = st.slider("O&M annuo (% CAPEX)", 0.0, 5.0, 1.5, 0.1)/100.0

    # Energia annua servita (stima): produzione annua limitata da carichi e batteria.
    # Usiamo un fattore di utilizzo basato sulla simulazione giornaliera attuale → scala a 365
    served_factor = 1.0 if scenario == "Grid-connected" else clamp((served_load_kwh / max(0.1, daily_load_kwh)), 0.0, 1.0)
    energy_served_year = produzione_annua_kwh * (0.9 if scenario == "Grid-connected" else served_factor)  # 0.9: esporti/sprechi un po'

    # CRF e LCOE
    if rate == 0:
        crf = 1.0/years
    else:
        crf = (rate * (1+rate)**years) / ((1+rate)**years - 1)
    annualized_capex = crf * capex_total
    o_and_m = o_and_m_pct * capex_total
    lcoe = (annualized_capex + o_and_m) / max(1.0, energy_served_year)

    st.write(f"**Energia servita annua stimata:** {energy_served_year:,.0f} kWh/anno")
    st.write(f"**LCOE (costo livellato):** {lcoe*100:.1f} c€/kWh")

    if scenario == "Grid-connected":
        prezzo_rete = st.slider("Prezzo energia da rete (€/kWh)", 0.10, 0.50, 0.28, 0.01)
        risparmio_annuo = prezzo_rete * min(produzione_annua_kwh, energy_served_year)
        payback = capex_total / max(1.0, risparmio_annuo)
        st.write(f"**Risparmio annuo stimato:** {risparmio_annuo:,.0f} € → **Payback semplice:** {payback:.1f} anni")
    else:
        costo_gen = st.slider("Costo alternativo (generatore diesel, €/kWh)", 0.30, 1.50, 0.70, 0.05,
                              help="Costo totale kWh tra carburante, manutenzione, ammortamento generatore")
        saving_vs_gen = max(0.0, (costo_gen - lcoe) * energy_served_year)
        st.write(f"**Vantaggio vs generatore:** {saving_vs_gen:,.0f} €/anno (stimato)")

    st.info("Le stime economiche sono indicative e non sostituiscono un business plan. Aggiorna i parametri per scenari diversi.")

# -------------------------------------------------------------
# 5) Burocrazia & BOM (distinta base)
# -------------------------------------------------------------
with tab5:
    st.subheader("Burocrazia – panoramica")
    if scenario == "Off-grid":
        st.markdown("""
        **Off-grid (isolato)** – in genere:
        - Nessuna pratica con **GSE** o distributore.
        - **Verifica urbanistica/edilizia locale**: anche senza allaccio, strutture su tetto/terreno possono richiedere comunicazione o titolo edilizio (es. CILA/SCIA) a seconda di vincoli paesaggistici e tipologia.
        - Rispetto **CEI** e norme di sicurezza: sezionamento DC, SPD Tipo II lato DC/AC, messa a terra, protezioni magnetotermiche/differenziali lato AC dove serve.
        - Manuale d’impianto e schema elettrico aggiornato.
        """)
    else:
        st.markdown("""
        **Grid-connected** – in sintesi:
        - Pratica di connessione con il **distributore** (es. e-distribuzione) e dispositivo di **interfaccia** conforme.
        - Gestione con **GSE** (scambio sul posto o altre forme secondo normativa vigente).
        - Eventuali **titoli edilizi** e rispetto norme CEI 0-21 / 0-16.
        - Collaudo e dichiarazioni di conformità.
        """)

    st.subheader("Distinta base minima (indicativa)")
    bom = [
        f"Moduli FV: {n_mod_tot} × {mod_sel.model} ({mod_sel.Wp:.0f} Wp)",
        f"Inverter: {inv_sel.model} ({inv_sel.ac_kw:.1f} kW, {inv_sel.mppt_count} MPPT)",
        f"Batterie: {bat_units} × {bat_sel.model} ({bat_sel.kwh:.0f} kWh ciascuna)",
        f"Cavi DC {dc_section} mm² (≈ {int(dc_len)} m) + connettori MC4",
        f"Cavi AC {ac_section} mm² (≈ {int(ac_len)} m)",
        "Sezionatore DC, fusibili per stringa se in parallelo, SPD Tipo II lato DC/AC",
        "Quadro elettrico, interruttori, messa a terra, struttura di fissaggio idonea",
    ]
    st.write("\n".join([f"- {item}" for item in bom]))

# -------------------------------------------------------------
# 6) Report – esportazione sintetica
# -------------------------------------------------------------
with tab6:
    st.subheader("Esporta report tecnico (Markdown)")
    report = []
    report.append(f"# Report FV – {localita}")
    report.append("\n## Dati di input")
    report.append(f"Potenza: {potenza_kwp} kWp | Baseline: {baseline_kwh_per_kwp} kWh/kWp·anno | Perdite: {system_losses_pct}% | Ombre: {shading_pct}%")
    report.append(f"Tilt: {tilt}° | Azimuth: {azimuth}° | Superficie: {superficie_m2} m² | Scenario: {scenario}")
    report.append("\n## Produzione attesa")
    report.append(f"Produzione annua stimata: {produzione_annua_kwh:.0f} kWh/anno (fattore orientamento {f_orient:.2f})")
    report.append("\n## Componenti")
    report.append(f"Moduli: {n_mod_tot} × {mod_sel.model} – {mod_sel.Wp:.0f} Wp | Inverter: {inv_sel.model}")
    report.append(f"Stringhe: {n_strings} × {n_series} moduli | Vmp stringa ≈ {vmp_string:.0f} V | I_mpp ≈ {string_impp:.1f} A")
    report.append(f"Cavi DC consigliati: {dc_section} mm² | Cavi AC consigliati: {ac_section} mm²")
    report.append("\n## Simulazione oraria (giornata tipo)")
    report.append(f"Produzione: {pv_kw.sum():.1f} kWh | Carico: {daily_load_kwh:.1f} kWh | Servito: {served_load_kwh:.1f} kWh")
    if scenario == "Grid-connected":
        report.append(f"Import rete: {grid_import_kwh:.1f} kWh | Curtailment: {curtailed_kwh:.1f} kWh")
    else:
        report.append(f"Non servito: {unmet_kwh:.1f} kWh | Curtailment: {curtailed_kwh:.1f} kWh")
    report.append("\n## Economia")
    report.append(f"CAPEX Totale: {capex_total:,.0f} € | LCOE: {lcoe*100:.1f} c€/kWh | Orizzonte: {years} anni")
    if scenario == "Grid-connected":
        report.append(f"Payback semplice: {payback:.1f} anni")
    else:
        report.append(f"Vantaggio vs generatore: {saving_vs_gen:,.0f} €/anno")
    report.append("\n## Burocrazia (nota)")
    report.append("Vedi tab dedicata per checklist indicativa. Verificare sempre requisiti locali e norme CEI applicabili.")

    md = "\n".join(report)
    st.download_button("⬇️ Scarica report.md", data=md, file_name="report_fv_terlizzi.md", mime="text/markdown")

# -------------------------------------------------------------
# Guida decisionale (riassunto, sempre visibile nel footer)
# -------------------------------------------------------------
st.markdown("---")
if scenario == "Off-grid":
    guidance = (
        "**Guida**: stai privilegiando indipendenza e burocrazia ridotta. Assicurati che *energia non servita* sia ≈0 nella simulazione e che la capacità batteria sia adeguata a più giorni di maltempo."
    )
else:
    guidance = (
        "**Guida**: connesso in rete → più pratiche ma massima continuità. Ottimizza LCOE e verifica payback con il prezzo della rete."
    )
st.info(guidance)
