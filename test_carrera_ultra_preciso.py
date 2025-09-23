# test_carrera_ultra_preciso.py
"""
Test Vocacional Ultra Preciso (Streamlit)
- 55 preguntas organizadas en 5 dominios
- ~60 carreras
- Pesos por pregunta derivados de plantillas + boosts por tema
- Filtros de incompatibilidad
- Normalización que hace que Top 3-5 destaquen (90-100%)
- Resultados guardados en 'resultados_test_vocacional.csv' (append)
"""

import streamlit as st
import numpy as np
import pandas as pd
import hashlib
import time
import os
from datetime import datetime

# ------------------------
# Config página
# ------------------------
st.set_page_config(page_title="Test Vocacional Ultra Preciso", layout="wide")

# ------------------------
# Preguntas (55) organizadas por dominios
# ------------------------
DOMAINS = {
    "Intereses Académicos y Personales": [
        "Me interesa la tecnología y la programación.",                 #0
        "Me atrae la biología, medicina y la salud.",                #1
        "Disfruto las actividades artísticas o creativas.",          #2
        "Me interesan los problemas sociales y el trabajo comunitario.",#3
        "Me interesa diseñar objetos, espacios o estructuras (arquitectura/diseño).",#4
        "Me llaman la atención las finanzas y economía.",            #5
        "Me gusta la investigación científica y el laboratorio.",   #6
        "Me interesa la comunicación, medios y periodismo.",         #7
        "Me atrae la gestión de empresas y emprendimiento.",        #8
        "Me interesa el estudio del comportamiento humano (psicología).",#9
        "Me gusta explorar nuevas tecnologías o innovaciones constantemente."#10
    ],
    "Habilidades y Destrezas": [
        "Soy bueno resolviendo problemas matemáticos y lógicos.",    #11
        "Tengo destreza manual y habilidades para el diseño físico.",#12
        "Tengo facilidad para redactar y comunicar ideas por escrito.",#13
        "Me destaco explicando conceptos a otras personas.",         #14
        "Tengo habilidad para analizar datos y cifras.",            #15
        "Soy capaz de trabajar con software y herramientas técnicas.",#16
        "Puedo planear y gestionar proyectos.",                     #17
        "Tengo sensibilidad estética y gusto por lo visual.",       #18
        "Soy perseverante en tareas largas que requieren detalle.", #19
        "Me adapto bien a nuevas herramientas y tecnologías.",     #20
        "Soy capaz de aprender procedimientos complejos rápidamente."#21
    ],
    "Personalidad": [
        "Prefiero trabajar en equipo antes que solo.",               #22
        "Me siento cómodo liderando a otros.",                      #23
        "Tengo tolerancia al estrés y presión.",                    #24
        "Soy empático y considero las emociones de otros.",         #25
        "Me gusta seguir procedimientos y rutinas.",                #26
        "Disfruto tomar decisiones responsables y con impacto.",    #27
        "Soy curioso y busco aprender por iniciativa propia.",       #28
        "Me gusta la libertad y autonomía en el trabajo.",          #29
        "Valoro la estabilidad laboral sobre la incertidumbre.",    #30
        "Me atrae la innovación y experimentar nuevas ideas.",      #31
        "Me adapto fácilmente a diferentes tipos de personas."      #32
    ],
    "Valores y Motivaciones": [
        "Valoro ayudar a la sociedad y el bien común.",             #33
        "Me motiva ganar bien económicamente.",                    #34
        "Quiero un trabajo con reconocimiento social.",            #35
        "Prefiero un trabajo con balance vida-personal.",          #36
        "Me importa la sostenibilidad y el medio ambiente.",       #37
        "Quiero oportunidades de crecimiento profesional claras.",#38
        "Me interesa la seguridad laboral y contratos estables.",  #39
        "Prefiero trabajos que permitan viajar o movilidad.",      #40
        "Me motiva crear o producir obras o servicios propios.",   #41
        "Valoro la ética y la justicia en mi trabajo.",            #42
        "Prefiero que mi trabajo tenga un impacto directo y tangible en otros."#43
    ],
    "Estilo de Aprendizaje y Trabajo": [
        "Prefiero trabajos con horarios flexibles.",               #44
        "Me adapto bien a turnos o jornadas cambiantes.",         #45
        "Disfruto tareas repetitivas y rutinarias.",              #46
        "Prefiero retos y cambios constantes en el día a día.",   #47
        "Me concentro mejor con alta autonomía.",                 #48
        "Me es más fácil trabajar en ambientes estructurados.",   #49
        "Disfruto combinar teoría y práctica en el aprendizaje.", #50
        "Prefiero roles que requieran presencia física en sitio.",#51
        "Acepto delegar responsabilidades y enfocarme en parte del proceso.",#52
        "Me gusta que mi trabajo tenga impacto tangible y visible.",#53
        "Aprendo mejor cuando puedo experimentar y probar cosas por mí mismo."#54
    ]
}

QUESTIONS = []
DOMAIN_RANGES = []
for domain, qs in DOMAINS.items():
    start = len(QUESTIONS)
    QUESTIONS.extend(qs)
    end = len(QUESTIONS)
    DOMAIN_RANGES.append((domain, start, end))

NUM_Q = len(QUESTIONS)  # should be 55

# ------------------------
# Carreras (60) y metadatos
# ------------------------
CAREERS_META = {
    # Salud y ciencias de la vida
    "Medicina": ("Salud","Atención clínica y estudio del cuerpo humano"),
    "Enfermería": ("Salud","Cuidado y apoyo directo a pacientes"),
    "Odontología": ("Salud","Cuidado de salud bucal"),
    "Fisioterapia": ("Salud","Rehabilitación física"),
    "Nutrición": ("Salud","Alimentación y salud"),
    "Biotecnología": ("Ciencias","Aplicaciones biológicas"),
    "Farmacia": ("Salud","Medicamentos y terapias"),
    "Salud Pública": ("Salud","Gestión de salud poblacional"),
    "Psicología Clínica": ("Salud","Terapia psicológica"),
    "Veterinaria": ("Salud","Salud animal"),
    # Ingeniería / Tecnología
    "Ingeniería de Software": ("Ingenierías","Desarrollo de software"),
    "Ingeniería Civil": ("Ingenierías","Obras e infraestructura"),
    "Ingeniería Mecánica": ("Ingenierías","Máquinas y sistemas"),
    "Ingeniería Electrónica": ("Ingenierías","Electrónica y sistemas"),
    "Ingeniería Industrial": ("Ingenierías","Optimización y procesos"),
    "Ingeniería Ambiental": ("Ingenierías","Sostenibilidad y medio ambiente"),
    "Ingeniería en Telecomunicaciones": ("Ingenierías","Redes y comunicaciones"),
    "Robótica": ("Ingenierías","Sistemas automáticos y robots"),
    "Ingeniería Biomédica": ("Ingenierías","Tecnología para salud"),
    "Ingeniería en Energías Renovables": ("Ingenierías","Energía sostenible"),
    # Ciencias exactas
    "Biología": ("Ciencias","Investigación en seres vivos"),
    "Química": ("Ciencias","Química y procesos"),
    "Física": ("Ciencias","Fenómenos físicos"),
    "Matemáticas Aplicadas": ("Ciencias","Modelado y análisis"),
    "Ciencia de Datos": ("TecnologiaData","Análisis de datos e IA"),
    "Inteligencia Artificial": ("TecnologiaData","Modelos inteligentes"),
    "Ciberseguridad": ("TecnologiaData","Protección de sistemas"),
    "Astronomía": ("Ciencias","Estudio del universo"),
    "Ingeniería en Materiales": ("Ciencias","Materiales y propiedades"),
    "Ingeniería en Nanotecnología": ("Ciencias","Tecnologías a escala nano"),
    # Arte y diseño
    "Arquitectura": ("ArteDiseño","Diseño de espacios"),
    "Diseño Gráfico": ("ArteDiseño","Comunicación visual"),
    "Diseño Industrial": ("ArteDiseño","Diseño de productos"),
    "Diseño de Moda": ("ArteDiseño","Moda y estilo"),
    "Artes Visuales": ("ArteDiseño","Expresión artística"),
    "Animación y VFX": ("ArteDiseño","Animación y efectos"),
    "Música": ("ArteDiseño","Composición e interpretación"),
    "Teatro": ("ArteDiseño","Actuación y dirección"),
    "Cine y Producción": ("ArteDiseño","Producción audiovisual"),
    "Fotografía": ("ArteDiseño","Captura y edición visual"),
    # Negocios y economía
    "Administración de Empresas": ("Negocios","Gestión empresarial"),
    "Contaduría": ("Negocios","Registro y análisis financiero"),
    "Finanzas": ("Negocios","Mercados y gestión financiera"),
    "Economía": ("Negocios","Análisis económico"),
    "Marketing": ("Negocios","Estrategias de mercado"),
    "Comercio Internacional": ("Negocios","Negocios entre países"),
    "Logística y Cadena de Suministro": ("Negocios","Flujo de bienes"),
    "Recursos Humanos": ("Negocios","Gestión de personas"),
    "Emprendimiento": ("Negocios","Crear y desarrollar negocios"),
    "Administración Pública": ("Negocios","Gestión gubernamental"),
    # Humanidades y comunicación
    "Derecho": ("Legales","Estudio y aplicación de leyes"),
    "Ciencias Políticas": ("Humanidades","Sistemas políticos"),
    "Sociología": ("Humanidades","Estudio de la sociedad"),
    "Antropología": ("Humanidades","Culturas y comportamiento"),
    "Historia": ("Humanidades","Estudio del pasado"),
    "Filosofía": ("Humanidades","Pensamiento crítico"),
    "Comunicación": ("Comunicacion","Medios y mensaje"),
    "Periodismo": ("Comunicacion","Reportaje y noticias"),
    "Relaciones Internacionales": ("Humanidades","Diplomacia y relaciones"),
    "Criminología": ("Humanidades","Estudio del delito"),
    # Servicios, turismo, educación y otros
    "Gastronomía": ("Hospitalidad","Cocina y experiencia culinaria"),
    "Hotelería y Turismo": ("Hospitalidad","Gestión turística"),
    "Gestión Cultural": ("Hospitalidad","Administración cultural"),
    "Trabajo Social": ("ServiciosSociales","Intervención social"),
    "Terapia Ocupacional": ("ServiciosSociales","Rehabilitación funcional"),
    "Pedagogía": ("Educación","Teoría y práctica educativa"),
    "Educación Especial": ("Educación","Atención a necesidades especiales"),
    "Lenguas Modernas": ("Educación","Estudio de idiomas"),
    "Ingeniería en Alimentos": ("Ciencias","Procesos alimentarios"),
    "Innovación y Desarrollo": ("Negocios","Proyectos innovadores")
}

CAREER_NAMES = list(CAREERS_META.keys())

# ------------------------
# Plantillas base por familia: influencia por dominio (scale 0..1)
# Order of domains: [Intereses, Habilidades, Personalidad, Valores, Estilo]
FAMILY_TEMPLATES = {
    "Salud":               [0.9, 0.8, 0.7, 0.8, 0.6],
    "Ingenierías":         [0.8, 0.9, 0.6, 0.5, 0.7],
    "Ciencias":            [0.7, 0.9, 0.6, 0.5, 0.6],
    "ArteDiseño":          [0.9, 0.6, 0.6, 0.4, 0.6],
    "Negocios":            [0.6, 0.7, 0.7, 0.7, 0.7],
    "Humanidades":         [0.6, 0.5, 0.8, 0.8, 0.5],
    "TecnologiaData":      [0.8, 0.9, 0.6, 0.4, 0.7],
    "Legales":             [0.5, 0.6, 0.9, 0.8, 0.5],
    "Comunicacion":        [0.7, 0.6, 0.8, 0.6, 0.5],
    "Hospitalidad":        [0.6, 0.6, 0.7, 0.6, 0.8],
    "ServiciosSociales":   [0.6, 0.5, 0.9, 0.9, 0.5],
    "Educación":           [0.6, 0.6, 0.9, 0.8, 0.6]
}

# ------------------------
# Helper: map question indices by topic keywords - used to boost certain careers
topic_map = {
    "tech": [0,10,16,20],    # programación, innovación, software/tecnologías
    "bio": [1,6,33],         # biología, laboratorio, ayudar/población
    "art": [2,18,41,54],     # arte, estética, crear, experimentar
    "health": [1,33,24,43],  # salud, ayudar, tolerancia al estrés, impacto tangible
    "math": [11,15,21],      # matemáticas, análisis de datos, procedimientos complejos
    "comm": [7,13,14],       # comunicación, redacción, explicar
    "lead": [23,27],         # liderazgo, decisiones
    "care": [25,33,43],      # empatía, ayudar, impacto tangible
    "business": [5,8,38],    # finanzas, emprendimiento, crecimiento profesional
    "design": [4,12,18],     # diseño, destreza manual, gusto visual
    "env": [37,54],          # sostenibilidad, experimentar con entorno
    "edu": [14,50,10],       # explicar, combinar teoría y práctica, curiosidad
}

# ------------------------
# Build deterministic career vectors:
# base from family template applied to domain question groups,
# then apply topic boosts and small deterministic noise (hash-based)
# ------------------------
def deterministic_factors(name, length):
    h = hashlib.md5(name.encode()).digest()
    arr = []
    for i in range(length):
        b = h[i % len(h)]
        factor = 0.9 + (b / 255.0) * 0.2  # 0.9 .. 1.1
        arr.append(factor)
    return np.array(arr)

def build_career_vector(career_name):
    family = CAREERS_META[career_name][0]
    template = FAMILY_TEMPLATES.get(family, [0.6]*5)
    vec = np.zeros(NUM_Q, dtype=float)
    # apply per-domain base
    for d_idx, (_, start, end) in enumerate(DOMAIN_RANGES):
        # each question in domain gets base = 1 + 4*(template[d_idx]) scaled by factor
        base_value = 1.0 + 4.0 * template[d_idx]  # between 1..5
        vec[start:end] = base_value
    # topic boosts by career name or family
    name_lower = career_name.lower()
    boosts = np.ones(NUM_Q)
    # determine applicable topics heuristically from name and family
    for topic, idxs in topic_map.items():
        if topic in name_lower or any(k in family.lower() for k in [topic]):  # family match simple
            for qi in idxs:
                if qi < NUM_Q:
                    boosts[qi] += 0.5  # increase importance
    # extra manual mappings (fine tune important careers)
    manual_boosts = {
        "medicina": ("bio",),
        "enfermería": ("health","care"),
        "odontología": ("health",),
        "ingeniería de software": ("tech","math"),
        "ciencia de datos": ("math","tech"),
        "diseño gráfico": ("art","design"),
        "arquitectura": ("design","art"),
        "derecho": ("comm",),
        "psicología clínica": ("care","comm"),
        "biotecnología": ("bio","lab"),
        "ingeniería ambiental": ("env",),
        "marketing": ("comm","business") if "Marketing" in CAREER_NAMES else None
    }
    for key, topics in manual_boosts.items():
        if key in name_lower:
            for t in topics or []:
                if t in topic_map:
                    for qi in topic_map[t]:
                        if qi < NUM_Q:
                            boosts[qi] += 0.8
    # deterministic small noise
    factors = deterministic_factors(career_name, NUM_Q)
    vec = vec * boosts * factors
    # clamp to 1..5
    vec = np.clip(vec, 1.0, 5.0)
    return vec

# Precompute career vectors
CAREER_VECTORS = {c: build_career_vector(c) for c in CAREER_NAMES}

# ------------------------
# Filtros de incompatibilidad (question index -> list of careers to penalize strongly if low)
# define some strong filters
INCOMPATIBILITY_RULES = [
    # (question_index, threshold_low, list_of_careers_or_keyword)
    (1, 3, ["Medicina","Enfermería","Odontología","Fisioterapia","Biotecnología","Farmacia","Veterinaria"]),
    (2, 3, ["Diseño Gráfico","Diseño Industrial","Artes Visuales","Animación y VFX","Fotografía","Diseño de Moda"]),
    (11, 3, ["Ingeniería de Software","Ciencia de Datos","Ingeniería Mecánica","Ingeniería Electrónica","Ingeniería Industrial","Matemáticas Aplicadas"]),
    (7, 3, ["Comunicación","Periodismo","Diseño Gráfico","Cine y Producción"]),
    (33, 3, ["Trabajo Social","Pedagogía","Educación Especial","Psicología Clínica"]),
    (37, 3, ["Ingeniería Ambiental","Ciencias Ambientales","Ingeniería Agrónoma" if "Ingeniería Agrónoma" in CAREER_NAMES else "Ingeniería Ambiental"]),
]

# ------------------------
# Compatibilidad y normalización profesional
# ------------------------
def cosine_similarity(a,b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na ==0 or nb==0:
        return 0.0
    return float(np.dot(a,b)/(na*nb))

def compute_scores(user_vector):
    raw = []
    for c in CAREER_NAMES:
        vec = CAREER_VECTORS[c]
        sim = cosine_similarity(user_vector, vec)
        # apply incompatibility rules: if user low on key question, penalize strongly
        for qidx, thresh, careers in INCOMPATIBILITY_RULES:
            if isinstance(careers, list):
                if c in careers and user_vector[qidx] < thresh:
                    sim *= 0.35  # penalty factor
            else:
                # if careers given as keyword (not used here)
                pass
        raw.append(sim)
    raw = np.array(raw)
    # if all equal (very rare), avoid division by zero
    if raw.max() - raw.min() < 1e-9:
        scaled = np.ones_like(raw) * 50.0
    else:
        # First normalize 0..1
        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        # apply spread/power to emphasize top (tunable)
        spread = np.power(norm, 1.7)
        # map to 30..95 (to later scale top into 90..100)
        scaled = 30 + 65 * spread  # base spread
        # then scale top region to 60..100 based on zscore-like
        # compute percentiles and push top 5% upward
        # final linear stretch to 0..100
        # ensure min at 5, max near 98..100
        min_target = 5.0
        max_target = 98.0
        scaled = min_target + (scaled - scaled.min()) * (max_target - min_target) / (scaled.max() - scaled.min() + 1e-12)
        # small final boost to top values to try to reach ~95-100 for top 3
        top_idx = scaled.argsort()[::-1][:5]
        for i, idx in enumerate(top_idx):
            # strongest boost for best, descending
            boost = 1.0 + 0.08 * (5 - i)  # e.g., 1.4 for i=0? actually 1.4 too big; use smaller:
            boost = 1.0 + 0.06 * (5 - i)
            scaled[idx] = min(100.0, scaled[idx] * boost)
    # final clamp 0..100
    scaled = np.clip(scaled, 0.0, 100.0)
    return scaled

# ------------------------
# Persistence: save results to CSV (append)
# ------------------------
RESULTS_FILE = "resultados_test_vocacional.csv"

def save_results(user_vector, scores):
    timestamp = datetime.utcnow().isoformat()
    row = {}
    row["timestamp_utc"] = timestamp
    # include answers as q0..q54
    for i, v in enumerate(user_vector):
        row[f"q{i}"] = float(v)
    # include top scores and the whole scores as columns
    for i, career in enumerate(CAREER_NAMES):
        row[f"score_{career}"] = float(scores[i])
    # also include top3 as separate fields
    top3_idx = np.argsort(scores)[::-1][:3]
    for rank, idx in enumerate(top3_idx, start=1):
        row[f"top{rank}_career"] = CAREER_NAMES[idx]
        row[f"top{rank}_score"] = float(scores[idx])
    # append to CSV (create if not exists)
    df_row = pd.DataFrame([row])
    if not os.path.exists(RESULTS_FILE):
        df_row.to_csv(RESULTS_FILE, index=False, encoding="utf-8")
    else:
        df_row.to_csv(RESULTS_FILE, mode="a", header=False, index=False, encoding="utf-8")
    return RESULTS_FILE

# ------------------------
# UI: formulario de preguntas
# ------------------------
st.title("🎯 Test Vocacional — Versión Ultra Precisa")
st.write("Responde honestamente. Cada pregunta va de 1 (Nada) a 5 (Muchísimo).")

# show small instructions
with st.expander("Instrucciones para un uso adecuado de el test"):
    st.write(
        "- Toma unos 10–20 minutos para responder con calma no hay tiempo limite.\n"
        "- Solamente responde según tus gustos y hábitos totalmente reales para una mayor precisión.\n"
        "- Las respuestas preguntas van dirigidas de forma general y en cualquier situación.\n"
        "- Las preguntas incluyen cosas generales y específicas para precisar las carreras.\n"
        
    )

# inputs: present domains as expanders
user_answers = np.zeros(NUM_Q, dtype=float)
for domain, start, end in DOMAIN_RANGES:
    with st.expander(f"{domain} (preguntas {start+1}–{end})", expanded=False):
        for i in range(start, end):
            key = f"q{i}"
            # default mid value 3
            user_answers[i] = st.slider(f"{i+1}. {QUESTIONS[i]}", 1, 5, 3, key=key)

# helpful quick-fill examples for testing (not auto-run)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Ejemplo: Perfil Tecnológico"):
        # set tech-oriented answers high
        for i in range(NUM_Q):
            if i in topic_map["tech"] or i in topic_map["math"] or i in topic_map["tech"]:
                st.session_state[f"q{i}"] = 5
            else:
                st.session_state[f"q{i}"] = 3
        st.experimental_rerun()
with col2:
    if st.button("Ejemplo: Perfil Salud"):
        for i in range(NUM_Q):
            if i in topic_map["bio"] or i in topic_map["health"] or i in topic_map["care"]:
                st.session_state[f"q{i}"] = 5
            else:
                st.session_state[f"q{i}"] = 3
        st.experimental_rerun()
with col3:
    if st.button("Ejemplo: Perfil Arte/Creativo"):
        for i in range(NUM_Q):
            if i in topic_map["art"] or i in topic_map["design"]:
                st.session_state[f"q{i}"] = 5
            else:
                st.session_state[f"q{i}"] = 2
        st.experimental_rerun()

# compute when user presses button
if st.button("📊 Calcular compatibilidades"):
    # compute scores
    scores = compute_scores(user_answers)
    df = pd.DataFrame({"Carrera": CAREER_NAMES, "Compatibilidad (%)": scores})
    df = df.sort_values(by="Compatibilidad (%)", ascending=False).reset_index(drop=True)

    # persist results to CSV
    saved_file = save_results(user_answers, scores)

    # top 3 display with nicer formatting
    st.subheader("🏆 Top 3 carreras recomendadas")
    top3 = df.head(3)
    cols = st.columns(3)
    medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    for i, row in top3.iterrows():
        with cols[i]:
            st.markdown(
                f"<div style='background:{medal_colors[i]};padding:18px;border-radius:12px;text-align:center;'>"
                f"<h3 style='margin:0'>{'🥇🥈🥉'[i]} {row['Carrera']}</h3>"
                f"<h2 style='margin:0'>{row['Compatibilidad (%)']:.1f}%</h2>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.caption(CAREERS_META[row['Carrera']][1])

    # explanation: contributions per domain for top1
    best = df.iloc[0]
    best_name = best["Carrera"]
    st.subheader(f"🔎 Por qué {best_name} encaja contigo (contribuciones por dominio)")
    best_vec = CAREER_VECTORS[best_name]
    domain_contribs = []
    for domain, start, end in DOMAIN_RANGES:
        contrib = float(np.sum(user_answers[start:end] * best_vec[start:end]))
        domain_contribs.append((domain, contrib))
    dc_df = pd.DataFrame(domain_contribs, columns=["Dominio", "Contribución"]).set_index("Dominio")
    st.bar_chart(dc_df)

    # table and download
    st.subheader("📋 Resultados completos")
    st.dataframe(df, use_container_width=True)

    # allow download of this run as CSV
    csv_buf = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar resultados (CSV)", data=csv_buf, file_name=f"resultados_test_{int(time.time())}.csv", mime="text/csv")

    # show where full history is stored
    st.success(f"Resultados guardados localmente en: ./{saved_file} (se appenden cada vez que calculas).")

    st.info("Nota: Este test es orientativo. Si deseas mejorar aún más la precisión, puedo:\n"
            "- Afinar manualmente vectores de carreras individuales.\n"
            "- Añadir más preguntas filtro muy específicas.\n"
            "- Permitir edición de perfiles desde la UI (modo administrador).")

# footer: show small sample of saved file if exists
st.markdown("---")
if os.path.exists(RESULTS_FILE):
    try:
        sample = pd.read_csv(RESULTS_FILE, nrows=5)
        st.caption("Últimas entradas guardadas (archivo local):")
        st.dataframe(sample)
    except Exception:
        pass
