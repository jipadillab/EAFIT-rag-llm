import streamlit as st
import time
import numpy as np
import PyPDF2
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Playground · EAFIT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }

  .stApp {
    background: #0a0a0f;
    color: #e8e8f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e3a;
  }

  /* Header banner */
  .hero-banner {
    background: linear-gradient(135deg, #0d0d20 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(100,60,255,0.08) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .hero-sub {
    font-family: 'JetBrains Mono', monospace;
    color: #6b7280;
    font-size: 0.75rem;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
  }

  /* Column cards */
  .col-card {
    background: #111122;
    border: 1px solid #1e1e3a;
    border-radius: 10px;
    padding: 1.2rem;
    height: 100%;
    min-height: 320px;
  }
  .col-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e3a;
  }
  .col-llm   .col-card-title { color: #f87171; }
  .col-rag   .col-card-title { color: #60a5fa; }
  .col-opt   .col-card-title { color: #34d399; }

  /* Response text */
  .response-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #c8c8d8;
    white-space: pre-wrap;
  }

  /* Metric chips */
  .metric-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    margin-top: 0.8rem;
    margin-right: 0.3rem;
  }
  .chip-time  { background: #1c1c30; border: 1px solid #3a3a5c; color: #a78bfa; }
  .chip-sim   { background: #0d1f18; border: 1px solid #1a3a2a; color: #34d399; }
  .chip-none  { background: #1c1414; border: 1px solid #3a1e1e; color: #f87171; }

  /* Expander styling */
  details { border: 1px solid #1e1e3a !important; border-radius: 8px !important; }

  /* Streamlit override bits */
  .stSlider > div { color: #a78bfa !important; }
  label { color: #9ca3af !important; font-size: 0.82rem !important; }
  .stSelectbox label, .stFileUploader label { color: #9ca3af !important; }
  .stTextInput > div > div > input, .stTextArea textarea {
    background: #111122 !important;
    border: 1px solid #2a2a4a !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
  }
  .stButton > button {
    background: linear-gradient(135deg, #6d28d9, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.5rem !important;
    width: 100% !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* Divider */
  hr { border-color: #1e1e3a !important; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_image_groq(file_bytes: bytes, api_key: str) -> str:
    """Use Groq's vision model to OCR an image."""
    b64 = base64.b64encode(file_bytes).decode()
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Transcribe todo el texto visible en esta imagen, manteniendo el formato original lo mejor posible."}
            ]
        }],
        max_tokens=4096,
    )
    return resp.choices[0].message.content


@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_vectorstore(text: str, chunk_size: int, chunk_overlap: int, emb_model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return None, []
    vs = FAISS.from_texts(chunks, emb_model)
    return vs, chunks


def query_llm(llm: ChatGroq, question: str, system_prompt: str = "") -> tuple[str, float]:
    msgs = []
    if system_prompt:
        msgs.append(SystemMessage(content=system_prompt))
    msgs.append(HumanMessage(content=question))
    t0 = time.time()
    resp = llm.invoke(msgs)
    return resp.content, round(time.time() - t0, 3)


def query_rag(llm: ChatGroq, question: str, vectorstore, top_k: int,
              system_prompt: str = "", emb_model=None) -> tuple[str, float, float]:
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    base_sys = (
        f"{system_prompt}\n\n" if system_prompt else ""
    ) + "Usa ÚNICAMENTE el siguiente contexto para responder. Si la respuesta no está en el contexto, di 'No encontré esa información en el documento.'\n\nCONTEXTO:\n" + context

    prompt = f"Pregunta: {question}"
    t0 = time.time()
    resp = llm.invoke([SystemMessage(content=base_sys), HumanMessage(content=prompt)])
    elapsed = round(time.time() - t0, 3)

    # Cosine similarity between question embedding and top chunk
    sim = 0.0
    if emb_model and docs:
        q_emb = np.array(emb_model.embed_query(question)).reshape(1, -1)
        c_emb = np.array(emb_model.embed_query(docs[0].page_content)).reshape(1, -1)
        sim = round(float(cosine_similarity(q_emb, c_emb)[0][0]), 4)

    return resp.content, elapsed, sim


def render_response_card(css_class: str, label: str, accent: str,
                         response: str | None, elapsed: float | None,
                         cosine: float | None):
    time_html = f'<span class="metric-chip chip-time">⏱ {elapsed}s</span>' if elapsed else ""
    sim_html  = f'<span class="metric-chip chip-sim">cos {cosine}</span>' if cosine is not None else ""
    no_sim    = '<span class="metric-chip chip-none">sin contexto</span>' if cosine is None else ""
    text_html = f'<div class="response-box">{response}</div>' if response else '<div class="response-box" style="color:#3a3a5a">— esperando pregunta —</div>'

    st.markdown(f"""
    <div class="col-card {css_class}">
      <div class="col-card-title">{label}</div>
      {text_html}
      <div>{time_html}{sim_html}{no_sim}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    api_key = st.text_input("🔑 Groq API Key", type="password",
                             placeholder="gsk_...")

    st.markdown("**Modelo**")
    model_name = st.selectbox(
        "Model Select",
        ["llama3-70b-8192", "mixtral-8x7b-32768", "llama3-8b-8192"],
        label_visibility="collapsed"
    )

    temperature = st.slider("🌡 Temperature", 0.0, 1.0, 0.3, 0.05)
    chunk_size  = st.slider("✂️ Chunk Size (tokens)", 50, 2000, 500, 50)
    chunk_overlap = st.slider("🔗 Chunk Overlap", 0, 200, 50, 10)
    top_k       = st.slider("🔍 Top-K fragmentos", 1, 10, 3)

    st.markdown("---")
    st.markdown("**System Prompt (opcional)**")
    system_prompt = st.text_area(
        "Inyectar contexto al sistema",
        placeholder='Ej: "Si no sabes, responde: No sé"',
        height=90,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**📄 Cargar documento**")
    uploaded = st.file_uploader(
        "PDF o Imagen",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )

    if st.button("🧹 Limpiar sesión"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─── MAIN AREA ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <p class="hero-title">⚡ RAG Playground</p>
  <p class="hero-sub">EAFIT · MAESTRÍA CIENCIA DE LOS DATOS · TALLER 03 — RAG vs LLM</p>
</div>
""", unsafe_allow_html=True)

# ── Document ingestion ──────────────────────────────────────────────────────────
if "doc_text" not in st.session_state:
    st.session_state.doc_text = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if uploaded and api_key:
    file_bytes = uploaded.read()
    ext = uploaded.name.split(".")[-1].lower()

    with st.spinner("📖 Procesando documento..."):
        if ext == "pdf":
            text = extract_text_from_pdf(file_bytes)
        else:
            text = extract_text_from_image_groq(file_bytes, api_key)

        st.session_state.doc_text = text

        emb_model = load_embeddings_model()
        vs, chunks = build_vectorstore(text, chunk_size, chunk_overlap, emb_model)
        st.session_state.vectorstore = vs
        st.session_state.chunks = chunks

    st.success(f"✅ Documento listo · {len(chunks)} fragmentos · {len(text):,} caracteres")

elif uploaded and not api_key:
    st.warning("⚠️ Ingresa tu Groq API Key en el sidebar para procesar el documento.")

# ── Query area ─────────────────────────────────────────────────────────────────
st.markdown("### 💬 Pregunta al documento")
question = st.text_input("", placeholder="¿Qué quieres saber del documento?",
                          label_visibility="collapsed")
run_btn = st.button("▶  Comparar respuestas")

# Persist results
for key in ["res_llm", "res_rag", "res_opt",
            "t_llm",   "t_rag",   "t_opt",
            "sim_rag",  "sim_opt"]:
    if key not in st.session_state:
        st.session_state[key] = None

if run_btn and question:
    if not api_key:
        st.error("❌ Necesitas una Groq API Key.")
    else:
        emb_model = load_embeddings_model()
        llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=temperature)

        with st.spinner("🤖 Consultando modelos..."):
            # Col 1 – LLM simple
            try:
                r, t = query_llm(llm, question)
                st.session_state.res_llm = r
                st.session_state.t_llm   = t
            except Exception as e:
                st.session_state.res_llm = f"Error: {e}"

            # Col 2 – RAG default (chunk_size=500, top_k=3)
            if st.session_state.vectorstore:
                try:
                    vs_default, _ = build_vectorstore(
                        st.session_state.doc_text, 500, 50, emb_model)
                    r, t, sim = query_rag(llm, question, vs_default, 3,
                                          emb_model=emb_model)
                    st.session_state.res_rag = r
                    st.session_state.t_rag   = t
                    st.session_state.sim_rag = sim
                except Exception as e:
                    st.session_state.res_rag = f"Error: {e}"
            else:
                st.session_state.res_rag = "⚠️ Sube un documento primero."

            # Col 3 – RAG optimizado (parámetros del sidebar)
            if st.session_state.vectorstore:
                try:
                    r, t, sim = query_rag(llm, question, st.session_state.vectorstore,
                                          top_k, system_prompt, emb_model)
                    st.session_state.res_opt = r
                    st.session_state.t_opt   = t
                    st.session_state.sim_opt = sim
                except Exception as e:
                    st.session_state.res_opt = f"Error: {e}"
            else:
                st.session_state.res_opt = "⚠️ Sube un documento primero."

# ── Results grid ───────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    render_response_card(
        "col-llm", "01 · LLM SIMPLE (Zero-shot)", "#f87171",
        st.session_state.res_llm, st.session_state.t_llm, None
    )

with c2:
    render_response_card(
        "col-rag", "02 · RAG ESTÁNDAR (default)", "#60a5fa",
        st.session_state.res_rag, st.session_state.t_rag, st.session_state.sim_rag
    )

with c3:
    render_response_card(
        "col-opt", "03 · RAG OPTIMIZADO (sidebar)", "#34d399",
        st.session_state.res_opt, st.session_state.t_opt, st.session_state.sim_opt
    )

# ── Análisis conceptual ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Fase 4 · Análisis de Métricas y Conceptos")

with st.expander("1️⃣ Alucinación — ¿Cuándo inventa el LLM Simple?"):
    st.markdown("""
**Definición:** Una alucinación ocurre cuando el LLM genera información que *suena plausible* pero no está respaldada por ninguna fuente real.

**¿Cuándo ocurre en este experimento?**
- Cuando preguntas por datos numéricos específicos del documento (fechas, cifras, nombres propios).
- Cuando el modelo "completa" información con patrones estadísticos de su entrenamiento en lugar de los hechos reales.
- El LLM Simple no tiene acceso al documento → **inventa de su memoria**.

**Ejemplo observable:** Pregunta por el nombre del profesor o la fecha de entrega; el LLM Simple probablemente dará una respuesta distinta a la del documento. Los modelos RAG extraerán el dato exacto.
    """)

with st.expander("2️⃣ Inyección de Contexto — System Prompt y 'No sé'"):
    st.markdown("""
**¿Cómo cambia la respuesta?**

Añadir un *System Prompt* como:
> *"Si la respuesta no está explícita en el contexto entregado, responde ÚNICAMENTE: 'No encontré esa información en el documento.' No inventes."*

... obliga al modelo a ser **epistémicamente honesto**: en lugar de alucinar, reconoce los límites de su contexto.

**Impacto medido:**
| Sin System Prompt | Con System Prompt |
|---|---|
| Puede inventar respuestas | Dice "No sé" cuando no hay evidencia |
| Alta confianza aparente | Calibración realista |
| Puede mezclar conocimiento externo | Respuesta anclada al documento |

Usa el sidebar para inyectar tu propio System Prompt y observa el cambio en la Columna 3.
    """)

with st.expander("3️⃣ Fine-Tuning vs RAG — ¿Por qué RAG gana aquí?"):
    st.markdown("""
| Criterio | Fine-Tuning | RAG |
|---|---|---|
| **Costo** | Alto (GPU, tiempo, datos etiquetados) | Bajo (embeddings + retrieval) |
| **Actualización** | Re-entrenar el modelo completo | Solo actualizar el vector store |
| **Documentos privados** | Riesgo de memorización de datos sensibles | Los datos no entran al modelo |
| **Transparencia** | "Caja negra" | Fuentes recuperables y auditables |
| **Latencia** | Igual tras inferencia | Ligeramente mayor (retrieval step) |

**Conclusión:** Para un corpus que cambia frecuentemente (documentos empresariales, papers, normativas), RAG es **más eficiente, actualizable y seguro** que fine-tuning.
    """)

with st.expander("4️⃣ Transformer vs No-Transformer en Embeddings"):
    st.markdown("""
**¿Los embeddings dependen de Transformers?**

En este taller usamos `all-MiniLM-L6-v2` de Sentence Transformers, que **sí es una arquitectura Transformer** (BERT-like encoder).

**Comparación:**

| Tipo | Ejemplo | Ventaja |
|---|---|---|
| **Transformer** | BERT, MiniLM, E5 | Captura semántica contextual profunda |
| **No-Transformer** | Word2Vec, FastText, GloVe | Velocidad, menor memoria |
| **Híbrido** | BM25 + Transformer | Sparse + Dense retrieval (mejor recall) |

Los Transformers generan **embeddings contextualizados**: la misma palabra tiene representación diferente según el contexto de la oración, lo que mejora enormemente la calidad del retrieval semántico frente a enfoques bag-of-words.
    """)

with st.expander("🏆 Reto — Similitud de Coseno explicada"):
    st.markdown("""
**Métrica implementada:** Similitud de Coseno entre el embedding de la pregunta y el embedding del fragmento top-1 recuperado.

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

- **1.0** → pregunta y fragmento son semánticamente idénticos
- **0.0** → ortogonales (sin relación semántica)
- **< 0** → sentidos opuestos (raro en embeddings de texto)

El valor aparece bajo cada columna RAG con el tag verde `cos X.XXXX`.
Observa cómo el RAG Optimizado con Top-K mayor puede encontrar fragmentos más relevantes (mayor coseno).
    """)

# ── Chunks inspector ───────────────────────────────────────────────────────────
if st.session_state.chunks:
    with st.expander(f"🔬 Inspeccionar fragmentos ({len(st.session_state.chunks)} chunks)"):
        for i, chunk in enumerate(st.session_state.chunks[:15]):
            st.markdown(f"**Chunk {i+1}** · {len(chunk)} chars")
            st.code(chunk[:400] + ("..." if len(chunk) > 400 else ""), language="text")
        if len(st.session_state.chunks) > 15:
            st.caption(f"… y {len(st.session_state.chunks) - 15} fragmentos más.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#3a3a5a;font-family:JetBrains Mono,monospace;font-size:0.7rem;">'
    'EAFIT · Maestría en Ciencia de los Datos · Taller 03 · Prof. Jorge Iván Padilla Buriticá, Ph.D.'
    '</p>',
    unsafe_allow_html=True
)
