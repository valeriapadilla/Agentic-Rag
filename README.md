# RAG Help-Desk Proof of Concept

Este repositorio contiene un prototipo de **Retrieval-Augmented Generation (RAG)** aplicado a la clasificación y resolución de tickets de soporte IT, implementado con **LangGraph**, **Chroma** y **ChatGPT (GPT‑4o)**.

---

## Estructura del proyecto

```bash
├── data/                        # Carpeta con archivos de origen (CSV, PDF, .urls, etc.)
├── ingest.py                    # Pipeline ETL: carga, trocea, genera embeddings y persiste vector store
├── agent_rag.py                 # Agente RAG: orquesta retrieval + generate en LangGraph
├── vector_store/                # Índice persistido de Chroma (vectores + metadatos)
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación de instalación y uso
```

---

## Requisitos previos

* Python ≥ 3.8
* Una cuenta en OpenAI con acceso a GPT-4/Omega (o la variante GPT-4o)
* Instalación de [chromadb](https://pypi.org/project/chromadb/)

---

## Instalación

```bash
# Clona el repositorio
git clone https://github.com/tu-usuario/tu-rag-prototype.git
cd Agentic-Rag

# Crea y activa un entorno virtual
python -m venv .venv  && source .venv/bin/activate

# Instala dependencias
pip install -r requirements.txt
```

---

## 1. Generar el índice vectorial (`ingest_General.py`)

Ejecuta el pipeline ETL para procesar todos los archivos en `data/` y construir el vector store:

```bash
python ingest_general.py
```

* Detecta archivos por extensión (`.csv`, `.pdf`, `.txt`, `.md`, `.docx`, `.pptx`, `.urls`).
* Convierte cada recurso a texto (`Document`), agrega metadatos (`id`, `title`, `type`).
* Divide en chunks de \~200 tokens con solape de 40.
* Calcula embeddings con `OpenAIEmbeddings(text-embedding-ada-002)`.
* Persiste el índice en `vector_store/` usando **Chroma**.

---

## 2. Ejecutar el agente RAG (`agent_rag.py`)

El agente RAG orquesta:

1. **Decidir** si llamar al retrieve tool o responder directo.
2. **Recuperar** tickets similares (k=3).
3. **Evaluar** relevancia (yes/no). Si no, **refinar** consulta.
4. **Generar** la solución final o clasificación con prioridad.

```bash
python agent_rag.py
```

* Ingresa tu consulta o ejemplo de ticket.
* El flujo mostrará cada paso (nodos) y la respuesta final.

---

## Personalización y mejoras

* **Ajustar `k`** en `as_retriever(k=…)` para controlar cuántos vecinos.
* **Cambiar chunk\_size/overlap** en `ingest_general.py` según longitud de documentos.
* **Probar otros embeddings** (e.g. `all-MiniLM-L6-v2` local) o LLMs.
* **Modificar prompts** en nodos para adaptar formato de salida o idioma.
* Añadir nodos extra (ej. comprobación de aprobación humana).

---

## Contribuciones

¡Contribuciones bienvenidas! Abre un issue o PR para reportar bugs, solicitar mejoras o compartir ejemplos de uso.

---

© 2025 valeriapadilla
