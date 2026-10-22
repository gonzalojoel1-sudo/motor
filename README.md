# micro-Transformer para Raspberry Pi 400

Un modelo de lenguaje **de 16,2 M de parámetros** que **infiere en NumPy puro**
(con kernel INT8 + NEON opcional) y **se entrena en PyTorch** en la GPU, con un
tokenizador BPE de 8.196 tokens y exportación **GGUF** validada.

> **Estado honesto:** es un **modelo de prueba** en una etapa temprana. Ha visto
> el **0,25 %** de su corpus (737.280 de 300.108.401 tokens) y **todavía no
> conversa**. Sirve para validar toda la tubería —datos, entrenamiento, cuantización,
> exportación y servicio en la Pi— y para crecer a partir de ahí.

---

## Resultados medidos

| métrica | valor |
|---|---|
| Parámetros (preset `20x` + Gated Attention) | **16.158.468** |
| Corpus | **300.108.401 tokens** · 151 shards · 22 fuentes · 40 % ES / 60 % EN |
| Tokenizador (BPE 8.196) | hash **`4350d55f77648a86`** |
| Holdout EN | **1,6395 bits/byte** (PPL 54,58) |
| Holdout ES | **3,5291 bits/byte** (PPL 477,59) |
| Pérdida de entrenamiento | 18,1919 → **4,0389** (480 pasos) |
| Pesos `.npz` (fp32) | 57,23 MB |
| Pesos `.int8.npz` | 14,81 MB (**0,961 B/parámetro**, error 0,18 %) |
| GGUF | 31,11 MB (validador independiente: **0 errores**) |
| Equivalencia PyTorch ↔ NumPy | **35/35** casos PASS |
| Tests | **152/152** (motor) · **169/169** (integración) · **86/86** (GPU) |
| Decode en la Pi 400 | *estimado* **16,6 ms/token** (60 tok/s) |

---

## Qué hay y qué no hay

**Hay:**
- Inferencia NumPy puro con KV-cache, GQA 4:2, SwiGLU, RoPE (YaRN), QK-Norm,
  **Gated Attention** y **Attention Sinks**.
- **Cuantización INT8** por canal con error del 0,18 % y **kernel en C con NEON**
  para el Cortex-A72 (`vmull_s8` + `vpadalq_s16`; sin `vdotq`, que ese SoC no tiene).
- **Exportación GGUF** con guard: si un tensor del modelo no está mapeado, **aborta**.
- **RAG local** en SQLite (FTS5-BM25 + coseno 384-D + RRF), hit@1 **0,7605** sobre
  714 consultas.
- **21 herramientas** locales (terminal con allowlist, sandbox de Python, notas,
  GPIO, calculadora AST, análisis de código...), **25** con `--herramientas-extra`
  y **5 más** de calidad de código con `--herramientas-calidad`.
- **Servidor compatible con la API de OpenAI** (`/v1/chat/completions`, SSE,
  tool calling) y ruta de **entrenamiento en GPU**.

**No hay (todavía):**
- Un modelo que converse. Con el 0,25 % del corpus visto, genera texto incoherente.
- MLA y MoE: están **medidos y aplazados** hasta 300-700 M (ver
  [`MLA_MOE_300M_700M.md`](MLA_MOE_300M_700M.md)).
- El índice del **RAG de internet**: las herramientas web están hechas y
  verificadas; el índice y las citas están diseñados pero no implementados
  (ver [`rag_internet/`](rag_internet/)).

---

## Inicio rápido

### En la Raspberry Pi 400

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r nuestro_motor/requirements.txt          # numpy

# compilar el kernel INT8 (opcional pero recomendado en la Pi)
cd nuestro_motor && make && cd ..

# chat en terminal (con el modelo del release, si lo tienes en modelos/)
python nuestro_motor/mi_propio_llm_pi400.py --chat

# servidor compatible con OpenAI
python nuestro_motor/mi_propio_llm_pi400.py --serve --port 8000
```

Los pesos no están en el repositorio: ver
[`nuestro_motor/modelos/README.md`](nuestro_motor/modelos/README.md).

### Tests

```bash
python nuestro_motor/mi_propio_llm_pi400.py --run-tests
python -m unittest discover -s nuestro_motor/tests
python -m unittest discover -s nuestro_motor/gpu/tests_gpu
```

Los tests que necesitan el corpus completo (no incluido) **se omiten** de forma
explícita en vez de fallar.

### Entrenar en la GPU

```bash
python nuestro_motor/gpu/colab_entrenar.py info          # entorno, corpus, tokenizador
python nuestro_motor/gpu/verificar_equivalencia.py       # PyTorch ↔ NumPy (35/35)
python nuestro_motor/gpu/colab_entrenar.py entrenar \
    --preset 20x --steps 18616 --batch-size 32 --seq-len 512 --lr 1e-3
python nuestro_motor/gpu/colab_entrenar.py exportar
```

> **fp32 y `--lr 1e-3`.** Con `6e-3` y batch grande el entrenamiento **diverge**
> (la loss sube). Está medido y documentado en
> [`nuestro_motor/gpu/COLAB.md`](nuestro_motor/gpu/COLAB.md).

---

## Estructura

```
nuestro_motor/               el motor del release y el nuevo
  mi_propio_llm_pi400.py            release (28.736 líneas, 152 tests)
  mi_propio_llm_pi400_v0.0.2-web.py nuevo con herramientas web (31.120 líneas, 179 tests)
  kernels.py  engine_pi400_accel.c  kernel INT8 + NEON
  embedding_engine.py               embeddings 384-D (764 lemas, 13 dominios)
  pipeline_datos.py                 construcción del corpus (reproducible)
  pipeline_fuentes_es.py            fuentes en español
  gpu/                              ruta PyTorch: entrenar, exportar, equivalencia
  tests/                            suite de integración (169) + banco RAG (714)
  corpus/                           metadatos, tokenizadores y holdouts
  modelos/                          métricas del release (los pesos van aparte)
  examples/                         ejemplos mínimos
  install_pi.sh  pi400.service      instalación y servicio systemd
super_motor/                 la "super versión" del motor, corregida y auditada
rag_internet/                RAG de internet independiente (en curso)
DOCUMENTACION/               documentación extensa (anexos A→M)
```

Los dos motores son **NumPy puro** y se ejecutan por separado; el nuevo añade
herramientas **opt-in** y las de red van **apagadas por defecto**.

---

## Datos y modelos: por qué no están aquí

| qué | por qué no | dónde |
|---|---|---|
| `corpus/` crudo y `shards/` (4,1 GB) | licencias de las fuentes + tamaño | se **regenera** con `pipeline_datos.py`; ver `corpus/LEEME_DATOS.md` |
| `modelos/*.npz`, `*.gguf` (103 MB) | artefactos binarios | **Releases** del repositorio, con su sha256 |
| `colab_envio/` | es un zip de lo mismo | el notebook está en `gpu/` |

El corpus **es reproducible**: `corpus/manifiesto.json` fija el número de tokens,
los 151 shards, las fuentes y el hash del tokenizador. Con
`pipeline_datos.py` + `pipeline_fuentes_es.py` se reconstruye exactamente.

---

## Seguridad

- **Sandbox de Python** en subproceso (`python -I`) con blocklist AST.
- **Terminal** con allowlist de binarios, sin shell, con timeout y sin operadores.
- **Blindaje de rutas**: nada fuera de las carpetas autorizadas.
- **Guard SSRF** en las herramientas web: bloquea loopback, RFC1918, link-local y
  metadatos de nube (`169.254.169.254`). El servidor **no sale a internet** salvo
  que se pase `--permitir-web`.
- El contenido de una URL **nunca se ejecuta**.

Detalle en [`rag_internet/RESULTADOS_VERIFICACION.md`](rag_internet/RESULTADOS_VERIFICACION.md).

---

## Limitaciones conocidas

1. **No conversa** con 0,25 % del corpus visto (ver arriba).
2. **El kernel NEON solo se ha validado por lectura y emulación** en x86: hay que
   compilarlo y medirlo en la Pi (`make`).
3. **Las latencias de la Pi son estimadas** (factor 3-6× sobre x86).
4. **MLA/MoE** no están implementados: son la hoja de ruta a 300-700 M.
5. El **RAG de internet** está a medias: herramientas sí, índice no.
6. El modelo **no busca en internet** por sí solo: lo hacen las herramientas, y el
   router determinista decide cuándo.

---

## Licencia

[MIT](LICENSE). El **corpus** y los **modelos** no se distribuyen aquí y conservan
las licencias de sus fuentes originales.
