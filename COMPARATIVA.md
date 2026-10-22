# COMPARATIVA.md — en qué somos mejores, con los números

> **Qué se compara.** Nuestro release (`nuestro_motor/`, copiado de `Downloads\v0.0.1`)
> contra la super versión (`super_motor/`, copiada de `Temp\opencode\super_v4`).
>
> **Condiciones de la medición**: mismo banco de **714 consultas** / **238
> documentos** (`tests/banco_rag.json`), mismo **Python 3.12.15**, mismo
> **NumPy 2.5.4**, misma máquina (**x86-64, Windows 11, i3-10110U @ 2,10 GHz**),
> el **2026-10-22**. Los dos motores se cargan desde sus copias de este paquete,
> con el mismo `MotorRAGNativo` y los mismos parámetros (`modo_semillas="no"`,
> `usar_reranker=True`, `umbral_min_score=0.01`).
>
> **Lo que NO se ha medido aquí y se marca como tal**: el comportamiento en la
> Raspberry Pi 400 (no hay Pi conectada) y el kernel C/NEON (no hay compilador
> ARM64 en esta máquina).

---

## 1. Tabla principal

| área | **nosotros** (`nuestro_motor`) | **super versión** (`super_motor`) | quién gana y por qué |
|---|---|---|---|
| **RAG hit@1 / MRR** | **0,7605 / 0,8018** (2,227 ms/consulta) | 0,7129 / 0,7637 (2,150 ms/consulta) | **NOSOTROS**: +4,76 puntos de hit@1 y +3,81 de MRR, al mismo coste. Medido con el mismo banco, el mismo Python y el mismo `top_k=5`. |
| **RAG cuántico** | **no lo tiene** | hit@1 = base (0,7129), MRR 0,7648, pero **8,53 ms/consulta** = **3,84×** más lento | **NOSOTROS**: la super versión no tiene esta trampa; su RAG cuántico es *aditivo* pero **neutral**. |
| **RAG hit@1 con `top_k=10`** | **0,7605 / hit@3 0,8445 / hit@5 0,8627 / nDCG@10 0,7975** | 0,7129 / 0,8095 / 0,8487 / 0,7685 | **NOSOTROS**: gana en las cuatro métricas. |
| **Tokenizador ↔ corpus** | **alineado** (test `test_180`; ids idénticos) | **alineado** (test `test_178`; 35.150 ids, **100 %**) | **empate**: los dos arreglados. |
| **Guard del export GGUF** | presente: un tensor no mapeado **aborta** con `ValueError` y **no deja fichero** | presente: idéntico | **empate**. |
| **Cargador INT8** | arreglado (`test_181`): escala por canal, layout empaquetado, `argmax` fp32 ≡ INT8 | **ya venía bien** (49 tensores INT8, error RMS **2,5951 %**) | **empate** en el resultado; el nuestro además trae `descuantizar_tensor_int8` y `matmul_int8_packed` en `kernels.py`. |
| **Corpus real** | **300.108.401 tokens**, 802.824 documentos, **151 shards**, 21 fuentes, 60,03 % en / 39,97 % es, **22 merges inútiles descartados**, tokenizador `4350d55f…` | `pipeline_datos.py` de **9.541 B / 247 líneas**: `construir_shards_corpus` pero **ninguna descarga HTTP** (0 `urlopen`/`requests`/`urllib`); su `--descargar` original no descargaba nada | **NOSOTROS, por goleada**: nuestro `pipeline_datos.py` son **243.079 B / 5.933 líneas / 98 funciones** y **sí** descarga. El nuestro tiene además `pipeline_fuentes_es.py` (72.878 B) con las fuentes en español. |
| **Ruta GPU / PyTorch** | `gpu/` completo: `modelo_torch.py`, `entrenar_gpu.py`, `exportar_a_pytorch.py`, `colab_entrenar.py`, `ENTRENAR_EN_COLAB.ipynb`, `presupuesto_memoria.py`, `verificar_equivalencia.py`. **Equivalencia 35/35 verificada hoy**; **86 tests** de `gpu/tests_gpu/` en verde | **no trae `gpu/` en absoluto** | **NOSOTROS**: es la diferencia entre «entrené en la GPU» y «no entrené en la GPU». |
| **Documentación de despliegue en la Pi** | `install_pi.sh` (17.212 B), `pi400.service` (2.312 B), `bench_pi.sh` (10.400 B), `pi400_bench.py` (32.246 B), `PI400.md` (24.454 B) | **no los trae** | **NOSOTROS**. |
| **Herramientas de calidad de código** | las **7 piezas rehechas**: reparador AST que **no destruye el fichero** (100 % de comentarios conservados, **70×** menos líneas reescritas), SAST `Bxxx` con **0 FP/0 FN**, tipos **8/8** con **0 %** de imports espurios, tests generados **0 %** tautológicos, score que ordena **100 > 95 > 83 > 15** | las suyas, con la **reparación léxica rota** (84,6 % compilaba) y los tests tautológicos | **NOSOTROS**: la super versión hereda lo que había **antes** de arreglarlo. |
| **Motor de embeddings** | **53.882 B**: `MAPA_CANONICA` 789, `MAPA_LEMAS` **764 lemas**, `CONCEPTOS_ES` 39, `DOMINIOS_ES` 13, `FRASES_ES` 41, `PALABRAS_VACIAS_ES` 96 | **42.811 B**: `MAPA_LEMAS` **1.196 lemas**, `CLUSTERS_SEMANTICOS` **64**, `DOMINIOS_POR_CLUSTER` 64, `PALABRAS_VACIAS_ES` 135 | **empate en tamaño de léxico** (764 vs 1.196 lemas); **NOSOTROS en el resultado**: +4,76 puntos de hit@1 con 35 % menos de lemas y menos código. |
| **Kernel C / `kernels.py`** | **14.426 B**: `cuantizar_tensor_int8`, **`descuantizar_tensor_int8`**, `empaquetar_tensor_int8`, **`matmul_int8_packed`**, `detectar_arquitectura_elf`, `es_binario_arm64`, `leer_e_machine`; `engine_pi400_accel.c` de **13.675 B** con `c_gemm_int8_packed`, `c_gemm_int8_packed_escala_canal`, `c_gemm_int8_true`, `c_rmsnorm`, `c_swiglu` | **8.449 B**: `cuantizar_tensor_int8`, `empaquetar_tensor_int8`, `matmul_int8_true`, `detectar_arquitectura_elf`, `leer_e_machine`; `engine_pi400_accel.c` de **7.654 B** | **NOSOTROS**: falta `descuantizar_tensor_int8` y el GEMM **empaquetado** (`matmul_int8_packed`), que es el que vectoriza con NEON (6-29× más rápido que el *strided* legacy). |
| **Suite de tests** | **152** (motor) + **169** (`tests/`) + **86** (`gpu/`) = **407** | **178** (166 del motor + 12 propios) | **empate en número**; **NOSOTROS en cobertura**: GPU, API, RAG, INT8, corpus, GGUF, herramientas extra. |
| **Catálogo de herramientas** | **21** por defecto · **25** con `--herramientas-extra` (memoria episódica, agenda, conversor, analizador) · **30** con `--herramientas-calidad` | **34** registradas de una vez (incluye `rag_cuantico_hibrido`, `grafo_conocimiento_rag`, `orquestar_tool_calls`, `planificar_y_ejecutar_dag`) | **ELLOS** en número; **nosotros** en que el nuestro tiene las **mismas** 4 herramientas extra **ya portadas y corregidas** (fórmula única de Ebbinghaus, Q1 garantizado, bandas validadas, varianza/IQR). |
| **Memoria episódica, agenda, conversor, analizador** | **portados** a nuestro motor bajo `--herramientas-extra` (catálogo 21 → 25) y **corregidos** | originales | **empate**: los tenemos los dos, y los nuestros están corregidos. |
| **MLA** | **no la tiene** | **×5,33** de KV (1.536 B/token frente a 8.192 con GQA 12:4, r=48) — **medido** | **ELLOS**, pero **no la necesitamos hoy**: ver [`MLA_MOE_300M_700M.md`](MLA_MOE_300M_700M.md). |
| **MoE** | **no lo tiene** | **despacho disperso real**: **7 frente a 13** llamadas a `matmul` en decode; **0,500×** de bytes de FFN leídos por token (E=4, k=2) — **medido** | **ELLOS**, pero **no compensa a 16 M**: ver el análisis. |
| **DAG + orquestador de tools** | no los tiene | parciales: el DAG **detecta el ciclo** (`ValueError`), pero el reintento del orquestador ejecuta con `args={}` | **ELLOS**, **con riesgo**: más superficie de fallo silencioso. |
| **Modelo de prueba** | preset `20x` + compuerta = **16.158.468 params** (medido), INT8 **0,9609 B/param**, holdout EN **1,6395 bits/byte** | el mismo preset = **16.158.468 params** (medido) | **empate** (es el mismo modelo base). |

---

## 2. Las tres diferencias que más pesan

### 2.1 · El RAG: 4,76 puntos de hit@1

Medido el 2026-10-22, mismo banco (714 consultas / 238 documentos), mismo
`top_k=5`, mismo Python:

| | hit@1 | hit@3 | hit@5 | MRR | nDCG@10 | ms/consulta |
|---|---|---|---|---|---|---|
| **nuestro_motor** | **0,7605** | **0,8445** | **0,8627** | **0,8018** | **0,7975** | 2,227 |
| **super_motor** | 0,7129 | 0,8095 | 0,8487 | 0,7637 | 0,7685 | 2,150 |

La diferencia viene del **motor de embeddings** y del peso del coseno denso
(`PESO_COSENO_SEMANTICO = 0,5`), no de la versión del motor: el barrido del peso
está documentado en el `README.md` (0 → 0,6751 · 0,25 → 0,7325 · **0,5 → 0,7577**
· 0,75 → 0,7535 · 1,0 → 0,7521).

Y la super versión tiene un **trampa de rendimiento**: su RAG cuántico es
«aditivo» (hit@1 = base) pero cuesta **8,53 ms/consulta** frente a **2,22 ms**
del base: **3,84×** por el mismo resultado.

### 2.2 · La ruta GPU: no existe en la super versión

Es la pieza que hace posible «entreno en Colab y sirvo en la Pi». En nuestro
release:

* `gpu/verificar_equivalencia.py` → **35/35 casos PASS** (verificado el 2026-10-22;
  peores valores: logits 3,004e-05, pérdida 2,861e-06, gradientes 5,751e-06;
  tolerancia 1e-04; 2 casos de identidad bit a bit con peor diferencia **0,0e+00**);
* `gpu/tests_gpu/` → **86/86 OK** (con `torch` 2.14.1+cpu);
* `gpu/presupuesto_memoria.py` → las tablas de VRAM (8 GB) y de RAM de la Pi 400
  que usa el análisis de crecimiento, **reproducibles**;
* `gpu/colab_entrenar.py` + `ENTRENAR_EN_COLAB.ipynb` (15 celdas) → el driver de
  Colab, con la comprobación de TF32 que **aborta** si la GPU no está en fp32 IEEE.

En la super versión: **nada de esto**. No hay `gpu/`, no hay Colab, no hay
`presupuesto_memoria.py`.

### 2.3 · El corpus: 300 M de tokens contra un pipeline que no descarga

| | nuestro | super versión |
|---|---|---|
| `pipeline_datos.py` | **243.079 B**, 5.933 líneas, **98 funciones**, **sí descarga** por HTTP | **9.541 B**, 247 líneas, **5 funciones**, **no descarga** (0 `urlopen`/`requests`/`urllib`) |
| Corpus construido | **300.108.401 tokens**, 802.824 docs, **151 shards**, 21 fuentes, 60,03 % en / 39,97 % es | — |
| Tokenizador | `4350d55f…`, 8.196 tokens, 7.808 merges, **22 merges inútiles descartados** | — |
| Holdout | EN 21.373 tokens + ES 258.082 tokens | — |
| Coste de tokenización | **953,19 s** (**314.845,8 tokens/s**) | — |

---

## 3. Lo que la super versión tiene y el nuestro no

Para que el veredicto sea completo, esto **también** es verdad:

| capacidad | super versión | nosotros | veredicto |
|---|---|---|---|
| **MLA** (compresión latente de la KV) | **real**, ×5,33 medido, con `test_162` (backward exacto y equivalencia bit a bit con/sin KV-Cache, incluida SWA + Attention Sinks + APC) | no | **ELLOS** |
| **MoE disperso** | **real**, 7 vs 13 llamadas a `matmul`, `expert_bias` persistido en el checkpoint (`test_163`) | no | **ELLOS** |
| **Gated Attention** | sí | **sí**, y con la garantía de **identidad bit a bit** con la compuerta a cero (`O *= 2·sigmoid(X_norm @ W_attn_gate)`, init a 0 ⇒ el modelo base no cambia ni un bit); loss **−0,130 / −0,102 / −0,083** (3/3 semillas) | **nosotros** (con garantía) |
| **Attention Sinks** | sí, con el bug de las posiciones ficticias **arreglado** (la CE pasa de 13,26 a 0,00 con ventana corta) | sí, con las anclas en su **posición absoluta verdadera**; CE de 1,8529 a 0,0695 (**−96,2 %**) | empate; **nosotros** con la advertencia extra (a T = 512 el beneficio es 0,00 % ⇒ sólo inferencia) |
| **MTP** | sí | sí, **portado, medido y apagado** (empeora el loss 0/3) | empate |
| **DAG + orquestador de tools** | sí | no | **ELLOS**, con el riesgo del reintento con `args={}` |
| **RAG cuántico (GraphRAG + HyDE + MMR + CRAG)** | sí | no | **ELLOS**, pero **neutral y 3,84× más lento** |

---

## 4. Conclusión

**Nuestro release es mejor motor para este proyecto** en todo lo que este
proyecto necesita: **RAG** (+4,76 pts), **corpus** (300 M de tokens), **ruta
GPU→Pi** (35/35), **despliegue en la Pi**, **calidad de código** y **cobertura
de tests**.

**La super versión es mejor motor en arquitectura**: MLA, MoE, DAG, orquestador y
Attention Sinks. Pero **esas capacidades sólo se pagan a partir de ~300 M de
parámetros**, y hoy el modelo tiene **16,2 M**.

La conclusión accionable no es «copiar la super versión»: es
**`GUIA_MERGE.md` hunk a hunk**, en el orden que fija esa guía, y **sólo el
escalón de 300 M** (§3 de `MLA_MOE_300M_700M.md`).
