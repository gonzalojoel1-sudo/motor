# LEEME.md — `v0.0.2 (super)` · paquete de consulta

> **Qué es este paquete.** Es una **copia de consulta** que reúne en un solo sitio
> (a) toda la documentación del proyecto, (b) los **dos motores** —el nuestro de
> release y la «super versión» corregida del motor del usuario— y (c) el
> **análisis de la hoja de ruta de crecimiento** (MLA y MoE a escala 300 M-700 M).
>
> **Este paquete NO es el sitio de trabajo.** Los originales siguen en su sitio y
> **no se han tocado**: aquí sólo hay copias verificadas con sha256. Si quieres
> cambiar algo, cámbialo en el original.
>
> **Instantánea**: 2026-10-22 · **Python**: 3.12.15 · **NumPy**: 2.5.4 ·
> **Hardware de las mediciones**: x86-64 (Windows 11), CPU Intel i3-10110U @ 2,10 GHz.

---

## 1. Mapa del paquete

```
v0.0.2 (super)/                 4,2 GB · 318 ficheros · autosuficiente para entrenar
├── LEEME.md                     <- ESTE FICHERO: la puerta de entrada
├── INFORME_SUPER_MOTOR.md       <- qué se le hizo a la super versión y qué falta
├── COMPARATIVA.md               <- en qué somos mejores, con números medidos
├── MLA_MOE_300M_700M.md         <- la hoja de ruta del crecimiento del modelo
├── MANIFIESTO_SHA256.json       <- sha256 de todos los ficheros copiados
├── DOCUMENTACION/               <- copia de toda la documentación del proyecto
│   ├── README.md                (106.041 B)  CHANGELOG.md  (74.281 B)
│   ├── HANDOFF.md               (12.800 B)   PI400.md      (24.454 B)
│   ├── gpu/README.md            (37.195 B)
│   ├── modelos/README.md        (7.872 B)
│   ├── corpus/README.md         (59.252 B)
│   └── DOCUMENTACION_PI400_v4.md (363.187 B, 4.964 líneas, CRLF, anexos A→M)
├── super_motor/                 <- el motor "esteroides" YA CORREGIDO + auxiliares
│   ├── mi_propio_llm_pi400.py   (1.339.243 B, 27.824 líneas, sha256 f42409bbff79…)
│   ├── pipeline_datos.py  kernels.py  embedding_engine.py  engine_pi400_accel.c
│   ├── database_manager.py  makefile.md  readme.md
│   ├── GUIA_MERGE.md  INFORME_SUPER_V4.md
│   ├── corpus/tokenizador.json  (hace que test_134 no se salte)
│   └── tests/                   <- sus 178 tests (166 del motor + 12 propios)
│       ├── test_super_v4.py  medir_super_v4.py  medir_mascara_json.py
│       ├── medir_rag_cuantico.py
│       └── medicion_ANTES.json  medicion_DESPUES.json
├── nuestro_motor/               <- nuestro release, COPIADO ENTERO (314 ficheros)
│   ├── mi_propio_llm_pi400.py   (1.360.816 B, 28.736 líneas, sha256 f64391210a87…)
│   ├── mi_propio_llm_pi400_v0.0.2-web.py  <- MOTOR NUEVO (31.120 líneas, 179 tests)
│   ├── kernels.py  embedding_engine.py  engine_pi400_accel.c
│   ├── pipeline_datos.py  pipeline_fuentes_es.py
│   ├── Makefile  requirements.txt  pyproject.toml  LICENSE
│   ├── README.md  CHANGELOG.md  HANDOFF.md  PI400.md
│   ├── install_pi.sh  pi400.service  bench_pi.sh  pi400_bench.py
│   ├── examples/              <- 3 ejemplos (agente, chat OpenAI, GGUF)
│   ├── gpu/                   <- la ruta PyTorch (equivalencia 35/35)
│   │   ├── modelo_torch.py  entrenar_gpu.py  exportar_a_pytorch.py
│   │   ├── verificar_equivalencia.py  presupuesto_memoria.py  construir_release.py
│   │   ├── colab_entrenar.py  ENTRENAR_EN_COLAB.ipynb  COLAB.md  preparar_colab.ps1
│   │   └── tests_gpu/         <- 86 tests de la ruta GPU
│   ├── tests/                 <- 169 tests
│   │   ├── banco_rag.json (714 consultas)  gguf_validador.py
│   │   └── test_api_contract.py  test_banco_rag.py  test_calidad_nativa_v4.py
│   │       test_fuentes_es.py  test_gguf.py  test_gguf_token_type.py
│   │       test_herramientas_extra.py
│   ├── colab_envio/           <- el paquete listo para subir a Colab (2,8 MB)
│   ├── modelos/               <- artefactos del release (108 MB)
│   │   ├── mi_llm_pi400_v0.0.1.npz (57,4 MB)  …int8.npz (15,2 MB)
│   │   ├── mi_llm_pi400_v0.0.1.gguf (31,9 MB)  informe_release.json  README.md
│   └── corpus/               <- CORPUS COMPLETO (4,1 GB)
│       ├── shards/            <- LOS 151 SHARDS (572,4 MB) ← con esto se entrena
│       ├── manifiesto.json  tokenizador.json  anclas.npz
│       ├── *.txt  *.bz2  *.jsonl  *.npz  (fuentes crudas, ~3,4 GB)
│       ├── holdout.bin  holdout_es.bin  holdout_v2.bin (evaluación)
│       └── backup_bilingue_v1/  backup_en_v2/
└── rag_internet/               <- CAPACIDAD NUEVA, EN CURSO (ver §7)
    ├── LEEME_RAG_INTERNET.md    <- estado real: qué hay y qué falta
    ├── RESULTADOS_VERIFICACION.md <- latencias reales, SSRF, RAM y suites
    ├── DISENO_RAG_INTERNET.md   <- arquitectura, presupuesto en la Pi, seguridad
    ├── motor_en_curso/
    │   ├── LEEME.md                          <- dónde está el ejecutable
    │   └── cambios_vs_release.diff           <- +2.439 / −44 líneas
    ├── fixtures/ddg_lite_fixture.html        <- HTML real para tests sin red
    ├── mediciones/              <- latencias medidas + informe de huecos
    └── prototipos/              <- sondas con las que se midió
```

**Este paquete ya es autosuficiente.** Con `nuestro_motor/` se puede
**entrenar en la GPU sin volver a `Downloads\v0.0.1\`**: están los 151 shards
del corpus, el tokenizador, el holdout, la ruta PyTorch completa y los
artefactos del release. Lo único que no se copió son dos carpetas de modelos
ONNX que **están vacías en el origen**.

| Comprobación de autosuficiencia | Resultado |
|---|---|
| ficheros del release copiados | **314 / 314**, sha256 idéntico |
| shards del corpus | **151 / 151** (572,4 MB) |
| `corpus/manifiesto.json` | 300.108.401 tokens, hash tokenizador `4350d55f77648a86` |
| artefactos del modelo | `.npz` 57,4 MB · `.int8.npz` 15,2 MB · `.gguf` 31,9 MB |
| ruta GPU | `gpu/verificar_equivalencia.py` **35/35** desde esta carpeta |

---

## 2. Cómo lanzar cada motor

Los dos motores son **NumPy puro** (`requirements.txt` pide sólo `numpy>=2.0`).
El servidor ASGI es opcional: `pip install "fastapi>=0.110" "uvicorn>=0.27" "httpx>=0.27"`.

### 2.1 · Motor nuestro (`nuestro_motor/`)

```bash
cd nuestro_motor

# Suite del motor (152 tests)
python mi_propio_llm_pi400.py --run-tests

# Suite de tests/ (169) y de la ruta GPU (86)
python -m unittest discover -s tests -v
python -m unittest discover -s gpu/tests_gpu -v

# Equivalencia PyTorch <-> NumPy: 35/35 casos
python gpu/verificar_equivalencia.py

# Presupuesto de VRAM (8 GB) y de RAM de la Pi 400: reproducible
python gpu/presupuesto_memoria.py --json presupuesto.json

# Servidor
python mi_propio_llm_pi400.py --servidor --preset 20x --gated-attn
```

### 2.2 · Super versión (`super_motor/`)

```bash
cd super_motor

# 166 tests del motor (con corpus/tokenizador.json presente: 0 omitidos)
python mi_propio_llm_pi400.py --run-tests

# Los 12 tests propios de la super versión
python -m unittest discover -s tests -v

# Los tres arneses de medición (no forman parte de la suite)
python tests/medir_super_v4.py
python tests/medir_mascara_json.py
python tests/medir_rag_cuantico.py
```

> **Nota**: `super_motor/tests/test_super_v4.py` referencia por ruta absoluta
> `Downloads\v0.0.1\corpus\tokenizador.json` y `Downloads\v0.0.1\tests\banco_rag.json`
> (ficheros que **siguen en su sitio**). En este paquete los equivalentes están en
> `nuestro_motor/corpus/tokenizador.json` y `nuestro_motor/tests/banco_rag.json`.

---

## 3. Resumen ejecutivo: las tres conclusiones

### (a) Qué se le hizo a la super versión → [`INFORME_SUPER_MOTOR.md`](INFORME_SUPER_MOTOR.md)

Se le corrigieron **9 defectos** (A→I) y se rehízo la suite de calidad (**N1→N8**).
Los tres que cambian el comportamiento:

| | antes | después |
|---|---|---|
| **A · Máscara JSON** | JSON válido **33,3 %** → **0 %** con la máscara; coste **7,9 → 20,7 ms/token** | **no superable**: desactivada por defecto (`MASCARA_JSON_EN_GENERACION=False`), fuera del bucle de generación; la clase y sus tests siguen vivos |
| **B · RAG cuántico** | **peor** que el base: hit@1 **0,5490** vs **0,7129** | **aditivo**: hit@1 **0,7129** (= base), MRR **0,7648** vs base **0,7637** — pero **8,53 ms** por consulta frente a **2,22 ms** |
| **C · Hueco de seguridad** | 8 rutas heredaban `allow_terminal=True` y llegaban a `subprocess` | cerradas, con test con espía |

Estado: **178 tests en verde** (166 del motor + 12 propios), 1 omitido si no está
`corpus/tokenizador.json` (en este paquete **no** se omite ninguno).

### (b) En qué somos mejores → [`COMPARATIVA.md`](COMPARATIVA.md)

Medido **en esta máquina, con el mismo banco de 714 consultas y el mismo Python**:

| área | nosotros | super versión | veredicto |
|---|---|---|---|
| **RAG hit@1 / MRR** | **0,7605 / 0,8018** (2,23 ms) | 0,7129 / 0,7637 (2,15 ms) | **nosotros** (+4,76 pts de hit@1) |
| **RAG cuántico** | no lo tiene | hit@1 = base, **3,84× más lento** | **nosotros** (no existe esa trampa) |
| **Corpus real** | **300.108.401 tokens**, 21 fuentes, 151 shards, 0 merges inútiles | `pipeline_datos.py` de 9,3 KB: **sin pipeline de corpus** | **nosotros**, por goleada |
| **Ruta GPU / PyTorch** | `gpu/` con equivalencia **35/35** (verificada hoy) y **86 tests** | **no trae `gpu/`** | **nosotros** |
| **Herramientas de calidad** | las 7 piezas **rehechas** (reparador que no destruye el fichero) | las suyas, con la reparación léxica rota | **nosotros** |
| **Motor de embeddings** | 53.882 B: **764 lemas**, 13 dominios | 42.811 B: 1.196 lemas, 64 clusters | empate en lemas; **nosotros** en RAG (+4,76 pts) |
| **MLA / MoE** | no las tiene | sí, y son **reales** (×5,33 de KV; despacho disperso) | **ellos**, pero sólo a partir de ~300 M (§4) |

### (c) Qué hacer con MLA/MoE a 300 M-700 M → [`MLA_MOE_300M_700M.md`](MLA_MOE_300M_700M.md)

**No se portan ahora.** El modelo de hoy tiene **16.158.468 parámetros** y su KV
completa (32.768 tokens) son **192 MiB** de los 3.206 MB libres en la Pi: **el 6,0 %**.
La MLA lo bajaría a 36 MiB (ahorra 156 MiB por sesión) y el MoE **no aporta nada
medible**. Portarlas hoy es pagar superficie de bug por nada.

La hoja de ruta que sí sale de las mediciones:

| escalón | params | ¿cabe en 8 GB de VRAM (fp32)? | decode en la Pi (**estimado**) |
|---|---|---|---|
| **hoy** | 16,2 M | sí (0,26 GB) | **16,6 ms/token** (60 tok/s) |
| 30 M | 29.540.676 | sí (3,24 GB) | ~30 ms/token (33 tok/s) |
| 100 M | ~100 M | sí (bf16: 1,2 GB) | ~103 ms/token (9,7 tok/s) |
| 300 M | 301.737.476 | **no** en fp32 (8,43 GB); sí en bf16 (5,34 GB) | **309 ms/token** (3,2 tok/s) |
| 700 M | ~700 M | **no** en fp32; no en bf16 (8,4 GB) | **720 ms/token** (1,4 tok/s) |

**Regla deducida de todo lo medido**: cada salto de tamaño exige
**(a)** más tokens de corpus, **(b)** la paridad PyTorch de lo que se añada, y
**(c)** un test de que el modelo de la GPU es exactamente el que calcula la Pi.

---

## 4. Cómo crece este proyecto

### 4.1 · Dónde está hoy (medido)

| Magnitud | Valor | Fuente |
|---|---|---|
| Modelo | preset `20x` + **Gated Attention** | `README.md` §"Artefactos de la release" |
| Parámetros | **16.158.468** | medido en los **dos** motores (`contar_parametros()`) |
| Artefacto que se sirve en la Pi | `mi_llm_pi400_v0.0.1.int8.npz`, **15.526.718 B** → **0,9609 B/param** | `modelos/` |
| Tokens consumidos | **737.280** (480 × 24 × 64) = el **0,25 %** del corpus | `README.md` línea 1510 |
| Corpus disponible | **300.108.401 tokens**, 802.824 documentos, 151 shards, 21 fuentes | `corpus/manifiesto.json` |
| Holdout inglés | **1,6395 bits/byte** (PPL 54,5845), n = 21.312 tokens | `README.md` línea 1507 |
| Holdout español | **3,5291 bits/byte** (PPL 477,5904) | `README.md` línea 1548 |

> El propio README lo dice: «aquí es una prueba de fontanería (480 pasos, 0,25 %
> del corpus)». **El modelo todavía no conversa** y no debe venderse como si lo hiciera.

### 4.2 · Los 4 hitos ya medidos que NO hay que rehacer

| hito | evidencia | estado |
|---|---|---|
| **1 · Ruta GPU → Pi 400** | `gpu/verificar_equivalencia.py` → **35/35 casos PASS** (logits 3,004e-05, pérdida 2,861e-06, gradientes 5,751e-06; tolerancia 1e-04); **86 tests** de `gpu/tests_gpu/` en verde | **CONECTADO** |
| **2 · Corpus reproducible** | `corpus/manifiesto.json`: 300.108.401 tokens, 802.824 docs, 151 shards, 21 fuentes, 60,03 % en / 39,97 % es, **22 merges inútiles descartados**, tokenizador `4350d55f…` | **CONECTADO** |
| **3 · INT8 con kernel NEON** | escala por canal, layout empaquetado, **49 tensores INT8 en RAM**, error RMS **2,5951 %**, `argmax` fp32 ≡ INT8, pico de RAM al cargar **316.704 KB** | **CONECTADO** |
| **4 · Guard del export GGUF** | un tensor no mapeado **aborta** con `ValueError` y **no deja fichero** (0 B). Validador independiente: **0 errores** sobre el GGUF de `modelos/` | **CONECTADO** |

> Y una quinta, que es la que hace posible todo lo demás:
> **la suite de calidad de código que repara sin destruir el fichero**
> (N1: **92,31 %** compilan y **92,31 %** son equivalentes, `valido_final` veraz al
> **100 %**; comentarios conservados al **100 %**, **70×** menos líneas reescritas).

### 4.3 · La regla

> **Cada salto de tamaño exige (a) más tokens de corpus, (b) la paridad PyTorch de
> lo que se añada, y (c) un test de que el modelo de la GPU es exactamente el que
> calcula la Pi.**

Motivos, todos medidos:

* **(a)** El modelo actual ha visto el **0,25 %** del corpus y su holdout sigue en
  **1,6395 bits/byte**: con 16,2 M de parámetros, Chinchilla (~20 tokens/param)
  pide **~323 M de tokens**, o sea **~1,08 epochs** — hoy es **0,25 %** de eso.
* **(b)** La A100 multiplica en `float32` con **TF32** (10 bits de mantisa): error
  ~1e-3, **diez veces** la tolerancia de 1e-04 con la que se verifica la
  equivalencia PyTorch↔NumPy. Con TF32, «el modelo entrenado en la GPU» **no es**
  «el modelo que calcula la Pi». El paquete lo fuerza a fp32 IEEE y **aborta** si no puede.
* **(c)** El mismo criterio que ya está en `gpu/verificar_equivalencia.py`:
  35/35 casos, incluidos los de **identidad bit a bit** (la compuerta apagada o a
  cero no puede cambiar ni un bit: peor diferencia **0,0e+00**).

---

## 5. Orígenes preservados

**Nada de lo que ya existía se ha modificado.** Todo lo de esta tabla se ha
**leído** y **copiado**; el sha256 se ha comprobado **antes y después** de cada
copia (`MANIFIESTO_SHA256.json`), y se ha vuelto a comprobar al final.

| Carpeta de origen | Fichero | Bytes | Fecha (mtime) | sha256 (16) |
|---|---|---|---|---|
| `Downloads\v0.0.1\` (**nuestro release**) | `mi_propio_llm_pi400.py` | 1.360.816 | 2026-10-22 01:56 | `f64391210a87202` |
| `Downloads\v0.0.1\` | `embedding_engine.py` | 53.882 | 2026-10-21 01:02 | `a0f104abca0faa4e` |
| `Downloads\v0.0.1\` | `pipeline_datos.py` | 243.079 | 2026-10-21 16:05 | `49f2e8b65b0bfc19` |
| `Downloads\v0.0.1\` | `kernels.py` | 14.426 | 2026-10-21 00:36 | `dbbdf5244175eb65` |
| `Downloads\v0.0.1\` | `README.md` | 106.041 | 2026-10-22 04:50 | `99e77a33f379b9` |
| `Downloads\v.0.0.1 (+)\` (**hermano 1**) | `mi_propio_llm_pi400.py` | 1.128.978 | 2026-10-21 22:32 | `77a8bd1c5096d529` |
| `Downloads\v.0.0.1 (+)\` | `readme.md` | 9.693 | 2026-10-21 22:32 | `60f532d3c4be83cf` |
| `Downloads\v0.0.1(3)\` (**hermano 2**) | `mi_propio_llm_pi400.py` | 992.553 | 2026-10-21 18:04 | `d8111f234be0ee34` |
| `Downloads\v0.0.1(3)\` | `embedding_engine.py` | 39.408 | 2026-10-21 18:04 | `6f3cf103c272f14f` |
| `Downloads\v.0.01 (esteroides)\` (**hermano 3**) | `mi_propio_llm_pi400.py` | 1.231.964 | 2026-10-21 23:19 | `d92e4274a22a4917` |
| `Downloads\v.0.01 (esteroides)\` | `embedding_engine.py` | 42.811 | 2026-10-21 23:19 | `1eac32bd98c35267` |
| `Downloads\v.0.01 (esteroides)\` | `readme.md` | 11.951 | 2026-10-21 23:19 | `68de349d59989d73` |
| `Downloads\llm\` (**documentación**) | `DOCUMENTACION_PI400_v4.md` | 363.187 | — | `0d862440c02fbbd1` |
| `Temp\opencode\super_v4\` (**super versión**) | `mi_propio_llm_pi400.py` | 1.339.243 | — | `f42409bbff793ad2` |

**El hash completo de los 67 ficheros copiados** está en
[`MANIFIESTO_SHA256.json`](MANIFIESTO_SHA256.json), junto con la ruta exacta de su origen.

### Aviso: dos ficheros cambiaron de origen durante el trabajo

`Downloads\v0.0.1\README.md` y `Downloads\v0.0.1\CHANGELOG.md` los escribió **un
proceso externo** (mtime 2026-10-22 04:49 y 04:50, mientras este paquete se
construía). **No fue este paquete**: el script de copia sólo lee y comprueba
hashes. Para que el paquete reflejase el release tal y como estaba al terminar,
esos dos ficheros se **volvieron a copiar** (hash verificado antes y después) y su
hash en `MANIFIESTO_SHA256.json` es el **definitivo**.

El resto (65 ficheros) **no ha cambiado ni un byte** desde que se copiaron.

---

## 6. Verificación de este paquete

Comandos ejecutados el **2026-10-22**, **desde dentro de este paquete**,
todos en verde:

| Comprobación | Resultado |
|---|---|
| `python nuestro_motor\mi_propio_llm_pi400.py --run-tests` | **152/152 OK** (0 omitidos: ya están `README.md`, `CHANGELOG.md` y el corpus) |
| `python -m unittest discover -s nuestro_motor\tests` | **169/169 OK** |
| `python -m unittest discover -s nuestro_motor\gpu\tests_gpu` | **86/86 OK** |
| `python nuestro_motor\gpu\verificar_equivalencia.py` | **35/35 PASS** |
| `python nuestro_motor\mi_propio_llm_pi400_v0.0.2-web.py --run-tests` | **179/179 OK** (152 + 27 nuevos) |
| `python super_motor\mi_propio_llm_pi400.py --run-tests` | **166/166 OK** |
| `python -m unittest discover -s super_motor\tests` | **12/12 OK** |
| `gate_final.py` (17 criterios) | **17/17 PASS** (100 concurrentes, 0 errores 5xx) |
| sha256 de los 314 ficheros del release | **idéntico al origen** |
| sha256 de los 4 orígenes | **sin cambios** |

> Los dos motores conviven en `nuestro_motor/` y se distinguen por el nombre.
> **El release es `mi_propio_llm_pi400.py`; el nuevo es
> `mi_propio_llm_pi400_v0.0.2-web.py`.** Se ejecutan por separado y no comparten
> estado (el nuevo sólo añade herramientas opt-in y va apagado por defecto).

---

## 7. La capacidad nueva: RAG de internet **independiente**

`rag_internet/` no es una carpeta de documentación: es la **mitad hecha** de la
capacidad que falta. Se incluye **incompleta a propósito**, para poder
continuar desde este paquete.

| | estado |
|---|---|
| Buscar en internet (`buscar_web`, DuckDuckGo lite) | **hecho y medido**: 826 ms en frío, 4,2 ms de caché |
| Abrir una URL (`leer_url`, sin ejecutar nada) | **hecho y medido**: Wikipedia ES 561 KB → 3.654 ms |
| Guard **SSRF** (7 IPs privadas) | **hecho y verificado**: 7/7 bloqueadas, sin abrir socket |
| Caché 24 h · 1 petición/2 s · `User-Agent` · `--sin-web` | **hecho y probado** |
| Leer/escribir ficheros · hora con zona · fechas | **hecho y probado** |
| **Índice web propio** (`rag_web_pi400.sqlite3`) | **falta** — diseñado en `DISENO_RAG_INTERNET.md` §3 |
| **`web_consultar` con citas** | **falta** — §5 |
| **Podado a 64 MB en la Pi** | **falta** — §4 y §6 |

**Por qué va separado y no se fusiona con el RAG local:** medido. Meter
multi-query/HyDE/MMR/grafo en el índice local bajó el hit@1 de **0,7129 a
0,5616** (**−15,1 puntos**) y multiplicó la latencia por 3,6. Dos índices
separados se pueden medir y apagar por separado; uno mezclado arrastra al que
funciona.

**Invariante que se exigirá al terminarlo:** después de 100 operaciones web, el
banco local de 714 consultas tiene que seguir dando **exactamente 0,7605**.

Lectura recomendada, en este orden:
1. [`rag_internet/LEEME_RAG_INTERNET.md`](rag_internet/LEEME_RAG_INTERNET.md) — qué hay y qué falta.
2. [`rag_internet/RESULTADOS_VERIFICACION.md`](rag_internet/RESULTADOS_VERIFICACION.md) — lo medido.
3. [`rag_internet/DISENO_RAG_INTERNET.md`](rag_internet/DISENO_RAG_INTERNET.md) — cómo se termina.
