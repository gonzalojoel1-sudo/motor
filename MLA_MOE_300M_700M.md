# MLA_MOE_300M_700M.md — la hoja de ruta del crecimiento del modelo

> **Qué es este documento.** **No** es una decisión de «portar MLA y MoE ahora».
> Es la **hoja de ruta** que dice, escalón a escalón, **qué falta para que el motor
> pueda servir un modelo más grande** y **en qué orden hay que traer cada cosa**.
>
> **La regla que lo gobierna** (deducida de todo lo medido en este proyecto):
> **cada salto de tamaño exige (a) más tokens de corpus, (b) la paridad PyTorch de
> lo que se añada, y (c) un test de que el modelo de la GPU es exactamente el que
> calcula la Pi.**
>
> **Cómo se han medido las cifras de aquí**:
> * Las de **arquitectura, parámetros, KV y MoE** salen de las implementaciones
>   reales de `super_motor/mi_propio_llm_pi400.py` (el motor con MLA y MoE), con
>   el instrumentador de bytes leídos descrito en §5. La fórmula de parámetros se
>   validó contra `contar_parametros()` del motor real en **5 escalas** (78.720,
>   889.984, 12.507.904, 12.040.864 y 44.882.048 parámetros): **coincidencia
>   exacta, delta = 0** en las cinco.
> * Las de **VRAM y de RAM de la Pi** salen de `nuestro_motor/gpu/presupuesto_memoria.py`,
>   que es **reproducible** y se ha ejecutado el 2026-10-22.
> * Las de **decode** se han **medido** en x86-64 y **estimado** para la Pi 400
>   (que **no** se ha medido: `PI400.md` lo marca como *pendiente*). Cada
>   estimación va marcada como **[EST]**.

---

## 1. Hoy: por qué MLA y MoE **no** se portan ahora

### 1.1 · Dónde está el proyecto

| magnitud | valor | fuente |
|---|---|---|
| modelo | preset `20x` + **Gated Attention** | `README.md` |
| **parámetros** | **16.158.468** | **medido** en los dos motores (`contar_parametros()`) |
| artefacto que se sirve en la Pi | `mi_llm_pi400_v0.0.1.int8.npz` = 15.526.718 B → **0,9609 B/param** | `modelos/` |
| **tokens consumidos** | **737.280** (480 × 24 × 64) = el **0,25 %** del corpus | `README.md` |
| corpus disponible | **300.108.401 tokens**, 802.824 docs, 151 shards, 21 fuentes | `corpus/manifiesto.json` |
| holdout EN | **1,6395 bits/byte** (PPL 54,5845), n = 21.312 tokens | `README.md` |
| **KV a 32.768 tokens** | **192 MiB** (6.144 B/token × 32.768) | **medido** con `RingBufferKVCache` |

### 1.2 · La cuenta que lo deja claro

```
RAM de la Pi 400                         4.096 MB
 − SO + Python + NumPy + SQLite          −450 MB
 − historial global (8.000.000 tokens)   −440 MB
 ─────────────────────────────────────────────
 = pesos + KV                              3.206 MB

La KV del modelo de HOY a 32.768 tokens   192 MiB  =  6,0 % de los 3.206 MB
```

**La MLA (r = 48) bajaría esa KV a 36 MiB** (×5,33): ahorra **156 MiB por
sesión** de un presupuesto de 3.206 MB. Con GQA caben **132 sesiones** a 4.096
tokens de contexto; con MLA, **709**. El número **no** es el problema.

Y el **MoE no aporta nada medible** a esta escala (§5.3): elude el problema
equivocado.

### 1.3 · El argumento definitivo: **el decode, no la KV**

En un decode, el motor **lee todos los pesos del modelo**: en INT8 con el kernel
NEON son **1 byte por parámetro y token**. Medido en x86-64 con los dos motores:

| | valor |
|---|---|
| bytes de pesos leídos por token de decode (16,2 M) | **64.493.568 B** |
| decode medido (x86-64, fp32, KV-cache) | **9,688 ms/token** (103,2 tok/s) |
| ancho de banda efectivo del motor | **6,657 GB/s** |
| ancho de banda secuencial de la máquina (micro-benchmark) | **6,85 GB/s** |
| **eficiencia del motor** | **97,2 %** |

> **El decode es un problema de ancho de banda, no de cómputo**: el motor consigue
> el **97,2 %** de la lectura secuencial de la máquina. No queda margen en la
> aritmética: lo único que cambia el tiempo por token son **los bytes que hay
> que leer**.

Y los bytes que hay que leer son **los pesos**, no la KV. Esto separa las dos
capacidades en dos palancas **distintas**:

| palanca | qué reduce | cuándo importa |
|---|---|---|
| **MLA** | la **RAM** que ocupa la KV del contexto | cuando la KV se come el presupuesto (a partir de ~500 M) |
| **MoE** | los **bytes leídos por token** (y por tanto el tiempo de decode) | cuando los pesos son el cuello de botella (a partir de ~500 M) |

A **16,2 M**:

* la KV son **192 MiB** de los 3.206 MB ⇒ **6,0 %**; la MLA ahorraría **156 MiB**
  (el **4,9 %** del presupuesto de RAM). **No compensa.**
* el decode lee **15,4 MB por token** (INT8); el MoE **no** cambia eso a esta
  escala, porque la FFN es el ~60 % de un modelo de 16 M y el MoE sólo reparte
  `n_hidden` en `E + 2` unidades (el modelo MoE tiene **menos** parámetros
  totales, no más: E=4 → 99.328 frente a 111.744 en la medición).

**Por eso hoy no se porta nada de esto.** Portar MLA y MoE ahora es:
pagar una superficie de bug nueva (el Anexo K del `DOCUMENTACION_PI400_v4.md`
documenta **89.024 B** perdidos en silencio por un exportador GGUF, y un
`test_162` que pasaba **sobre un contador que mentía 5,3×**) por un beneficio
**medible del 4,9 % de la RAM** y **cero** en el tiempo de decode.

---

## 2. La curva: los 8 escalones

### 2.1 · Presupuesto de VRAM para entrenar (8 GB)

Fórmula de `gpu/presupuesto_memoria.py`:
`pesos 4 B + gradientes 4 B + momentos de Adam 8 B = 16 B/param` (fp32);
`2 + 2 + 8 = 12 B/param` (bf16); `2 + 2 + 4 = 8 B/param` (AMP con estados en bf16).

| escalón | arquitectura (capas/D/cab:kv/hidden) | params | INT8 en la Pi | **VRAM fp32** | **VRAM bf16** | **VRAM AMP+bf16** |
|---|---|---|---|---|---|---|
| **hoy** | 6/384/12:4/1408 | **16,2 M** | 15,4 MB | 0,26 GB **SÍ** | 0,19 GB **SÍ** | 0,13 GB **SÍ** |
| 30 M | 7/576/9:3/1536 | 29,5 M | 28,2 MB | 0,47 GB **SÍ** | 0,35 GB **SÍ** | 0,24 GB **SÍ** |
| 60 M | 12/640/10:4/1728 | 58,9 M | 56,2 MB | 0,94 GB **SÍ** | 0,71 GB **SÍ** | 0,47 GB **SÍ** |
| 120 M | 18/768/12:4/2048 | 119,7 M | 114,1 MB | 1,91 GB **SÍ** | 1,44 GB **SÍ** | 0,96 GB **SÍ** |
| 200 M | 22/896/14:4/2432 | 196,8 M | 187,7 MB | 3,15 GB **SÍ** | 2,36 GB **SÍ** | 1,57 GB **SÍ** |
| **300 M** | 26/1024/16:4/2816 | **301,7 M** | 287,8 MB | **4,83 GB SÍ** | 3,62 GB **SÍ** | 2,41 GB **SÍ** |
| 500 M | 35/1152/18:6/3072 | 505,3 M | 481,9 MB | **8,08 GB NO** | 6,06 GB **SÍ** | 4,04 GB **SÍ** |
| 700 M | 35/1408/22:4/3584 | 705,9 M | 673,2 MB | **11,29 GB NO** | **8,47 GB NO** | 5,65 GB **SÍ** |

> **Ojo**: la tabla de arriba es sólo la parte de **pesos + gradientes +
> optimizador**. Falta el bloque de **logits + cross-entropy**
> (`2 × B × T × vocab × 4 B`), que con vocabulario 8.196 y secuencias largas es lo
> que hace que un modelo que «cabe en la tabla» **no quepa de verdad** (con B = 64,
> T = 512 son **2,15 GB** de logits). Con B = 32 × T = 512 el presupuesto de
> activaciones de los 30 M-300 M es manejable; a 500 M y 700 M hay que bajar el
> batch.

**El modelo más grande que cabe en 8 GB con el corpus de 300 M tokens es, por
VRAM, el de 700 M en AMP + bf16 (5,65 GB).** Pero el corpus y la Pi mandan (§2.3,
§2.5).

### 2.2 · ¿Cuántos tokens de corpus hacen falta? (Chinchilla ≈ 20 tokens/param)

| escalón | params | tokens de Chinchilla | **epochs del corpus de 300,1 M** | ¿asumible? |
|---|---|---|---|---|
| **hoy** | 16,2 M | 323 M | **1,08** | sí — pero hoy sólo se han consumido **737.280 (0,25 %)** |
| 30 M | 29,5 M | 591 M | **1,97** | sí |
| 60 M | 58,9 M | 1.178 M | **3,93** | sí: 4 pasadas por el corpus |
| 120 M | 119,7 M | 2.394 M | **7,98** | **al límite**: 8 epochs |
| 200 M | 196,8 M | 3.936 M | **13,11** | **no**: 13 epochs ⇒ sobreajuste |
| 300 M | 301,7 M | 6.035 M | **20,11** | **no**: 20 epochs ⇒ sobreajuste seguro |
| 500 M | 505,3 M | 10.106 M | **33,67** | **no** |
| 700 M | 705,9 M | 14.117 M | **47,04** | **no** |

**Respuesta directa a la pregunta**: *«si el corpus actual (300 M tokens) da para
un modelo de 300 M»* — **no**. Harían falta **6.035 M de tokens**, o sea **20,11
veces** el corpus. **20 epochs es sobreajuste seguro**: el modelo vería cada
documento 20 veces y la perplejidad de holdout se hundiría por sobreajuste, no
por aprendizaje.

**Cuánto tiene que crecer el corpus**, por escalón (a 20 tokens/param y con
margen para no sobreajustar, digamos ≤ 4 epochs):

| escalón | tokens necesarios | **× sobre el corpus actual** |
|---|---|---|
| 30 M | 591 M | **× 1,97** |
| 60 M | 1.178 M | **× 3,93** |
| 120 M | 2.394 M | **× 7,98** (ya en el límite de 4 epochs ⇒ **×2**) |
| 300 M | 6.035 M | **× 20,11** ⇒ con 4 epochs: **× 5,03** |
| 700 M | 14.117 M | **× 47,04** ⇒ con 4 epochs: **× 11,76** |

> El corpus actual **sí** da para el escalón de **30 M** y el de **60 M** (que es
> el siguiente salto natural desde 16,2 M). Para **120 M** hay que duplicarlo; a
> partir de **300 M** hay que multiplicarlo por 5, y a **700 M** por 12.

### 2.3 · RAM en la Pi 400: pesos INT8 + KV (3.206 MB libres)

| escalón | pesos INT8 | KV B/tok **GQA** | KV B/tok **MLA r=48** | ratio | ctx máx **GQA** | ctx máx **MLA** | sesiones @4096 (GQA / MLA) |
|---|---|---|---|---|---|---|---|
| **hoy** (6/384/12:4) | 15,4 MB | 6.144 | 1.152 | ×5,33 | 32.768 tok | 32.768 tok | 132 / 709 |
| 30 M (7/576/9:3) | 28,2 MB | 10.752 | 1.344 | ×8,00 | 32.768 tok | 32.768 tok | 75 / 605 |
| 60 M (12/640/10:4) | 56,2 MB | 24.576 | 2.304 | ×10,67 | 32.768 tok | 32.768 tok | 32 / 349 |
| 120 M (18/768/12:4) | 114,1 MB | 36.864 | 3.456 | ×10,67 | 32.768 tok | 32.768 tok | 21 / 229 |
| 200 M (22/896/14:4) | 187,7 MB | 45.056 | 4.224 | ×10,67 | 32.768 tok | 32.768 tok | 17 / 182 |
| **300 M** (26/1024/16:4) | 287,8 MB | 53.248 | 4.992 | ×10,67 | **32.768 tok** | 32.768 tok | **14** / 149 |
| 500 M (35/1152/18:6) | 481,9 MB | 107.520 | 6.720 | ×16,00 | **26.566 tok** | **32.768 tok** | 6 / 103 |
| 700 M (35/1408/22:4) | 673,2 MB | 71.680 | 6.720 | ×10,67 | 32.768 tok | 32.768 tok | 9 / 96 |

**La curva, en una frase**: hasta **300 M** el contexto completo de 32.768 tokens
cabe **sin** MLA. **A 500 M** (con `n_kv_head = 6`) el GQA se queda en **26.566
tokens** y la MLA lo devuelve a **32.768**. **A 700 M**, con `n_kv_head = 4`, la
propia arquitectura ya vuelve a caber y la MLA es **opcional** (pero si se sube
`n_kv_head` a 6, la KV pasa de 71.680 a 107.520 B/token y la MLA vuelve a ser
**imprescindible**: ×16,00).

> **Conclusión de la curva**: **la MLA sólo empieza a pagar entre 300 M y 500 M**,
> y sólo si se quiere el contexto completo **y** se insiste en `n_kv_head = 6`.
> **La palanca más barata, siempre, es bajar `n_kv_head` o `n_layer`** — nunca el
> embedding (es lo que da la calidad).

### 2.4 · **La velocidad de decode en la Pi** (el límite real)

Modelo de la medición:

```
t_por_token = (params × B_por_param) / ancho_de_banda_efectivo
```

* **B/param = 1** en INT8 con el kernel NEON (es lo que se sirve en la Pi).
* **Ancho de banda efectivo del motor: 6,657 GB/s [MEDIDO]** en x86-64
  (16,2 M de parámetros, 64.493.568 B leídos por token, 9,688 ms/token),
  = **97,2 %** de la lectura secuencial de la máquina (6,85 GB/s).
* **Ancho de banda de la Pi 400: 1,00 GB/s [EST]** — ancho de banda secuencial de
  **un** núcleo Cortex-A72 con NEON; la LPDDR4 da ~12,8 GB/s teóricos repartidos
  entre 4 núcleos y la cadena NumPy+BLAS no llega al pico. Se aplica la **misma
  eficiencia del motor medida en x86** (0,972).
* ⚠️ **La Pi 400 no se ha medido**: `PI400.md` §8 dice literalmente
  *«Decode ms/token: pendiente de medir en la Pi»*. Esta tabla es una
  **estimación**; el `bench_pi.sh` del release es lo que la convertiría en
  medición.

| escalón | MB leídos por token (INT8) | x86 ms/token **[MEDIDO→escalado]** | **Pi ms/token [EST]** | **Pi tok/s [EST]** | **Pi s por 100 tokens [EST]** | Pi s por **1.000** tokens [EST] |
|---|---|---|---|---|---|---|
| **hoy** (16,2 M) | 15,4 | 2,4 | **16,6 ms** | **60,2** | 1,7 | 16,6 |
| 30 M | 28,2 | 4,4 | **30,4 ms** | 32,9 | 3,0 | 30,4 |
| 60 M | 56,2 | 8,8 | **60,6 ms** | 16,5 | 6,1 | 60,6 |
| 120 M | 114,1 | 18,0 | **123,1 ms** | 8,1 | 12,3 | 123,1 |
| 200 M | 187,7 | 29,6 | **202,4 ms** | 4,9 | 20,2 | 202,4 |
| **300 M** | 287,8 | 45,3 | **310,4 ms (0,31 s)** | **3,2** | 31,0 | 310,4 |
| 500 M | 481,9 | 75,9 | **519,8 ms (0,52 s)** | 1,9 | 52,0 | 519,8 |
| **700 M** | 673,2 | 106,0 | **726,1 ms (0,73 s)** | **1,4** | 72,6 | 726,1 |

**La conclusión que hay que decir con todas las letras**:

> **A 300 M-700 M, la Pi da 0,31-0,73 s por token.** Con una respuesta de
> 300 tokens son **93 s a 300 M** y **218 s a 700 M**. **En la Pi, un modelo de
> ese tamaño sólo sirve para respuestas cortas**, o para generación en segundo
> plano mientras el usuario espera otra cosa.

Por tanto la estrategia correcta **no** es «un modelo grande en la Pi»:

| estrategia | cuándo |
|---|---|
| **modelo grande + INT8 + KV comprimida + contexto corto** | tareas de una pasada, resúmenes, clasificador: **300 M con ventana de 4.096 tokens** y la MLA si quieres más sesiones simultáneas. |
| **modelo pequeño + RAG + herramientas** | **el caso de uso real de este proyecto**: con 16,2 M y un RAG de hit@1 **0,7605** sobre 714 consultas, la respuesta sale del corpus, no de los pesos. Y las **herramientas** (21 en el catálogo) hacen el trabajo pesado. |
| **modelo grande en la GPU, resultado en la Pi** | si el trabajo se puede hacer *antes* (por lotes, no en línea). |

### 2.5 · El coste de entrenar cada escalón

`FLOPs ≈ 6 × params × tokens` (adelante + atrás). Estimaciones con **MFU 40 %**
(A100 40 GB bf16, 312 TFLOPS) y **35 %** (T4 fp16, 65 TFLOPS) **[EST]**.

| escalón | tokens de Chinchilla | **A100 40 GB** | **L4** | **T4** |
|---|---|---|---|---|
| hoy (16,2 M) | 323 M | **0,1 h** | 0,2 h | 0,3 h |
| 30 M | 591 M | **0,2 h** | 0,6 h | 1,3 h |
| 60 M | 1.178 M | **0,9 h** | 2,4 h | 5,1 h |
| 120 M | 2.394 M | **3,8 h** | 9,9 h | 21,0 h |
| 200 M | 3.936 M | **10,3 h** | 26,7 h | 56,7 h |
| **300 M** | 6.035 M | **24,3 h** | 62,7 h | 133,4 h |
| 500 M | 10.106 M | **68,2 h** | 175,8 h | 374,1 h |
| **700 M** | 14.117 M | **133,1 h** | 343,1 h | **730,0 h** |

> La A100 da **200 CU/mes** en el plan de Colab → **≈ 37 h de A100** al mes
> (5,40 CU/h). **El escalón de 300 M (24,3 h) entra en un mes de CU**, pero
> **el de 700 M (133,1 h) no**, y el de 500 M (68,2 h) tampoco con margen.
> Con un L4 (1,19 CU/h) el presupuesto mensual son **~168 h**: el de 300 M entra,
> el de 700 M **no**.

**El tiempo de GPU nunca es el problema: el corpus y la RAM de la Pi sí.**

---

## 3. Qué hay que portar **antes** de llegar a cada escalón

### 3.1 · Escalón 16,2 M → 30 M → 60 M — **no hace falta portar nada**

* Sube `n_layer`, `n_embd`, `n_hidden` en `gpu/presupuesto_memoria.py`.
* Reentrena con `gpu/entrenar_gpu.py` y vuelve a construir el release con
  `gpu/construir_release.py`.
* **Verifica la paridad** con `gpu/verificar_equivalencia.py` (**35/35**).
* Comprueba el presupuesto con `gpu/presupuesto_memoria.py`.
* El corpus actual **alcanza** (1,97 y 3,93 epochs).

### 3.2 · Escalón 120 M — **el corpus no da** (7,98 epochs)

Antes de entrenar 120 M hay que **duplicar el corpus** (o bajar a 4 epochs: 2,4 GB
de tokens nuevos) y volver a tokenizar (`pipeline_datos.py`, **314.845,8 tokens/s**
de throughput: 300 M de tokens en **15,9 min**).

### 3.3 · Escalón 300 M — **aquí empieza el trabajo de verdad**

Tres cosas, **en este orden**:

#### (1) La MLA necesita su **equivalente en PyTorch** — `nuestro_motor/gpu/modelo_torch.py`

> **Esto no es opcional.** El motor de la Pi es **NumPy** y el de la GPU es
> **PyTorch**. Si la MLA existe sólo en NumPy, **se entrena un modelo en la GPU y
> la Pi calcula otro**: exactamente el fallo que el paquete ya sabe detectar (el
> TF32 de la A100 da un error de ~1e-3, **diez veces** la tolerancia de 1e-04 de
> la equivalencia).

Lo que hay que traer, según `super_motor/GUIA_MERGE.md`:

| hunk | qué toca | de dónde | a dónde |
|---|---|---|---|
| **13** | el **guard del GGUF**: `_mapa_tensores_gguf` + `sin_mapear` | `super_motor/mi_propio_llm_pi400.py` | `nuestro_motor/mi_propio_llm_pi400.py` — **el penúltimo**, porque toca el exportador, que ya está validado por `tests/gguf_validador.py` y por el `.gguf` de `modelos/` |
| **14** | el **pre-tokenizador GPT-2**: `RE_UNIDAD_BPE` + `pre_tokenizar_gpt2` | idem + `pipeline_datos.py` | **el último**: cambiar la unidad de pre-tokenización **invalida los ids** del corpus y del GGUF ⇒ hay que re-entrenar el tokenizador y re-exportar |
| **1** | **seguridad de la terminal** (`allow_terminal=False` en 8 rutas) | idem | idem — no cambia ningún resultado medido |
| **2**-**7** | la suite de calidad (N1-N7) | idem | idem — son lógicas de análisis, no tocan el modelo; se pueden aplicar primero |

Los **tests** que hay que traer:

| test | qué fija |
|---|---|
| `test_162` | **MLA real**: `RingBufferKVCache` con `use_mla=True` **no** reserva `k_buffers` ni `v_buffers` (0 B); `bytes_reservados()` es **honesto** (suma lo que hay); compresión **×5,33** con `n_kv_head=4, d_k=32, kv_lora_rank=48` y **×8,0** con `r=8, d_k=8`; **equivalencia bit a bit** entre `usar_kv_cache=True` y `False`, incluida la ventana deslizante con Attention Sinks y el APC |
| `test_163` | **MoE + Gated Attention**: presupuesto iso-parámetro, **7 frente a 13** llamadas a `matmul` en decode, `expert_bias` que **no** se muta con `devolver_grads=True` |
| `test_164` | **MTP + compartición de pesos por bloques** |
| `test_165` | **Attention Sinks** que preservan las anclas en la ventana deslizante |
| `test_166` | **RAG context stitching** con ventana de vecinos |
| `test_177` | **gradcheck numérico** de las arquitecturas next-gen completo |

Y el **test nuevo** que este proyecto exige y la super versión **no** tiene:

> **La paridad PyTorch de la MLA.** `gpu/verificar_equivalencia.py` tiene hoy
> **35/35** casos. Al añadir la MLA hay que añadir **los casos de MLA** (logits,
> pérdida y gradientes de `W_dkv`, `W_uk`, `W_uv` y de la ruta de
> reconstrucción `_reconstruir_kv_mla`) con la **misma tolerancia de 1e-04**, y
> un caso de **identidad bit a bit** para la ruta sin MLA.

#### (2) El **guard del export GGUF** debe seguir abortando

La MLA se exporta como **`W_dkv` × `W_uk`** (un producto de matrices), no como
`W_k`/`W_v` densos. El guard de nuestro release **rastrea cada tensor del motor**
y **aborta** (`ValueError`, **0 B** escritos) si alguno no está mapeado. Al
portar la MLA hay que:

1. **añadir** `W_dkv`, `W_uk`, `W_uv` al mapa de tensores de extensión, **o**
2. **dejar que aborten** y aceptar que el GGUF de la MLA **no** se exporta
   todavía (la arquitectura `llama` de llama.cpp no expresa MLA).

> **Lo que NO hay que hacer**: convertir el «abortar» en «descartar en silencio».
> Eso es exactamente lo que hacía el motor hermano (Anexo K §K.2: **89.024 B**
> perdidos en **5** casos, sin avisar).

#### (3) Lo que **no** se toca

* **`nuestro_motor/embedding_engine.py`**: no se toca. El motor de embeddings
  es ortogonal a la arquitectura del Transformer.
* **La ruta GPU** (`gpu/`): no se toca **salvo** para añadir la paridad PyTorch
  de lo nuevo (que es lo que hay que *añadir*, no lo que hay que *cambiar*).
* **El corpus**: no se toca hasta que el escalón lo exija (§2.2).

### 3.4 · Escalón 500 M y 700 M — **el MoE entra en juego**

#### ¿Cuándo merece la pena el MoE?

Medido en el motor real (§5.3), el MoE **reduce los bytes de FFN leídos por
token** a `(1 + k) / (E + 2)` del denso:

| E, k | bytes de FFN leídos por token de decode (medido) | ratio frente al denso | params del modelo de prueba | llamadas a `matmul` |
|---|---|---|---|---|
| denso | 589.824 B | **1,000×** | 111.744 | 34 |
| **E=4, k=2** | **364.544 B** | **0,618×** | 99.328 | 70 |
| E=4, k=1 | 266.240 B | 0,451× | 99.328 | 58 |
| **E=8, k=2** | **222.208 B** | **0,377×** | 104.012 | 70 |
| E=8, k=1 | 163.840 B | 0,278× | 104.012 | 58 |
| **E=16, k=2** | **136.704 B** | **0,232×** | 104.616 | 73 |
| E=16, k=4 | 221.184 B | 0,375× | 104.616 | 106 |
| E=64, k=8 | 352.256 B | 0,597× | 245.152 | 166 |

*(medido con el modelo de prueba de 2 capas / D=64; n = 1 token de decode)*

**Pero el MoE sólo compensa si la GPU también lo implementa** y si la GPU
**también** lee de forma dispersa. Y aquí está el problema medido:

> **En el decode, el número de llamadas a `matmul` SUBE** (34 → 70 con E=8, k=2),
> porque hay un router + 1 experto compartido + k expertos por capa. Los **bytes**
> bajan, pero las **llamadas** suben, y en NumPy puro cada llamada tiene un coste
> fijo. **El MoE optimiza el ancho de banda, no el número de operaciones**: en la
> Pi, con el kernel NEON y 1 B/param, la cuenta es de bytes; en la GPU, con el
> GEMM vectorizado, la cuenta es de FLOPs.

**Conclusión sobre el MoE**: a **500 M-700 M** el MoE **sí** es el camino para
meter más capacidad en 3.206 MB (más parámetros con los mismos bytes leídos por
token), **pero** hay que cumplir **tres** condiciones:

1. **La GPU tiene que implementar el mismo despacho disperso** (si la GPU entrena
   un MoE denso y la Pi sirve uno disperso, se entrena un modelo y se sirve otro).
2. **`expert_biases` tiene que serializarse en el checkpoint**: el sesgo
   Aux-Loss-Free es **estado del modelo**, no un peso, y sin él el balanceo de
   expertos se reinicia en cada reanudación. El `test_163` ya lo fija.
3. **El `export_int8` tiene que cuantizar los expertos**: hoy el cuantizador
   recorre `self.params`, así que los expertos entran si están ahí, pero el
   **layout empaquetado** (`matmul_int8_packed`) es por matriz: hay que decidir si
   se empaqueta por experto (más bytes de escala) o por bloque.

**El preset MoE que maximiza la capacidad servible en 3.206 MB** es, con la
implementación de la super versión (`_dimensiones_ffn_capa`: el presupuesto
`n_hidden` se reparte en `E + 2` sub-expertos, `H_unit = n_hidden // (E + 2)`):

| preset | H_unit | bytes de FFN leídos por token | capacidad (params) en 3.206 MB |
|---|---|---|---|
| denso | n_hidden | 1,000× | 100 % |
| **E=8, k=2** | n_hidden/10 | **0,377×** | **~95 %** del denso (el MoE reparte `n_hidden` en 9 unidades y activa 3) |
| E=16, k=2 | n_hidden/18 | 0,232× | ~90 % |
| E=16, k=4 | n_hidden/18 | 0,375× | ~90 % |
| E=64, k=8 | n_hidden/66 | 0,597× | ~85 % |

> **E=8, k=2 es el punto dulce**: lee el **37,7 %** de los bytes de FFN y es el
> primero que reduce el tiempo por token de forma clara sin que la dispersión
> (y las llamadas extra) se coma el ahorro.

---

## 4. Conclusión accionable

### 4.1 · ¿Portar MLA, MoE o las dos?

| | hoy (16,2 M) | 30-60 M | 120 M | **300 M** | 500-700 M |
|---|---|---|---|---|---|
| **MLA** | **NO** | **NO** | **NO** | **SÍ** (si se quiere el contexto completo y `n_kv_head = 4`… o si se quiere más sesiones) | **opcional** (a 700 M con `n_kv_head = 4` la GQA ya cabe) |
| **MoE** | **NO** | **NO** | **NO** | **NO** (la VRAM de 8 GB es el límite, no los bytes leídos) | **SÍ**, con E=8, k=2 |

**Las dos, pero no ahora y no a la vez.**

### 4.2 · Con qué parámetros

* **MLA**: `r_kv = 48` (el valor de la super versión, y el que da ×5,33 con
  `n_kv_head=4, d_k=32`), `n_kv_head = 4` (no 6) y `d_k = 64`. Con `r_kv = 64` la
  compresión es algo peor que con 48 (×10,67 → ×10,67 en esta arquitectura) y no
  compensa: **48 es el punto medido**.
* **MoE**: **E = 8, k = 2**, 1 experto compartido (`_dimensiones_ffn_capa` reparte
  `n_hidden` en `E + 2 = 10` unidades, de las que se activan `1 + k = 3`).
* **Antes que nada**: `n_kv_head = 4` en vez de 6 y, si hace falta, `n_layer` más
  bajo. **La palanca más barata es siempre la del GQA, no la MLA.**

### 4.3 · ¿Merece la pena el coste en decode?

* **La MLA**: **sí, pero sólo a partir de 300 M** y sólo si el contexto completo
  es un requisito. Su coste en decode está medido en el Anexo K: **×1,50** y
  **loss +0,028 a 25 pasos** (comprimir la KV cuesta cómputo). Es un coste
  **pequeño** comparado con el beneficio de memoria.
* **El MoE**: **el decode mejora** (bytes leídos 0,377× con E=8, k=2) **pero las
  llamadas a `matmul` suben de 34 a 70**. En la Pi, que es memory-bound con un
  GEMM propio, el balance es favorable; en la GPU hay que medirlo.

### 4.4 · Qué hay que hacer en la GPU (el modelo más grande que cabe en 8 GB con el corpus de 300 M)

| pregunta | respuesta |
|---|---|
| ¿el modelo más grande que cabe en **8 GB de VRAM**? | **700 M en AMP + bf16** (5,65 GB) — pero ver la fila siguiente |
| ¿el modelo más grande que **el corpus de 300 M tokens** permite? | **60 M** (3,93 epochs) o **120 M** si se duplica el corpus (7,98 epochs ⇒ 2 epochs con el corpus duplicado) |
| ¿el modelo más grande que **la Pi 400** puede servir con velocidad usable? | **120 M** (123 ms/token [EST]): a 300 M ya son 0,31 s/token y a 700 M 0,73 s/token |
| **⇒ el siguiente escalón real** | **30 M** (cabe en todo: 0,47 GB de VRAM, 1,97 epochs, 30 ms/token, 75 sesiones simultáneas) |

**El plan, en orden:**

1. **Ahora → 30 M**: subir el preset, reentrenar, re-verificar la paridad (35/35) y
   re-verificar el presupuesto. **Sin portar nada.**
2. **30 M → 60 M**: igual, el corpus da (3,93 epochs).
3. **60 M → 120 M**: **duplicar el corpus** antes. Sigue sin hacer falta la MLA.
4. **120 M → 300 M**: aquí sí, **portar la MLA con su paridad PyTorch** (hunks
   13 y 14 de `GUIA_MERGE.md`, más los `test_162`…`test_166` y `test_177`), y
   decidir el `n_kv_head` final (4, no 6).
5. **300 M → 500-700 M**: **portar el MoE** (E=8, k=2) **con** la GPU
   implementando el mismo despacho disperso y `expert_biases` en el checkpoint;
   y multiplicar el corpus por 5 (300 M) o por 12 (700 M).

### 4.5 · La frase que resume el documento

> **La MLA y el MoE no son una mejora del motor: son un desbloqueo de tamaño.**
> El motor ya mide 16,2 M; con 16,2 M, la KV es el 6,0 % del presupuesto de RAM
> (la MLA ahorraría el 4,9 %) y el decode lee 15,4 MB por token (el MoE no cambia
> eso a esta escala). **Portar MLA y MoE hoy es optimizar el 5 % de un problema
> y pagar el 100 % de la complejidad.**
> Lo que hay que hacer hoy es **más corpus** (el modelo ha visto el **0,25 %**)
> y **mantener la paridad PyTorch**. La MLA entra en el escalón de 300 M; el MoE,
> en el de 500 M.

---

## 5. Apéndice: cómo se midió

### 5.1 · Instrumentación de los bytes leídos

Se instrumentaron las **tres** vías por las que el motor lee un peso:

| vía | cuándo se usa |
|---|---|
| `_matmul_peso` | el GEMM normal: atención, FFN densa, expertos en el camino *batch* |
| `_p_para_lectura` | la ruta **dispersa** de decode (`u_vec @ W`), que **no** pasa por `_matmul_peso` |
| `_forward_moe_expertos` | marca dónde empieza y acaba cada bloque MoE (prefill vs decode) |

El paso de **decode** se aisló como todo lo que ocurre **después** del último
bloque MoE de *prefill* (el motor marca el prefill con `guardar_cache=True` y el
decode con `False`).

### 5.2 · El control: la fórmula de parámetros

La fórmula de parámetros (espejo 1:1 de `TransformerBlock.__init__` y
`TransformerNumPyLLM.__init__`) se validó contra `contar_parametros()` del motor
real:

| escala | params reales | fórmula | Δ |
|---|---|---|---|
| 2 capas, D=64, H=8:4 | 78.720 | 78.720 | **0** |
| 4 capas, D=128, H=8:2 | 889.984 | 889.984 | **0** |
| 6 capas, D=384, H=12:4 | 12.507.904 | 12.507.904 | **0** |
| 8 capas, D=384, H=12:4, **MLA r=48 + MoE E=4 k=2 + compuerta** | 12.040.864 | 12.040.864 | **0** |
| 10 capas, D=576, H=18:6 | 44.882.048 | 44.882.048 | **0** |

### 5.3 · La KV, medida y no estimada

`RingBufferKVCache.bytes_reservados()` **medido** (ventana inicial de 128 tokens,
crecimiento lineal) vs la fórmula `n_layer × (2 × n_kv_head × d_k × 4)` (GQA) y
`n_layer × r_kv × 4` (MLA):

| configuración | bytes/token **medidos** | fórmula | ratio GQA/MLA |
|---|---|---|---|
| 8 capas, 4 KV, d=32, GQA | **1.024** | 1.024 | — |
| 8 capas, 4 KV, d=32, **MLA r=48** | **192** | 192 | **×5,33** |
| 8 capas, 4 KV, d=32, **MLA r=8** | **32** | 32 | **×32,0** |
| 24 capas, 4 KV, d=64, GQA | **2.048** | 2.048 | — |
| 24 capas, 4 KV, d=64, **MLA r=48** | **192** | 192 | **×10,67** |

> El **×5,33** de la super versión sale de `n_layer = 8`, `d_k = 32`,
> `n_kv_head = 4`, `r_kv = 48`: **1.536 B/token** (MLA) frente a **8.192 B/token**
> (GQA 12:4). Con `d_k = 64` (la arquitectura de 300 M) la compresión sube a
> **×10,67**.

### 5.4 · Lo que **no** está medido

| hueco | por qué | cómo se cierra |
|---|---|---|
| **el decode en la Pi 400** | no hay Pi conectada; `PI400.md` §8 lo marca como pendiente | `bench_pi.sh` + `pi400_bench.py` del release |
| **el kernel C/NEON** | no hay compilador ARM64 en esta máquina | `make` en la Pi; `engine_pi400_accel.c` |
| **la VRAM real y el ritmo real en la A100** | hace falta Colab | `gpu/colab_entrenar.py info` |
| **la calidad del modelo a 30 M+** | no se ha entrenado | `gpu/entrenar_gpu.py` + holdout |
| **la paridad PyTorch de la MLA** | la MLA no está en nuestro motor | §3.3 (1) |
