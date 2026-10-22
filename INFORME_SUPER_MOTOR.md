# INFORME_SUPER_MOTOR.md — qué se le hizo a la super versión y qué falta

> **Qué es la «super versión».** `super_motor/mi_propio_llm_pi400.py` es el motor
> «esteroides» del usuario (`Downloads\v.0.01 (esteroides)`) **ya corregido**:
> 27.824 líneas, sha256 `f42409bbff793ad2025d74dd2ef148b24d4aa6b66641596c60929eedee8657b1`.
> Es un **superconjunto** del nuestro: tiene MLA, MoE, Gated Attention, Attention
> Sinks, MTP, DAG y orquestador de tools que nuestro motor **no** tiene.
>
> Este informe dice **qué se le corrigió**, **con qué número**, y **qué queda**.
> Todas las cifras salen de `super_motor/tests/medicion_ANTES.json` y
> `medicion_DESPUES.json` (los dos arneses que producing el motor) o de una
> medición hecha el **2026-10-22** en esta máquina y que se indica como tal.

---

## 0. Resumen en una línea

Se le corrigieron **9 defectos** (A→I) y se rehízo la suite de calidad de código
(**N1→N8**). **El motor «esteroides» es ahora más capaz que el nuestro en
arquitectura** (MLA, MoE, compuerta, sinks, MTP, DAG), pero **sigue siendo peor en
calidad de recuperación** (RAG) y **sigue sin la ruta GPU, sin corpus y sin
documentación de despliegue**. Detalle en [`COMPARATIVA.md`](COMPARATIVA.md).

---

## 1. Los 9 arreglos (antes → después)

### A · La máscara JSON rompía la salida → **no superable, se desactiva**

La FSM de enmascarado JSON, en el bucle de generación, **destruía** la salida:

| | con máscara | sin máscara |
|---|---|---|
| JSON válido | **0 %** | **33,3 %** |
| coste | **20,7 ms/token** | **7,9 ms/token** |

Medido con el modelo de referencia (vocab 8.196, 6 prompts). El suavizado del
enmascarado (−1e9 → −6,0) no lo arregla.

**Arreglo**: `MASCARA_JSON_EN_GENERACION = False` (desactivada por defecto) y
**fuera del bucle de generación**. **La clase y sus tests siguen vivos**: el
`test_198` verifica que el flag está apagado y el `test_199` que la clase todavía
sirve (guía el nombre del argumento requerido). Es una decisión honesta: no
se puede tener las dos cosas.

> **Lo que NO se hizo**: no se borró la funcionalidad. Se dejó documentada como
> «no superable con el modelo de referencia», que es un resultado, no un hueco.

### B · El RAG cuántico era **peor** que el base → ahora es **aditivo**

El problema: la fusión RRF + el boost del grafo + el reorden MMR + el «cosido» del
CRAG **descartaban el ranking base**. Medido sobre el banco de **714 consultas /
238 documentos**:

| | hit@1 | hit@3 | hit@5 | MRR | ms/consulta |
|---|---|---|---|---|---|
| RAG base (ANTES) | **0,7129** | 0,8095 | 0,8487 | 0,7637 | 2,22 |
| RAG cuántico (ANTES) | **0,5490** | 0,6429 | 0,6961 | 0,6004 | 10,12 |
| RAG cuántico (DESPUÉS, aditivo) | **0,7129** | 0,8095 | 0,8487 | **0,7648** | **8,53** |

**Arreglo**: el ranking base se conserva **íntegro** y las etapas avanzadas
(GraphRAG, HyDE, Multi-Query, MMR, CRAG) **sólo añaden** candidatos al final.
Resultado: hit@1 **igual** al base y MRR **+0,0011**.

**Lo que queda**: el cuántico **sigue sin mejorar** el resultado y **cuesta 3,84×**
(8,53 ms frente a 2,22 ms). **HyDE y GraphRAG no aportan ninguna mejora medible**
(hit@1/3/5 idénticos con y sin ellas). Se dejan (no se eliminan) y se documenta el
número.

### C · Hueco de seguridad → **cerrado**

8 rutas heredaban `allow_terminal=True` y acababan **`subprocess`**:
orquestador, DAG, `_ejecutar_tool_call_directa`, router, ReAct y agente.
Ahora `allow_terminal=False` por defecto en las 8, y los envoltorios de
herramienta **no pueden reactivarlo**.

**Test**: `test_195` (espía sobre `subprocess.run`: **no se llama nunca** desde
`tool_orquestar_tool_calls` ni de `tool_planificar_y_ejecutar_dag`),
`test_196` (`allow_terminal` es `False` por defecto) y `test_197` (la allowlist y
el timeout se siguen respetando).

### D · El tokenizador divergía del corpus → **arreglado al 100 %**

`RE_UNIDAD_BPE` divergía de la regex GPT-2 de llama.cpp en **9 de 9** contracciones.

**Arreglo**: se porta la **regex GPT-2 de llama.cpp** al motor
(`RE_UNIDAD_BPE`) y `pipeline_datos.py` declara `"pretokenizador": "gpt2"`.
`"C'mon"` → `['"', 'C', "'m", 'on', '"']`.

| | antes | después |
|---|---|---|
| ids coincidentes con el tokenizador **real** del corpus | **59,21 %** | **100 %** |
| ids comparados | — | **35.150** (120.000 caracteres de TinyStories) |
| pre-tokens motor / corpus | 42 / 75 | **76 / 76** |

Tests nuevos: `test_190/191/192`.

### E · El guard del export GGUF → **aborta en vez de perder tensores en silencio**

**Arreglo**: se rastrea cada tensor del motor; los de capacidad (QK-Norm,
compuerta, MTP, MoE, `b2`) salen como **tensores de extensión** y un tensor
**no mapeado aborta** la exportación **sin dejar fichero**.

Medido (`medicion_DESPUES.json` → `N8d_guard_gguf`):

| | antes | después |
|---|---|---|
| export normal | 0 abortos | **OK**, 14.752 B |
| tensor no mapeado | **se perdía en silencio** | **`ValueError`**: «El exportador GGUF no sabe mapear estos tensores del motor: `W_tensor_fantasma`…» |
| ¿deja fichero incompleto? | sí | **no** (`fichero_creado_aunque_faltasen_tensores = False`) |

Tests nuevos: `test_193/194`.

### F · El cargador INT8 → **ya estaba bien** (se dejó como estaba)

Auditoría: **49 tensores INT8**, **12/12 tokens idénticos**, error RMS de
cuantización **2,5951 %**, `argmax` fp32 ≡ INT8, la ventana deslizante INT8 sin
fallos, y `use_qk_norm`/`use_gated_attn` siguen **True** tras cargar.

Lo **único** que se añadió:
* la comprobación de la compuerta mirando en `params` **y** en `int8_params`
  (lo que faltaba: si el flag se leía sólo de `params`, se perdía al cargar un
  checkpoint INT8);
* el accesor **`_p_para_lectura()`** para leer pesos 2-D **sin borrar la copia
  INT8** (`self._p()` materializaba el float32 y destruía la ruta rápida del
  GEMM INT8 vectorizado con NEON).

### G · Los tests de mutación → de **8 de 30** a **13 de 13**

13 mutaciones reales: Ebbinghaus, orden del DAG, `$prev`, factor `lb`/Levenshtein,
factor KB, media, mediana, percentiles, peso de la memoria, persistencia de la
agenda, ciclos Pomodoro, selección TextRank, nombre indefinido.

**`test_178` reescrito**: ahora usa `analizar_nombres_indefinidos()`, que detecta
un nombre inexistente **incluso en código muerto** (antes sólo hacía `ast.parse`).

### H · Los 4 motores útiles → con la fórmula correcta y **datos validados**

| motor | antes | después |
|---|---|---|
| **Ebbinghaus** | dos fórmulas distintas (retención 168 h / 1 acceso = **0,33109** vs la documentada **0,46208**) | **una sola fórmula**: **0,51342** (medido en `medicion_DESPUES.json`, `memoria_retencion_7dias`) |
| **Agenda** | el cuadrante podía no ser Q1 | `proxima_accion_q1` **garantiza Q1**; la persistencia **sobrevive al reinicio** |
| **Conversor** | bandas de colores sin validar; sin dBm, sin energía, sin ley de Ohm | **bandas validadas** (5/5 casos), **dBm ↔ W/mW**, **energía** (1 kWh = 3,6 MJ), **ley de Ohm** (1 Ω con 5 V → 5 A, `I = V / R`) |
| **Analizador** | sin varianza ni IQR; TextRank no contiguo | **varianza 4440,0** e **IQR 10,0** (p25 = 15,0 · p95 = 139,0, n = 11); resumen TextRank **contiguo** |

### I · La documentación y el `pipeline_datos.py` → con **sólo cifras medidas**

* `readme.md` **reescrito con las cifras medidas**: **14 afirmaciones falsas
  corregidas** (166/166 con 1 omitido, la fórmula de Ebbinghaus, las latencias
  reales, dBm, 14 dominios, `makefile.md`, `readme.md`, el pre-tokenizador gpt2,
  y el kernel C marcado como **NO VERIFICADO**).
* `pipeline_datos.py` **carga el motor por ruta** (`importlib` en vez de por
  nombre): con `--run-tests` el módulo es `__main__` y el import por nombre
  fallaba. Esto arregla el `test_179` vía CLI.

---

## 2. La suite de calidad rehecha (N1→N8)

Medido en `medicion_ANTES.json` vs `medicion_DESPUES.json`.

| # | antes | después | cómo se midió |
|---|---|---|---|
| **N1** | reparador AST: **76,92 %** compilaba, **38,46 %** equivalentes, `valido_final` veraz al **92,31 %** | **92,31 %** compilan · **92,31 %** equivalentes · `valido_final` veraz al **100 %** | **13 casos adversariales**; la equivalencia se comprueba sobre el **AST**, que es la fuente de verdad |
| **N2** | 100 / 100 / 83,33 / 40 (la métrica estaba mal medida) | **100 / 100 / 100 / 100** (compila / idempotente / imports / semántica) | 6 casos |
| **N3** | 0 de 9 códigos B; **B403/B404 se escapaban** | **9 de 9**; B403 y B404 detectados; **0 falsos positivos**, **0 falsos negativos** | 11 vulnerabilidades inseguras, 13 LOC efectivas |
| **N4** | 8/8 con `from typing import Iterable` espurio | **8/8**; **0 %** de imports espurios | **45 fragmentos reales** |
| **N5** | **100 %** tautológicos; 0 bugs de lógica detectados | **0 %** tautológicos; **5** tests fallan contra el bug y **9/9** pasan contra el correcto | 9 tests generados, 8 doctests |
| **N6** | **100/100 con la lógica rota** | 100 perfecto · **80** bug lógico · 100 estilo · 10 no compila; **0** con 100 y bug | 4 casos |
| **N7** | **100 %** de tokens que rompen la sintaxis | **0 %** (reducción del **100 %**) | 36 tokens sin máscara → 600 con máscara; **0,00578 ms/token** |
| **N8a** | tokenizador **59,21 %** de coincidencia | **100 %** (35.150 ids) | ver D |
| **N8b** | — | cargador INT8: 17 tensores INT8 + 15 fp32, **error RMS 2,5951 %**, `argmax` coincide | ver F |
| **N8d** | export GGUF silencioso | **aborta** | ver E |

---

## 3. Qué queda (lo que NO se ha resuelto)

| # | Pendiente | Por qué |
|---|---|---|
| 1 | **La FSM JSON en generación** | **No superable** con el modelo de referencia: el JSON válido cae de 33,3 % a 0 % y el coste sube de 7,9 a 20,7 ms/token. Desactivada por defecto (A). |
| 2 | **HyDE y GraphRAG** | **No aportan ninguna mejora medible** en el banco de 714 consultas. Se dejan y se documenta el número (B). |
| 3 | **El kernel C/NEON** | **No verificable en esta máquina** (no hay compilador ARM64): el `readme.md` lo marca como **NO VERIFICADO**. El kernel de nuestro release sí tiene loaders con verificación de arquitectura ELF. |
| 4 | **El analizador de nombres indefinidos** | Es **conservador**: sirve para detectar un nombre inexistente (también en código muerto), pero **no** se usa como puerta de calidad de los módulos completos (falsos positivos en patrones dinámicos). |
| 5 | **El corpus de 300 M, la ruta `gpu/` y las suites del motor de release** | No se han tocado: son territorio de [`GUIA_MERGE.md`](super_motor/GUIA_MERGE.md). |
| 6 | **El DAG con ciclos** | `ordenar_niveles_topologicos` **lanza `ValueError`** ante un ciclo (bien), pero el orquestador de tools reintenta con `args={}`: hay riesgo de reintentos sin argumentos. |
| 7 | **El RAG cuántico no compensa** | Aun siendo aditivo, cuesta **3,84×** por consulta (8,53 vs 2,22 ms) para el mismo hit@1. |

---

## 4. Estado de los tests (verificado el 2026-10-22)

| Suite | Resultado |
|---|---|
| `super_motor/mi_propio_llm_pi400.py --run-tests` | **166/166 OK**, 0 omitidos (gracias a `super_motor/corpus/tokenizador.json`) |
| `super_motor/tests/test_super_v4.py` | **12/12 OK** |
| **Total** | **178 tests, 178 en verde** |

El arnés `super_motor/tests/medir_super_v4.py` es el que produce
`medicion_ANTES.json` y `medicion_DESPUES.json`; `medir_mascara_json.py` y
`medir_rag_cuantico.py` miden A y B por separado. **No forman parte de la suite**
(`--run-tests`), son herramientas de medición.

### El orden de merge, si algún día se quiere portar algo

`super_motor/GUIA_MERGE.md` tiene los **15 hunks** en orden, con la regla de oro:
*no sustituir el fichero entero* (el de los esteroides es un superconjunto), sino
aplicar hunks uno a uno, verificando la suite del release (**152 + 169 + 86**) antes
del siguiente. Los hunks que tocan el **tokenizador** o el **cargador INT8** van
**los últimos**, porque cambian el formato de los ids y el ciclo de vida de los pesos.
El **guard del GGUF** va **el penúltimo**: toca el exportador, que en nuestro
release ya está validado por `tests/gguf_validador.py` y por el `.gguf` de `modelos/`.

---

## 5. Conclusión

La super versión es **un motor con más arquitectura** (MLA, MoE, compuerta,
sinks, MTP, DAG, orquestador de tools) y **con los 9 defectos de esta lista ya
corregidos**. Pero sigue siendo **el motor de otro proyecto**:

* su **RAG pierde 4,76 puntos de hit@1** frente al nuestro (0,7129 vs 0,7605);
* **no trae la ruta GPU** (`gpu/` con equivalencia 35/35), **ni el corpus de 300 M
  tokens**, **ni la documentación de despliegue en la Pi** (`install_pi.sh`,
  `pi400.service`, `bench_pi.sh`, `PI400.md`);
* su `pipeline_datos.py` pesa **9.541 B** y **no tiene pipeline de corpus**: su
  `--descargar` original no descargaba nada.

**Las dos cosas que sí conviven**: la **MLA** y el **MoE** son **reales** en la
super versión (medidos: ×5,33 de compresión de KV y despacho disperso de verdad).
Pero sólo son útiles **a partir de ~300 M de parámetros** — y eso lo explica
[`MLA_MOE_300M_700M.md`](MLA_MOE_300M_700M.md), que es la hoja de ruta del
crecimiento y no una decisión de ahora.
