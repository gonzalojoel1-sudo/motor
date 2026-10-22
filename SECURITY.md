# Política de seguridad

## Modelo de amenaza

Este proyecto ejecuta **código y sale a internet** en nombre del usuario, así que
el diseño asume que:

1. **El modelo no es de fiar**: puede inventar llamadas a herramientas con
   argumentos malformados o peligrosos. Toda validación es **estructural**
   (esquema, tipos, límites), nunca por confianza en el texto generado.
2. **La red no es de fiar**: cualquier URL puede resolver a una IP interna o
   devolver contenido hostil.
3. **El contenido descargado no es de fiar**: se trata como **datos**, nunca como
   instrucciones (defensa contra inyección de prompt).

## Garantías implementadas

| superficie | medida |
|---|---|
| **SSRF** (herramientas web) | resolución DNS + `ipaddress`: se bloquean loopback, RFC1918, link-local, reservadas, multicast, no especificadas y site-local. **Fail-closed**: si no se puede validar la IP, se bloquea. Cubre `169.254.169.254`. |
| **Esquemas de URL** | solo `http`/`https`, puertos `80/443/8080/8443`. Nada de `file:`, `ftp:`, `javascript:`, `data:`. |
| **Contenido web** | nunca se ejecuta JavaScript ni HTML activo; tope de 2 MB y 10 s de timeout; `User-Agent` identificable; `robots.txt`. |
| **Servidor** | las herramientas de red están **apagadas por defecto**: hacen falta `--permitir-web` o `PI400_PERMITIR_WEB=1`. `PI400_SIN_WEB=1` lo apaga todo siempre. |
| **Terminal** | `subprocess` con argv (sin shell), allowlist de binarios, timeout, y bloqueo de operadores (`;`, `|`, `&`, `>`, `$(`, backticks). |
| **Python** | subproceso `python -I` con blocklist AST (23 módulos, 18 nombres) y timeout. |
| **Rutas** | blindaje contra *path traversal*: solo se lee/escribe dentro de las carpetas autorizadas. |
| **Escritura de ficheros** | atómica (`tempfile` + `os.replace`); un `.py` con error de sintaxis **se rechaza**. |
| **Inyección de prompt** | el contenido web se delimita y el prompt del sistema declara que es **dato no ejecutable**. |

## Cómo reportar

Abre un *issue* **privado** (Security → Report a vulnerability) o escribe a los
mantenedores. Incluye:

- versión / commit,
- sistema operativo y arquitectura (x86 o Pi 400),
- reproducción mínima,
- impacto esperado.

No publiques detalles explotables antes de que se corrija.

## Fuera de alcance

- **El modelo no está alineado**: no filtres sus salidas a usuarios finales sin
  supervisión.
- **Datos del corpus**: no se distribuyen en este repositorio; su licencia y su
  tratamiento son responsabilidad de quien los reconstruya.
- **Compilación del kernel NEON**: se valida por lectura y emulación en x86; en la
  Pi hay que compilarlo con `make` y revisarlo.
