# Agrupamiento de franjas de operación en transporte masivo (aprendizaje no supervisado)

Actividad 4 – Inteligencia Artificial · Corporación Universitaria Iberoamericana
Docente: Sandra Bautista
Integrantes: Johan David Cifuentes Linares, Brayan David Gil Barbosa

## Descripción
El proyecto aplica **K-Means** (aprendizaje no supervisado) a registros de operación de estaciones de un sistema de transporte masivo (caso TransMilenio) para **descubrir perfiles de operación** sin etiquetas previas. Se basa en el capítulo 16 (Técnicas de agrupamiento) de Palma Méndez (2008).

El modelo encuentra **4 grupos**:

| Grupo | Perfil | Registros |
|---|---|---|
| 0 | Hora pico (alta carga) | 153 |
| 1 | Madrugada/noche (baja demanda) | 48 |
| 2 | Valle entre semana | 125 |
| 3 | Fin de semana | 74 |

## Estructura del repositorio
| Archivo | Contenido |
|---|---|
| `datos_estaciones.csv` | Dataset (400 registros, sin etiquetas) |
| `generar_dataset.py` | Script que genera el dataset sintético (semilla 42) |
| `diccionario_datos.md` | Descripción de cada variable |
| `modelo_kmeans.py` | Modelo: escalado, selección de k, entrenamiento, perfiles, PCA y registro nuevo |
| `pruebas_modelo.py` | Pruebas: calidad de datos, tabla de k, estabilidad, comparación con jerárquico, registros nuevos |
| `resultados_pruebas/` | Tablas (CSV) y gráficas generadas por las pruebas |
| `documento_pruebas.pdf` | Documento con las pruebas realizadas y su interpretación |
| `requirements.txt` | Dependencias |

## Fuentes de datos
No se encontró un dataset público con estas variables para el proyecto, por lo que se generó una muestra sintética con `generar_dataset.py` a partir de cuatro comportamientos realistas (los perfiles no se guardan en el CSV). Ver `diccionario_datos.md`.

## Cómo ejecutar
```bash
git clone <URL_DEL_REPOSITORIO>
cd Transport1
python3 -m venv venv && source venv/bin/activate   # en Windows: venv\Scripts\activate
pip install -r requirements.txt

python3 generar_dataset.py     # opcional: regenera datos_estaciones.csv
python3 modelo_kmeans.py       # entrena el modelo; genera codo_silueta.png, grupos_pca.png y datos_estaciones_agrupados.csv
python3 pruebas_modelo.py      # ejecuta las pruebas; guarda resultados en resultados_pruebas/
```

## Resultados principales
- Datos sin nulos ni duplicados; ocupación > 100 % en 65 registros (hacinamiento, se conserva).
- k = 4: silueta 0,514 (mayor salto respecto a k = 3).
- Estabilidad: ARI = 1,0 con seis semillas distintas (n_init = 10).
- Clustering jerárquico: ARI = 0,995 frente a K-Means.
- Registros nuevos: se asignan al grupo esperado; la distancia al centroide permite detectar registros atípicos.

## Limitaciones
El dataset es sintético; los resultados deben validarse con datos reales antes de usarse en operación.