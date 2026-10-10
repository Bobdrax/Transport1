# DICCIONARIO DE DATOS — Proyecto Transporte Masivo (Actividad 4)
## Dataset: datos_estaciones.csv (400 registros)

**Contexto:** Registros simulados de franjas de operacion en estaciones de un
sistema de transporte masivo. El objetivo es **descubrir perfiles de operacion**
mediante agrupamiento (K-Means), aprendizaje **no supervisado**: el dataset no
contiene variable objetivo ni etiquetas.

| Variable | Tipo | Valores | Descripcion |
|---|---|---|---|
| id_registro | Numerica (entero) | 1 - 400 | Identificador del registro (no se usa en el modelo) |
| hora_dia | Numerica (entero) | 4 - 22 | Hora del dia del registro (se transforma en es_hora_pico) |
| es_fin_semana | Numerica (binaria) | 0, 1 | 1 si el registro corresponde a sabado o domingo |
| lluvia | Numerica (binaria) | 0, 1 | 1 si hubo lluvia en la franja |
| pasajeros_hora | Numerica (entero) | ~100 - 3200 | Pasajeros que ingresan a la estacion por hora |
| frecuencia_buses_min | Numerica (entero) | 3 - 19 | Minutos promedio entre buses |
| ocupacion_pct | Numerica (decimal) | ~10 - 110 | Porcentaje de ocupacion de los buses (puede superar 100 por hacinamiento) |
| tiempo_espera_min | Numerica (decimal) | ~5 - 20 | Minutos promedio de espera de un pasajero |

## Variable derivada (en el codigo)
- **es_hora_pico** (binaria): 1 si hora_dia esta en 6-9 o 16-19. Se usa en lugar de
  hora_dia porque la hora es una variable ciclica y K-Means usa distancias.

## Notas
- Problema de **agrupamiento (clustering)**: no hay etiqueta que predecir.
- Variables usadas por el modelo: es_fin_semana, lluvia, pasajeros_hora,
  frecuencia_buses_min, ocupacion_pct, tiempo_espera_min, es_hora_pico (7 variables,
  todas numericas; se escalan con StandardScaler antes de entrenar).
- Datos sinteticos generados con `generar_dataset.py` (semilla 42), a partir de
  cuatro comportamientos realistas: hora pico, valle entre semana, fin de semana
  y madrugada/noche. Esos perfiles NO se guardan en el CSV.
- No se encontro un dataset publico con estas variables para el caso del proyecto,
  por lo que se genero una muestra sintetica.

## Referencia teorica
Palma Mendez, J. T. (2008). Inteligencia artificial: metodos, tecnicas y
aplicaciones. Capitulo 16: Tecnicas de agrupamiento. Madrid: McGraw-Hill Espana.
