"""
Generador del dataset sintetico - Actividad 4 (aprendizaje no supervisado)
Proyecto: Transporte masivo - Agrupamiento de franjas de operacion en estaciones.
El CSV resultante NO incluye etiquetas: los grupos los descubre el algoritmo.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 400

# Perfiles latentes usados solo para simular datos realistas 
perfiles = {
    "pico":      dict(p=0.35, horas=[6, 7, 8, 9, 16, 17, 18, 19], fs=0.0,
                      pas=(1800, 3000), frec=(3, 6),   ocu=(85, 110), esp=(5, 12)),
    "valle":     dict(p=0.30, horas=[10, 11, 12, 13, 14, 15],      fs=0.0,
                      pas=(600, 1200),  frec=(7, 12),  ocu=(40, 65),  esp=(6, 10)),
    "fin_semana": dict(p=0.20, horas=list(range(9, 21)),            fs=1.0,
                      pas=(400, 1000),  frec=(10, 16), ocu=(30, 55),  esp=(8, 15)),
    "madrugada": dict(p=0.15, horas=[4, 5, 20, 21, 22],            fs=0.0,
                      pas=(100, 500),   frec=(14, 19), ocu=(10, 35),  esp=(10, 20)),
}

filas = []
nombres = list(perfiles)
probs = [perfiles[n]["p"] for n in nombres]
for _ in range(N):
    cfg = perfiles[rng.choice(nombres, p=probs)]
    fin_semana = int(cfg["fs"]) if cfg["fs"] == 1.0 else 0
    lluvia = int(rng.random() < (0.45 if cfg["pas"][0] > 1000 else 0.25))
    pas = rng.uniform(*cfg["pas"]) * (1.08 if lluvia else 1.0)
    filas.append({
        "hora_dia": int(rng.choice(cfg["horas"])),
        "es_fin_semana": fin_semana,
        "lluvia": lluvia,
        "pasajeros_hora": int(round(pas)),
        "frecuencia_buses_min": int(round(rng.uniform(*cfg["frec"]))),
        "ocupacion_pct": round(float(rng.uniform(*cfg["ocu"])), 1),
        "tiempo_espera_min": round(float(rng.uniform(*cfg["esp"])), 1),
    })

df = pd.DataFrame(filas)
df.insert(0, "id_registro", range(1, N + 1))
df.to_csv("datos_estaciones.csv", index=False)
print(df.shape)
print(df.describe().round(1).T)
