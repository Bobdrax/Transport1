"""
pruebas_modelo.py
Pruebas del componente de aprendizaje no supervisado (K-Means) - Transporte masivo.
Reproduce el mismo preprocesamiento de modelo_kmeans.py y genera resultados en resultados_pruebas/
Uso: python pruebas_modelo.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

RUTA = "datos_estaciones.csv"
K_MODELO = 4
SEMILLAS = [0, 1, 7, 42, 123, 2024]
SALIDA = "resultados_pruebas"
os.makedirs(SALIDA, exist_ok=True)

df = pd.read_csv(RUTA)
df["es_hora_pico"] = df["hora_dia"].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)
X = df.drop(["id_registro", "hora_dia"], axis=1)
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# 1. Calidad de datos
print("== 1. Calidad de datos ==")
crudo = df.drop(columns=["id_registro"])
q1, q3 = crudo.quantile(0.25), crudo.quantile(0.75)
iqr = q3 - q1
calidad = pd.DataFrame({
    "nulos": crudo.isna().sum(),
    "min": crudo.min(), "max": crudo.max(),
    "media": crudo.mean().round(2), "desv": crudo.std().round(2),
    "outliers_IQR": ((crudo < q1 - 1.5 * iqr) | (crudo > q3 + 1.5 * iqr)).sum(),
})
print(calidad)
print("Filas:", len(df), "| duplicados:", int(df.duplicated().sum()),
      "| ids duplicados:", int(df["id_registro"].duplicated().sum()))
print("Ocupación > 100%:", int((df["ocupacion_pct"] > 100).sum()))
calidad.to_csv(f"{SALIDA}/1_calidad_datos.csv")

# 2. Tabla de k
print("\n== 2. Tabla de k ==")
filas = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xz)
    filas.append({"k": k, "inercia": round(km.inertia_, 1),
                  "silueta": round(silhouette_score(Xz, km.labels_), 3),
                  "davies_bouldin": round(davies_bouldin_score(Xz, km.labels_), 3)})
tabla_k = pd.DataFrame(filas)
print(tabla_k.to_string(index=False))
tabla_k.to_csv(f"{SALIDA}/2_tabla_k.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].plot(tabla_k.k, tabla_k.inercia, "o-"); ax[0].set_title("Inercia (codo)")
ax[1].plot(tabla_k.k, tabla_k.silueta, "s-", color="tab:orange"); ax[1].set_title("Silueta")
ax[2].plot(tabla_k.k, tabla_k.davies_bouldin, "^-", color="tab:green"); ax[2].set_title("Davies-Bouldin (menor es mejor)")
for a in ax: a.set_xlabel("k")
plt.tight_layout(); plt.savefig(f"{SALIDA}/2_tabla_k.png", dpi=150); plt.close()

# 3. Estabilidad con semillas
print("\n== 3. Estabilidad con semillas (k=4) ==")
ref = KMeans(n_clusters=K_MODELO, n_init=10, random_state=42).fit(Xz).labels_
filas = []
for s in SEMILLAS:
    km = KMeans(n_clusters=K_MODELO, n_init=10, random_state=s).fit(Xz)
    filas.append({"semilla": s, "inercia": round(km.inertia_, 1),
                  "silueta": round(silhouette_score(Xz, km.labels_), 3),
                  "ARI_vs_semilla_42": round(adjusted_rand_score(ref, km.labels_), 3)})
est = pd.DataFrame(filas)
print(est.to_string(index=False))
est.to_csv(f"{SALIDA}/3_estabilidad_semillas.csv", index=False)
# estabilidad con n_init=1 (inicialización única, más sensible)
ari1 = [adjusted_rand_score(ref, KMeans(n_clusters=K_MODELO, n_init=1, random_state=s).fit(Xz).labels_) for s in range(30)]
print(f"n_init=1, 30 semillas: ARI medio={np.mean(ari1):.3f} min={np.min(ari1):.3f}")

# 4. Comparación con jerárquico
print("\n== 4. K-Means vs jerárquico (k=4) ==")
filas = [{"metodo": "kmeans", "silueta": round(silhouette_score(Xz, ref), 3),
          "davies_bouldin": round(davies_bouldin_score(Xz, ref), 3), "ARI_vs_kmeans": 1.0}]
etq_h = {}
for link in ["ward", "complete", "average"]:
    h = AgglomerativeClustering(n_clusters=K_MODELO, linkage=link).fit_predict(Xz)
    etq_h[link] = h
    filas.append({"metodo": f"jerarquico_{link}", "silueta": round(silhouette_score(Xz, h), 3),
                  "davies_bouldin": round(davies_bouldin_score(Xz, h), 3),
                  "ARI_vs_kmeans": round(adjusted_rand_score(ref, h), 3)})
comp = pd.DataFrame(filas)
print(comp.to_string(index=False))
comp.to_csv(f"{SALIDA}/4_comparacion_jerarquico.csv", index=False)
print("\nTabla cruzada kmeans vs ward:")
print(pd.crosstab(pd.Series(ref, name="kmeans"), pd.Series(etq_h["ward"], name="ward")))

# Dendrograma (Ward) sobre una muestra
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(Xz, "ward")
plt.figure(figsize=(8, 4))
dendrogram(Z, truncate_mode="lastp", p=20, no_labels=True)
plt.title("Dendrograma (Ward, 20 últimos nodos)"); plt.ylabel("Distancia")
plt.tight_layout(); plt.savefig(f"{SALIDA}/4_dendrograma.png", dpi=150); plt.close()

# 5. Perfil de grupos
print("\n== 5. Perfil de grupos (k=4, semilla 42) ==")
km = KMeans(n_clusters=K_MODELO, n_init=10, random_state=42).fit(Xz)
perfil = X.assign(grupo=km.labels_).groupby("grupo").mean().round(1)
perfil["n"] = pd.Series(km.labels_).value_counts().sort_index()
print(perfil.to_string())
perfil.to_csv(f"{SALIDA}/5_perfil_grupos.csv")

# 6. Registros nuevos
print("\n== 6. Registros nuevos ==")
nombres = {0: "Hora pico (alta carga)", 1: "Madrugada/noche (baja demanda)",
           2: "Valle entre semana", 3: "Fin de semana"}
cols = list(X.columns)
def reg(pico, fs, lluvia, pas, frec, ocu, esp):
    return dict(es_hora_pico=pico, es_fin_semana=fs, lluvia=lluvia, pasajeros_hora=pas,
                frecuencia_buses_min=frec, ocupacion_pct=ocu, tiempo_espera_min=esp)
casos = {
    "A. 7 a.m., lluvia, 2500 pas/h":   reg(1, 0, 1, 2500, 4, 95.0, 8.0),
    "B. 13 h entre semana":            reg(0, 0, 0, 900, 10, 55.0, 8.0),
    "C. Sabado 15 h":                  reg(0, 1, 0, 700, 13, 42.0, 11.0),
    "D. 5 a.m. entre semana":          reg(0, 0, 0, 250, 17, 20.0, 16.0),
    "E. Atipico (extremo)":            reg(1, 0, 1, 6000, 1, 150.0, 40.0),
}
nuevos = pd.DataFrame(casos).T[cols].astype(float)
Zn = scaler.transform(nuevos)
nuevos["grupo"] = km.predict(Zn)
nuevos["perfil"] = nuevos["grupo"].map(nombres)
nuevos["dist_centroide"] = km.transform(Zn).min(axis=1).round(2)
dist_train = km.transform(Xz).min(axis=1)
nuevos["p95_train"] = round(float(np.percentile(dist_train, 95)), 2)
print(nuevos[["grupo", "perfil", "dist_centroide", "p95_train"]].to_string())
nuevos.to_csv(f"{SALIDA}/6_registros_nuevos.csv")

# Figura PCA coloreada
from sklearn.decomposition import PCA
c = PCA(2, random_state=42).fit_transform(Xz)
plt.figure(figsize=(6, 4.5)); plt.scatter(c[:, 0], c[:, 1], c=km.labels_, cmap="viridis", s=18)
plt.title("Grupos K-Means (PCA)"); plt.tight_layout(); plt.savefig(f"{SALIDA}/5_pca_grupos.png", dpi=150)
print("\nListo ->", SALIDA)