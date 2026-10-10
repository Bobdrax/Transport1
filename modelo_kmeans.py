"""
Modelo de aprendizaje NO supervisado: agrupamiento K-Means
Proyecto: Transporte masivo - Perfiles de operacion de estaciones
Basado en el capitulo 16 de Palma Mendez (2008): tecnicas de agrupamiento
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # guarda las graficas sin abrir ventanas 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

# 1. Cargar datos 
df = pd.read_csv('datos_estaciones.csv')

# Ingenieria de caracteristicas: la hora es ciclica (las 22 h estan "cerca" de las 4 h),
# por lo que se reemplaza por una variable binaria mas util para agrupar.
df['es_hora_pico'] = df['hora_dia'].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)
X = df.drop(['id_registro', 'hora_dia'], axis=1)   # el id no es una caracteristica

# 2. Escalado: K-Means usa distancias, las variables deben tener la misma escala
escalador = StandardScaler()
X_esc = escalador.fit_transform(X)

# 3. Elegir el numero de grupos k 
ks = range(2, 9)
inercias, siluetas = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_esc)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_esc, km.labels_))
    print(f"k={k}  inercia={km.inertia_:.1f}  silueta={siluetas[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inercias, 'o-'); ax[0].set_title('Metodo del codo')
ax[0].set_xlabel('k'); ax[0].set_ylabel('Inercia')
ax[1].plot(list(ks), siluetas, 's-', color='tab:orange'); ax[1].set_title('Coeficiente de silueta')
ax[1].set_xlabel('k'); ax[1].set_ylabel('Silueta')
plt.tight_layout(); plt.savefig('codo_silueta.png', dpi=150)

# Criterio: se elige k=4. La silueta da un salto claro de k=3 a k=4 y despues solo
# mejora marginalmente, mientras la inercia deja de bajar bruscamente (codo).
# Con k>4 los grupos se fragmentan y dejan de ser interpretables.
k_optimo = 4
print("\nk elegido:", k_optimo)

# 4. Entrenamiento final
modelo = KMeans(n_clusters=k_optimo, n_init=10, random_state=42)
df['grupo'] = modelo.fit_predict(X_esc)
print("Silueta final:", round(silhouette_score(X_esc, df['grupo']), 3))

# 5. Interpretacion: perfil promedio de cada grupo
print("\nTamano de cada grupo:\n", df['grupo'].value_counts().sort_index())
perfil = df.drop(['id_registro', 'hora_dia'], axis=1).groupby('grupo').mean().round(1)
print("\nPERFIL PROMEDIO POR GRUPO:\n", perfil)

# Nombres interpretativos 
nombres = {0: 'Hora pico (alta carga)', 1: 'Madrugada/noche (baja demanda)',
           2: 'Valle entre semana', 3: 'Fin de semana'}
df['perfil'] = df['grupo'].map(nombres)
print("\nPERFILES:\n", df['perfil'].value_counts())

# 6. Visualizacion en 2D con PCA
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_esc)
plt.figure(figsize=(7, 5))
plt.scatter(coords[:, 0], coords[:, 1], c=df['grupo'], cmap='viridis', s=25)
plt.title('Grupos de franjas de operacion (proyeccion PCA)')
plt.xlabel(f'Componente 1 ({pca.explained_variance_ratio_[0]*100:.0f}% varianza)')
plt.ylabel(f'Componente 2 ({pca.explained_variance_ratio_[1]*100:.0f}% varianza)')
plt.colorbar(label='Grupo'); plt.tight_layout()
plt.savefig('grupos_pca.png', dpi=150)

# 7. Guardar resultado con el grupo asignado
df.to_csv('datos_estaciones_agrupados.csv', index=False)

# 8. Asignar un registro nuevo a un grupo
nuevo = pd.DataFrame({'es_hora_pico': [1], 'es_fin_semana': [0], 'lluvia': [1],
                      'pasajeros_hora': [2500], 'frecuencia_buses_min': [4],
                      'ocupacion_pct': [95.0], 'tiempo_espera_min': [8.0]})
g = modelo.predict(escalador.transform(nuevo[X.columns]))[0]
print("\nRegistro nuevo (7 a.m., lluvia, 2500 pasajeros/h) -> grupo:", g, '-', nombres[g])
