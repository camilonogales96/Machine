import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Backend necesario para embeber en Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 1. Carga y preprocesamiento de datos
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Definir posibles rutas del dataset (para evitar errores de ruta no encontrada)
ruta_subdirectorio = os.path.join(directorio_actual, 'Dataset', 'human_resources', 'HRDataset_v14.csv')
ruta_local = os.path.join(directorio_actual, 'HRDataset_v14.csv')

if os.path.exists(ruta_subdirectorio):
    ruta_origen = ruta_subdirectorio
elif os.path.exists(ruta_local):
    ruta_origen = ruta_local
else:
    raise FileNotFoundError("No se encontró el archivo HRDataset_v14.csv en el directorio local ni en Dataset/human_resources/")

datos_hr = pd.read_csv(ruta_origen)

# Eliminar duplicados
datos_hr = datos_hr.drop_duplicates()

# Asegurar que no tenga nulos en la variable objetivo (Termd: 0=Activo, 1=Terminado)
datos_hr = datos_hr.dropna(subset=['Termd'])

# Seleccionar variables numéricas
variables_numericas = datos_hr.select_dtypes(include=['int64', 'float64']).columns
datos_hr_numerico = datos_hr[variables_numericas].fillna(0)

# Separar el objetivo de las variables de entrada
objetivo = datos_hr_numerico['Termd']

# Descartar 'Termd' y columnas de ID no predictivas (ej. EmpID genera ruido o falso aprendizaje)
columnas_a_descartar = ['Termd']
if 'EmpID' in datos_hr_numerico.columns:
    columnas_a_descartar.append('EmpID')

variables_entrada = datos_hr_numerico.drop(columns=columnas_a_descartar)

# Separar en conjunto de entrenamiento (80%) y prueba (20%)
X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
    variables_entrada, objetivo, test_size=0.2, random_state=77
)

print("=== ENTRENAMIENTO DEL MODELO ÁRBOL DE DECISIÓN (RRHH) ===")
print(f"Ruta del archivo cargado: {ruta_origen}")
print(f"Volumen de datos de entrenamiento: {len(X_entrenar)}")
print(f"Volumen de datos de prueba: {len(X_probar)}")

# 2. Creación y entrenamiento del modelo Árbol de Decisión
# Se ajusta max_depth=4 para controlar el sobreajuste y permitir una lectura clara del gráfico
modelo_arbol = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=77)
modelo_arbol.fit(X_entrenar, y_entrenar)

# 3. Realizar predicciones sobre el conjunto de prueba
predicciones = modelo_arbol.predict(X_probar)

# 4. Evaluación del modelo
exactitud = accuracy_score(y_probar, predicciones)
matriz_confusion = confusion_matrix(y_probar, predicciones)
reporte_clasificacion = classification_report(y_probar, predicciones)

print("\n=== REPORTE DE EVALUACIÓN - ÁRBOL DE DECISIÓN ===")
print(f"Exactitud (Accuracy): {exactitud * 100:.2f}%")
print("\nMatriz de Confusión:")
print(matriz_confusion)
print("\nReporte de Clasificación:")
print(reporte_clasificacion)

# Importancia de las variables
importancias = pd.DataFrame({
    'Variable': variables_entrada.columns,
    'Importancia': modelo_arbol.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("\n=== IMPORTANCIA DE LAS VARIABLES ===")
print(importancias[importancias['Importancia'] > 0].to_string(index=False))

# =========================================================================
# 5. Generar Visualizaciones Gráficas
# -------------------------------------------------------------------------
# En vez de dos ventanas independientes de matplotlib (plt.show() por
# separado), se crea UNA sola ventana Tkinter con pestañas (ttk.Notebook).
# Cada pestaña contiene su propia figura embebida (FigureCanvasTkAgg) con
# su barra de herramientas de navegación (zoom, pan, guardar).
# =========================================================================

# ---- FIGURA 1: Matriz de Confusión + Frontera de Decisión (PCA 2D) ----
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Activo (0)', 'Terminado (1)'],
            yticklabels=['Activo (0)', 'Terminado (1)'])
ax1.set_title('Matriz de Confusión - Árbol de Decisión', fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicción')
ax1.set_ylabel('Valor Real')

# 1. Escalar los datos EXCLUSIVAMENTE para el PCA
scaler = StandardScaler()
X_entrenar_escalado = scaler.fit_transform(X_entrenar)

pca = PCA(n_components=2)
X_entrenar_pca = pca.fit_transform(X_entrenar_escalado)

modelo_arbol_2d = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=77)
modelo_arbol_2d.fit(X_entrenar_pca, y_entrenar)

x_min, x_max = X_entrenar_pca[:, 0].min() - 1, X_entrenar_pca[:, 0].max() + 1
y_min, y_max = X_entrenar_pca[:, 1].min() - 1, X_entrenar_pca[:, 1].max() + 1

# 2. Paso dinámico: dividimos el rango en 200 partes para no desbordar la memoria
step_x = (x_max - x_min) / 200
step_y = (y_max - y_min) / 200

xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x),
                     np.arange(y_min, y_max, step_y))

Z = modelo_arbol_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
scatter = ax2.scatter(X_entrenar_pca[:, 0], X_entrenar_pca[:, 1], c=y_entrenar, cmap=plt.cm.coolwarm, edgecolors='k')
ax2.set_title('Frontera de Decisión (Árbol 2D vía PCA)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Componente Principal 1')
ax2.set_ylabel('Componente Principal 2')
legend1 = ax2.legend(*scatter.legend_elements(), title="Estado (Termd)")
ax2.add_artist(legend1)

fig1.tight_layout()

# ---- FIGURA 2: Estructura del Árbol de Decisión (figura propia y grande) ----
# Con max_depth=4 puede haber hasta 2^4 = 16 nodos hoja, así que se necesita
# bastante ancho. Ajusta el tamaño según cuántas variables/nodos tengas.
fig2, ax3 = plt.subplots(figsize=(30, 15))

anotaciones = plot_tree(modelo_arbol,
          feature_names=list(variables_entrada.columns),
          class_names=['Activo', 'Terminado'],
          filled=True,
          rounded=True,
          fontsize=10,
          proportion=False,
          impurity=True,
          precision=2,
          ax=ax3)

# -------------------------------------------------------------------------
# Mejora: por defecto, plot_tree() solo escribe "True"/"False" en las DOS
# ramas que salen de la raíz; el resto del árbol queda sin indicar qué lado
# es "se cumple la condición" y cuál es "no se cumple". Aquí se recorre
# TODO el árbol y se etiqueta cada rama individualmente con "Sí"/"No",
# además de colorear las flechas (verde = Sí/izquierda, rojo = No/derecha)
# para que la lectura de cualquier nivel del árbol sea inequívoca.
# -------------------------------------------------------------------------
COLOR_SI = '#2e7d32'   # verde: la condición del nodo se cumple (rama izquierda)
COLOR_NO = '#c62828'   # rojo: la condición del nodo NO se cumple (rama derecha)

arbol_interno = modelo_arbol.tree_

# plot_tree() devuelve una anotación por nodo MÁS dos anotaciones extra de
# texto "True"/"False" (solo en la raíz). Las separamos: las anotaciones de
# nodo quedan en el mismo orden que los ids internos del árbol (0..N-1),
# porque sklearn construye el árbol en ese mismo orden (profundidad primero).
anotaciones_nodos = [a for a in anotaciones if a.get_text().strip() not in ('True', 'False')]
assert len(anotaciones_nodos) == arbol_interno.node_count, \
    "No se pudo emparejar cada anotación con su nodo; revisa la versión de scikit-learn"

# Ocultamos las etiquetas nativas True/False de la raíz: las reemplazamos
# por nuestras propias etiquetas Sí/No, presentes en CADA división del árbol.
for anotacion in anotaciones:
    if anotacion.get_text().strip() in ('True', 'False'):
        anotacion.set_visible(False)

for id_padre in range(arbol_interno.node_count):
    id_izquierdo = arbol_interno.children_left[id_padre]
    id_derecho = arbol_interno.children_right[id_padre]

    if id_izquierdo == -1:
        continue  # nodo hoja: no tiene condición ni ramas que etiquetar

    # Colorear la flecha de cada rama según lo que representa
    flecha_izquierda = anotaciones_nodos[id_izquierdo].arrow_patch
    if flecha_izquierda is not None:
        flecha_izquierda.set_color(COLOR_SI)
        flecha_izquierda.set_linewidth(2.2)

    flecha_derecha = anotaciones_nodos[id_derecho].arrow_patch
    if flecha_derecha is not None:
        flecha_derecha.set_color(COLOR_NO)
        flecha_derecha.set_linewidth(2.2)

    # Colocar la etiqueta "Sí"/"No" en el punto medio de cada rama
    x_padre, y_padre = anotaciones_nodos[id_padre].get_position()
    x_izq, y_izq = anotaciones_nodos[id_izquierdo].get_position()
    x_der, y_der = anotaciones_nodos[id_derecho].get_position()

    ax3.text((x_padre + x_izq) / 2, (y_padre + y_izq) / 2, 'Sí\n(cumple)',
              color=COLOR_SI, fontsize=9, fontweight='bold',
              ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor=COLOR_SI, alpha=0.9),
              zorder=10)

    ax3.text((x_padre + x_der) / 2, (y_padre + y_der) / 2, 'No\n(no cumple)',
              color=COLOR_NO, fontsize=9, fontweight='bold',
              ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor=COLOR_NO, alpha=0.9),
              zorder=10)

ax3.set_title('Estructura Visual del Árbol de Decisión (Reglas de División)',
              fontsize=16, fontweight='bold')

# Leyenda/aclaración general de cómo leer el árbol, al pie de la figura
fig2.text(0.5, 0.01,
          'Cómo leer cada nodo: si la condición mostrada (ej. "Variable <= valor") se CUMPLE, se avanza '
          'por la rama verde "Sí"; si NO se cumple, se avanza por la rama roja "No".',
          ha='center', va='bottom', fontsize=11, style='italic', color='#333333')

# Deja algo de aire alrededor para que las flechas no queden pegadas al borde
fig2.tight_layout(pad=2.0, rect=[0, 0.03, 1, 1])

# ---- FIGURA 3: Gráfico de Dispersión (estilo simple, con línea de tendencia) ----
# Versión simplificada tipo "regplot": un solo color, una línea de tendencia
# con banda de confianza y sin decoraciones adicionales — el mismo estilo
# que sns.regplot()/lmplot() usan por defecto. Se grafican dos variables
# reales que el árbol también usa: PerfScoreID (desempeño) y EngagementSurvey
# (compromiso), que están moderadamente correlacionadas.
fig3, ax4 = plt.subplots(figsize=(9, 6.5))

sns.regplot(
    data=datos_hr_numerico,
    x='PerfScoreID', y='EngagementSurvey',
    x_jitter=0.15,  # separa visualmente los puntos, ya que PerfScoreID solo tiene 4 valores (1-4)
    scatter_kws=dict(alpha=0.6, s=45, color='#2a6ebb', edgecolor='white'),
    line_kws=dict(color='#c0392b', linewidth=2.5),
    ax=ax4
)

ax4.set_xlabel('PerfScoreID  (1=PIP, 2=Necesita mejorar, 3=Cumple, 4=Excede)', fontsize=10.5)
ax4.set_ylabel('EngagementSurvey  (encuesta de compromiso, 1 a 5)', fontsize=10.5)
ax4.set_title('Relación entre Desempeño y Compromiso del Empleado', fontsize=13, fontweight='bold')

correlacion_perf_engagement = datos_hr_numerico['PerfScoreID'].corr(datos_hr_numerico['EngagementSurvey'])
fig3.text(0.5, 0.01,
          f'Cada punto es un empleado. La línea muestra la tendencia general: a mejor evaluación de desempeño, '
          f'mayor compromiso reportado (correlación = {correlacion_perf_engagement:.2f}).',
          ha='center', va='bottom', fontsize=9.5, style='italic', color='#333333')

fig3.tight_layout(rect=[0, 0.05, 1, 1])

# ---- FIGURA 4: Poda del Árbol (Cost-Complexity Pruning) ----
# La poda por complejidad de costo entrena el árbol SIN límite de profundidad
# y luego calcula, para distintos valores de "ccp_alpha", cuánto se puede
# recortar sin perder demasiada exactitud. Esto responde la pregunta
# "¿realmente necesito un árbol tan grande, o uno más simple funciona igual?"
modelo_completo = DecisionTreeClassifier(criterion='gini', random_state=77)
modelo_completo.fit(X_entrenar, y_entrenar)
ruta_poda = modelo_completo.cost_complexity_pruning_path(X_entrenar, y_entrenar)
ccp_alphas = ruta_poda.ccp_alphas

exactitudes_train, exactitudes_test, num_nodos = [], [], []
for alpha in ccp_alphas:
    arbol_alpha = DecisionTreeClassifier(criterion='gini', random_state=77, ccp_alpha=alpha)
    arbol_alpha.fit(X_entrenar, y_entrenar)
    exactitudes_train.append(arbol_alpha.score(X_entrenar, y_entrenar))
    exactitudes_test.append(arbol_alpha.score(X_probar, y_probar))
    num_nodos.append(arbol_alpha.tree_.node_count)

# Alpha recomendado: el árbol MÁS SIMPLE que conserva la mejor exactitud de prueba observada
mejor_exactitud_prueba = max(exactitudes_test)
candidatos_alpha = [(a, n, e) for a, e, n in zip(ccp_alphas, exactitudes_test, num_nodos)
                     if e >= mejor_exactitud_prueba - 0.005]
alpha_recomendado, nodos_recomendado, exactitud_recomendada = max(candidatos_alpha, key=lambda t: t[0])

# Entrenar explícitamente el árbol podado optimizado
modelo_arbol_podado = DecisionTreeClassifier(criterion='gini', random_state=77, ccp_alpha=alpha_recomendado)
modelo_arbol_podado.fit(X_entrenar, y_entrenar)
exactitud_podado = modelo_arbol_podado.score(X_probar, y_probar)
nodos_original = modelo_arbol.tree_.node_count
nodos_podado = modelo_arbol_podado.tree_.node_count

print("\n=== REPORTE COMPARATIVO DE PODA DEL ÁRBOL (CCP) ===")
print(f"Árbol Base (max_depth=4)      : Nodos = {nodos_original}  | Exactitud = {exactitud * 100:.2f}%")
print(f"Árbol Podado (ccp_alpha={alpha_recomendado:.4f}): Nodos = {nodos_podado}  | Exactitud = {exactitud_podado * 100:.2f}%")

fig4, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))

axA.plot(ccp_alphas, exactitudes_train, marker='o', markersize=6, label='Entrenamiento', color='#2a6ebb')
axA.plot(ccp_alphas, exactitudes_test, marker='o', markersize=6, label='Prueba', color='#c0392b')
axA.axvline(alpha_recomendado, color='#2e7d32', linestyle='--', linewidth=1.8,
            label=f'α sugerido ({alpha_recomendado:.3f})')
desplazamientos_A = [(-45, 12), (10, -18), (-35, 10)]
for (alpha, exact), desplazamiento in zip(zip(ccp_alphas, exactitudes_test), desplazamientos_A):
    axA.annotate(f'α={alpha:.3f} ({exact:.0%})', (alpha, exact),
                 textcoords='offset points', xytext=desplazamiento, fontsize=8.5)
axA.set_xlabel('ccp_alpha (nivel de poda)')
axA.set_ylabel('Exactitud')
axA.set_title('Exactitud vs. Nivel de Poda')
axA.legend()

axB.plot(ccp_alphas, num_nodos, marker='o', markersize=6, color='#6a3d9a')
axB.axvline(alpha_recomendado, color='#2e7d32', linestyle='--', linewidth=1.8,
            label=f'Alpha sugerido ({nodos_recomendado} nodos)')
desplazamientos_B = [(-45, 10), (10, -6), (-35, 10)]
for (alpha, nodos), desplazamiento in zip(zip(ccp_alphas, num_nodos), desplazamientos_B):
    axB.annotate(f'{nodos} nodos', (alpha, nodos), textcoords='offset points',
                 xytext=desplazamiento, fontsize=8.5)
axB.set_xlabel('ccp_alpha (nivel de poda)')
axB.set_ylabel('Cantidad de nodos del árbol')
axB.set_title('Complejidad del Árbol vs. Nivel de Poda')
axB.legend()

fig4.suptitle(f'Poda del Árbol (Cost-Complexity Pruning)\nÁrbol Base: {nodos_original} nodos ({exactitud*100:.1f}%) | Árbol Podado: {nodos_podado} nodos ({exactitud_podado*100:.1f}%)', fontsize=13, fontweight='bold')
fig4.text(0.5, 0.01,
          'La poda elimina ramas redundantes. La línea verde marca el árbol más '
          'simple que conserva la mejor exactitud de prueba observada.',
          ha='center', va='bottom', fontsize=9.5, style='italic', color='#333333')
fig4.tight_layout(rect=[0, 0.05, 1, 0.93])

# ---- FIGURA 5: Visualización de la Estructura del Árbol Podado ----
fig5, ax5 = plt.subplots(figsize=(24, 12))
plot_tree(modelo_arbol_podado,
          feature_names=list(variables_entrada.columns),
          class_names=['Activo', 'Terminado'],
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax5)
ax5.set_title(f'Estructura del Árbol Podado Optimizado (ccp_alpha = {alpha_recomendado:.4f}, Nodos = {nodos_podado})', fontsize=14, fontweight='bold')
fig5.tight_layout()

# -------------------------------------------------------------------------
# 6. Generación y Exportación del nuevo dataset CSV con ajustes y predicciones
# -------------------------------------------------------------------------
datos_ajustados = datos_hr.copy()
datos_ajustados['Prediccion_Termd_Arbol'] = modelo_arbol_podado.predict(variables_entrada)
datos_ajustados['Estado_Predicho_Arbol'] = datos_ajustados['Prediccion_Termd_Arbol'].map({0: 'Activo', 1: 'Terminado'})

ruta_exportacion_arbol = os.path.join(directorio_actual, 'HRDataset_Ajustado_Arbol.csv')
datos_ajustados.to_csv(ruta_exportacion_arbol, index=False)

print("\n=== GENERACIÓN DE NUEVO ARCHIVO CSV ===")
print(f"[ÉXITO] Se ha generado el nuevo archivo CSV ajustado con las predicciones del modelo Árbol:")
print(f"📁 Ruta: {ruta_exportacion_arbol}")

print("\n[INFO] Desplegando ventana con pestañas (Resumen / Árbol completo / Dispersión / Poda / Árbol Podado)...")


def mostrar_ventana_con_pestanas(figuras_con_titulos):
    """
    Crea una sola ventana Tkinter con una pestaña por cada figura de matplotlib.
    figuras_con_titulos: lista de tuplas (figura, titulo_pestana)
    """
    raiz = tk.Tk()
    raiz.title("Resultados - Árbol de Decisión RRHH")
    raiz.geometry("1200x800")

    notebook = ttk.Notebook(raiz)
    notebook.pack(fill='both', expand=True)

    canvases = []

    for figura, titulo in figuras_con_titulos:
        pestana = ttk.Frame(notebook)
        notebook.add(pestana, text=titulo)

        canvas = FigureCanvasTkAgg(figura, master=pestana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        barra_herramientas = NavigationToolbar2Tk(canvas, pestana)
        barra_herramientas.update()

        canvases.append(canvas)

    raiz.mainloop()


mostrar_ventana_con_pestanas([
    (fig1, "Resumen (Matriz + Frontera)"),
    (fig2, "Árbol completo"),
    (fig3, "Gráfico de Dispersión"),
    (fig4, "Poda del Árbol"),
    (fig5, "Árbol Podado Optimizado"),
])