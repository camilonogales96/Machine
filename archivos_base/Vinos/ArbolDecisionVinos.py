from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)


base_dir = Path(__file__).resolve().parent

ruta_origen = None

for p in [base_dir] + list(base_dir.parents):
    if (p / "winequality-red.csv").exists():
        ruta_origen = p / "winequality-red.csv"
        break

if not ruta_origen:
    ruta_origen = base_dir.parent / "winequality-red.csv"

datos_vino = pd.read_csv(ruta_origen)

print("=== DATASET ORIGINAL ===")
print("Cantidad de registros:", len(datos_vino))
print("Cantidad de columnas:", len(datos_vino.columns))
print("Valores nulos:", datos_vino.isnull().sum().sum())

datos_vino = datos_vino.drop_duplicates()
datos_vino = datos_vino.dropna()

objetivo = datos_vino["quality"]
variables_entrada = datos_vino.drop("quality", axis=1)

X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
    variables_entrada,
    objetivo,
    test_size=0.20,
    random_state=77
)

normalizador = StandardScaler()

X_entrenar_normalizado = normalizador.fit_transform(X_entrenar)
X_probar_normalizado = normalizador.transform(X_probar)

X_entrenar_normalizado = pd.DataFrame(
    X_entrenar_normalizado,
    columns=variables_entrada.columns,
    index=X_entrenar.index
)

X_probar_normalizado = pd.DataFrame(
    X_probar_normalizado,
    columns=variables_entrada.columns,
    index=X_probar.index
)

datos_normalizados = pd.DataFrame(
    normalizador.transform(variables_entrada),
    columns=variables_entrada.columns,
    index=variables_entrada.index
)

datos_normalizados["quality"] = objetivo.values

ruta_destino = ruta_origen.parent / "dataset_vino_limpio.csv"

datos_normalizados.to_csv(
    ruta_destino,
    index=False
)

print("\n=== REPORTE DE PREPROCESAMIENTO ===")
print("Registros finales limpios:", len(datos_vino))
print("Variables de entrada:", len(variables_entrada.columns))
print("Volumen del Set de Entrenamiento:", len(X_entrenar))
print("Volumen del Set de Prueba:", len(X_probar))
print(
    "Valores nulos después de la limpieza:",
    datos_vino.isnull().sum().sum()
)

print("\n=== VISTA PREVIA DEL DATASET ===")
print(datos_normalizados.head())

print(f"\n[ÉXITO] Dataset limpio exportado en: {ruta_destino}")

print("\n==============================================")
print(" ENTRENAMIENTO DEL MODELO ÁRBOL DE DECISIÓN")
print("==============================================")

modelo_arbol = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=77
)

modelo_arbol.fit(
    X_entrenar_normalizado,
    y_entrenar
)

print("\n[ÉXITO] Modelo entrenado correctamente")

predicciones_entrenamiento = modelo_arbol.predict(
    X_entrenar_normalizado
)

predicciones_prueba = modelo_arbol.predict(
    X_probar_normalizado
)

exactitud_entrenamiento = accuracy_score(
    y_entrenar,
    predicciones_entrenamiento
)

exactitud_prueba = accuracy_score(
    y_probar,
    predicciones_prueba
)

error_entrenamiento = 1 - exactitud_entrenamiento
error_generalizacion = 1 - exactitud_prueba

precision_macro = precision_score(
    y_probar,
    predicciones_prueba,
    average="macro",
    zero_division=0
)

sensibilidad_macro = recall_score(
    y_probar,
    predicciones_prueba,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_probar,
    predicciones_prueba,
    average="macro",
    zero_division=0
)

clases = np.sort(objetivo.unique())

matriz_confusion = confusion_matrix(
    y_probar,
    predicciones_prueba,
    labels=clases
)

reporte_clasificacion = classification_report(
    y_probar,
    predicciones_prueba,
    labels=clases,
    zero_division=0
)

reporte_diccionario = classification_report(
    y_probar,
    predicciones_prueba,
    labels=clases,
    output_dict=True,
    zero_division=0
)

especificidades = []

for i in range(len(clases)):

    tp = matriz_confusion[i, i]
    fn = matriz_confusion[i, :].sum() - tp
    fp = matriz_confusion[:, i].sum() - tp
    tn = matriz_confusion.sum() - tp - fn - fp

    if (tn + fp) != 0:
        especificidad = tn / (tn + fp)
    else:
        especificidad = 0

    especificidades.append(especificidad)

especificidad_macro = np.mean(especificidades)

print("\n==============================================")
print(" VALIDACIÓN DEL MODELO")
print("==============================================")

print(
    f"Exactitud de entrenamiento: "
    f"{exactitud_entrenamiento * 100:.2f}%"
)

print(
    f"Exactitud de prueba: "
    f"{exactitud_prueba * 100:.2f}%"
)

print(
    f"Error de entrenamiento: "
    f"{error_entrenamiento * 100:.2f}%"
)

print(
    f"Error de generalización: "
    f"{error_generalizacion * 100:.2f}%"
)

print(
    f"Precisión promedio: "
    f"{precision_macro * 100:.2f}%"
)

print(
    f"Sensibilidad / Recall promedio: "
    f"{sensibilidad_macro * 100:.2f}%"
)

print(
    f"Especificidad promedio: "
    f"{especificidad_macro * 100:.2f}%"
)

print(
    f"F1 promedio: "
    f"{f1_macro * 100:.2f}%"
)

print(
    f"Diferencia entre error de generalización y entrenamiento: "
    f"{(error_generalizacion - error_entrenamiento) * 100:.2f} puntos porcentuales"
)

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(matriz_confusion)

print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(reporte_clasificacion)

print("\n=== VALIDACIÓN POR CLASE ===")

for i, clase in enumerate(clases):

    datos_clase = reporte_diccionario[str(clase)]

    print(f"\nCalidad {clase}")

    print(
        f"Precisión: "
        f"{datos_clase['precision'] * 100:.2f}%"
    )

    print(
        f"Sensibilidad / Recall: "
        f"{datos_clase['recall'] * 100:.2f}%"
    )

    print(
        f"Especificidad: "
        f"{especificidades[i] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{datos_clase['f1-score'] * 100:.2f}%"
    )

    print(
        f"Muestras: "
        f"{int(datos_clase['support'])}"
    )

importancias = pd.DataFrame({
    "Variable": variables_entrada.columns,
    "Importancia": modelo_arbol.feature_importances_
}).sort_values(
    by="Importancia",
    ascending=False
)

print("\n=== IMPORTANCIA DE LAS VARIABLES ===")

print(
    importancias[
        importancias["Importancia"] > 0
    ].to_string(index=False)
)

fig1, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

imagen = ax1.imshow(
    matriz_confusion,
    interpolation="nearest",
    cmap="Blues"
)

fig1.colorbar(
    imagen,
    ax=ax1
)

ax1.set_title(
    "Matriz de Confusión - Árbol de Decisión",
    fontsize=12,
    fontweight="bold"
)

ax1.set_xlabel("Calidad predicha")
ax1.set_ylabel("Calidad real")

ax1.set_xticks(
    range(len(clases))
)

ax1.set_yticks(
    range(len(clases))
)

ax1.set_xticklabels(clases)
ax1.set_yticklabels(clases)

for i in range(matriz_confusion.shape[0]):

    for j in range(matriz_confusion.shape[1]):

        ax1.text(
            j,
            i,
            matriz_confusion[i, j],
            ha="center",
            va="center"
        )

pca = PCA(
    n_components=2
)

X_entrenar_pca = pca.fit_transform(
    X_entrenar_normalizado
)

modelo_arbol_2d = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=77
)

modelo_arbol_2d.fit(
    X_entrenar_pca,
    y_entrenar
)

x_min = X_entrenar_pca[:, 0].min() - 1
x_max = X_entrenar_pca[:, 0].max() + 1

y_min = X_entrenar_pca[:, 1].min() - 1
y_max = X_entrenar_pca[:, 1].max() + 1

step_x = (x_max - x_min) / 200
step_y = (y_max - y_min) / 200

xx, yy = np.meshgrid(
    np.arange(
        x_min,
        x_max,
        step_x
    ),
    np.arange(
        y_min,
        y_max,
        step_y
    )
)

Z = modelo_arbol_2d.predict(
    np.c_[
        xx.ravel(),
        yy.ravel()
    ]
)

Z = Z.reshape(
    xx.shape
)

mapa_clases = {
    clase: indice
    for indice, clase in enumerate(clases)
}

Z_indices = np.vectorize(
    mapa_clases.get
)(Z)

y_indices = np.array([
    mapa_clases[valor]
    for valor in y_entrenar
])

ax2.contourf(
    xx,
    yy,
    Z_indices,
    alpha=0.3,
    cmap=plt.cm.viridis
)

scatter = ax2.scatter(
    X_entrenar_pca[:, 0],
    X_entrenar_pca[:, 1],
    c=y_indices,
    cmap=plt.cm.viridis,
    edgecolors="k",
    s=30
)

ax2.set_title(
    "Frontera de Decisión (Árbol 2D vía PCA)",
    fontsize=12,
    fontweight="bold"
)

ax2.set_xlabel(
    "Componente Principal 1"
)

ax2.set_ylabel(
    "Componente Principal 2"
)

handles, _ = scatter.legend_elements()

ax2.legend(
    handles,
    [
        f"Calidad {clase}"
        for clase in clases
    ],
    title="Quality"
)

fig1.tight_layout()

fig2, ax3 = plt.subplots(
    figsize=(28, 14)
)

plot_tree(
    modelo_arbol,
    feature_names=list(
        variables_entrada.columns
    ),
    class_names=[
        str(clase)
        for clase in modelo_arbol.classes_
    ],
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=False,
    impurity=True,
    ax=ax3
)

ax3.set_title(
    "Estructura Visual del Árbol de Decisión - Calidad del Vino",
    fontsize=16,
    fontweight="bold"
)

fig2.tight_layout(
    pad=2.0
)

profundidades = list(
    range(1, 16)
)

errores_entrenamiento = []
errores_prueba = []

for profundidad in profundidades:

    modelo_temporal = DecisionTreeClassifier(
        criterion="gini",
        max_depth=profundidad,
        random_state=77
    )

    modelo_temporal.fit(
        X_entrenar_normalizado,
        y_entrenar
    )

    pred_entrenamiento_temporal = modelo_temporal.predict(
        X_entrenar_normalizado
    )

    pred_prueba_temporal = modelo_temporal.predict(
        X_probar_normalizado
    )

    exactitud_entrenamiento_temporal = accuracy_score(
        y_entrenar,
        pred_entrenamiento_temporal
    )

    exactitud_prueba_temporal = accuracy_score(
        y_probar,
        pred_prueba_temporal
    )

    errores_entrenamiento.append(
        1 - exactitud_entrenamiento_temporal
    )

    errores_prueba.append(
        1 - exactitud_prueba_temporal
    )

fig3, ax4 = plt.subplots(
    figsize=(10, 6)
)

ax4.plot(
    profundidades,
    errores_entrenamiento,
    marker="o",
    label="Error de entrenamiento"
)

ax4.plot(
    profundidades,
    errores_prueba,
    marker="o",
    label="Error de prueba"
)

ax4.axvline(
    x=4,
    linestyle="--",
    label="Profundidad utilizada = 4"
)

ax4.set_title(
    "Complejidad y Error del Modelo",
    fontsize=14,
    fontweight="bold"
)

ax4.set_xlabel(
    "Profundidad máxima del árbol"
)

ax4.set_ylabel(
    "Tasa de error"
)

ax4.set_xticks(
    profundidades
)

ax4.legend()
ax4.grid(alpha=0.3)

fig3.tight_layout()


def mostrar_ventana_con_pestanas(figuras_con_titulos):

    raiz = tk.Tk()

    raiz.title(
        "Resultados y Validación - Árbol de Decisión Calidad del Vino"
    )

    raiz.geometry(
        "1200x800"
    )

    notebook = ttk.Notebook(
        raiz
    )

    notebook.pack(
        fill="both",
        expand=True
    )

    pestana_validacion = ttk.Frame(
        notebook
    )

    notebook.add(
        pestana_validacion,
        text="Validación"
    )

    titulo_validacion = ttk.Label(
        pestana_validacion,
        text="Validación del Modelo",
        font=("Arial", 18, "bold")
    )

    titulo_validacion.pack(
        pady=15
    )

    tabla_general = ttk.Treeview(
        pestana_validacion,
        columns=("metrica", "valor"),
        show="headings",
        height=9
    )

    tabla_general.heading(
        "metrica",
        text="Métrica"
    )

    tabla_general.heading(
        "valor",
        text="Resultado"
    )

    tabla_general.column(
        "metrica",
        width=400
    )

    tabla_general.column(
        "valor",
        width=200,
        anchor="center"
    )

    metricas = [
        (
            "Exactitud de entrenamiento",
            f"{exactitud_entrenamiento * 100:.2f}%"
        ),
        (
            "Exactitud de prueba",
            f"{exactitud_prueba * 100:.2f}%"
        ),
        (
            "Error de entrenamiento",
            f"{error_entrenamiento * 100:.2f}%"
        ),
        (
            "Error de generalización",
            f"{error_generalizacion * 100:.2f}%"
        ),
        (
            "Precisión promedio",
            f"{precision_macro * 100:.2f}%"
        ),
        (
            "Sensibilidad / Recall promedio",
            f"{sensibilidad_macro * 100:.2f}%"
        ),
        (
            "Especificidad promedio",
            f"{especificidad_macro * 100:.2f}%"
        ),
        (
            "F1 promedio",
            f"{f1_macro * 100:.2f}%"
        ),
        (
            "Diferencia de error",
            f"{(error_generalizacion - error_entrenamiento) * 100:.2f} puntos"
        )
    ]

    for metrica, valor in metricas:

        tabla_general.insert(
            "",
            "end",
            values=(
                metrica,
                valor
            )
        )

    tabla_general.pack(
        pady=10
    )

    titulo_clases = ttk.Label(
        pestana_validacion,
        text="Validación por clase",
        font=("Arial", 14, "bold")
    )

    titulo_clases.pack(
        pady=10
    )

    tabla_clases = ttk.Treeview(
        pestana_validacion,
        columns=(
            "calidad",
            "precision",
            "sensibilidad",
            "especificidad",
            "f1",
            "muestras"
        ),
        show="headings",
        height=len(clases)
    )

    tabla_clases.heading(
        "calidad",
        text="Calidad"
    )

    tabla_clases.heading(
        "precision",
        text="Precisión"
    )

    tabla_clases.heading(
        "sensibilidad",
        text="Sensibilidad"
    )

    tabla_clases.heading(
        "especificidad",
        text="Especificidad"
    )

    tabla_clases.heading(
        "f1",
        text="F1"
    )

    tabla_clases.heading(
        "muestras",
        text="Muestras"
    )

    for columna in (
        "calidad",
        "precision",
        "sensibilidad",
        "especificidad",
        "f1",
        "muestras"
    ):

        tabla_clases.column(
            columna,
            width=130,
            anchor="center"
        )

    for i, clase in enumerate(clases):

        datos_clase = reporte_diccionario[
            str(clase)
        ]

        tabla_clases.insert(
            "",
            "end",
            values=(
                clase,
                f"{datos_clase['precision'] * 100:.2f}%",
                f"{datos_clase['recall'] * 100:.2f}%",
                f"{especificidades[i] * 100:.2f}%",
                f"{datos_clase['f1-score'] * 100:.2f}%",
                int(datos_clase["support"])
            )
        )

    tabla_clases.pack(
        fill="x",
        padx=20,
        pady=10
    )

    canvases = []

    for figura, titulo in figuras_con_titulos:

        pestana = ttk.Frame(
            notebook
        )

        notebook.add(
            pestana,
            text=titulo
        )

        canvas = FigureCanvasTkAgg(
            figura,
            master=pestana
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        barra_herramientas = NavigationToolbar2Tk(
            canvas,
            pestana
        )

        barra_herramientas.update()

        canvases.append(
            canvas
        )

    raiz.mainloop()


print(
    "\n[INFO] Desplegando ventana de resultados..."
)

mostrar_ventana_con_pestanas([
    (
        fig1,
        "Resumen (Matriz + Frontera)"
    ),
    (
        fig2,
        "Árbol completo"
    ),
    (
        fig3,
        "Complejidad y Error"
    )
])