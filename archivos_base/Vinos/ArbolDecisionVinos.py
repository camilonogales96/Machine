from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
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

X_entrenar_normalizado = normalizador.fit_transform(
    X_entrenar
)

X_probar_normalizado = normalizador.transform(
    X_probar
)

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

print(
    f"\n[ÉXITO] Dataset limpio exportado en: "
    f"{ruta_destino}"
)

print("\n==============================================")
print(" ENTRENAMIENTO DEL ÁRBOL BASE")
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

error_entrenamiento = (
    1 - exactitud_entrenamiento
)

error_generalizacion = (
    1 - exactitud_prueba
)

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

clases = np.sort(
    objetivo.unique()
)

matriz_confusion = confusion_matrix(
    y_probar,
    predicciones_prueba,
    labels=clases
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

    fn = (
        matriz_confusion[i, :].sum()
        - tp
    )

    fp = (
        matriz_confusion[:, i].sum()
        - tp
    )

    tn = (
        matriz_confusion.sum()
        - tp
        - fn
        - fp
    )

    if (tn + fp) != 0:
        especificidad = (
            tn / (tn + fp)
        )
    else:
        especificidad = 0

    especificidades.append(
        especificidad
    )

especificidad_macro = np.mean(
    especificidades
)

print("\n=== VALIDACIÓN DEL ÁRBOL BASE ===")

print(
    f"Exactitud entrenamiento: "
    f"{exactitud_entrenamiento * 100:.2f}%"
)

print(
    f"Exactitud prueba: "
    f"{exactitud_prueba * 100:.2f}%"
)

print(
    f"Error entrenamiento: "
    f"{error_entrenamiento * 100:.2f}%"
)

print(
    f"Error generalización: "
    f"{error_generalizacion * 100:.2f}%"
)

print(
    f"Precisión promedio: "
    f"{precision_macro * 100:.2f}%"
)

print(
    f"Recall promedio: "
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

print("\nMatriz de Confusión:")
print(matriz_confusion)

print("\nReporte de Clasificación:")

print(
    classification_report(
        y_probar,
        predicciones_prueba,
        labels=clases,
        zero_division=0
    )
)

print("\n==============================================")
print(" OPTIMIZACIÓN DEL ÁRBOL")
print("==============================================")

parametros = {

    "criterion": [
        "gini",
        "entropy"
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ]
}

busqueda = GridSearchCV(
    DecisionTreeClassifier(
        random_state=77
    ),
    parametros,
    cv=5,
    scoring="f1_macro",
    n_jobs=1
)

busqueda.fit(
    X_entrenar_normalizado,
    y_entrenar
)

modelo_arbol_optimizado = (
    busqueda.best_estimator_
)

print("\nMejores parámetros:")
print(busqueda.best_params_)

print(
    f"\nMejor F1 macro en validación cruzada: "
    f"{busqueda.best_score_ * 100:.2f}%"
)

predicciones_entrenamiento_opt = (
    modelo_arbol_optimizado.predict(
        X_entrenar_normalizado
    )
)

predicciones_prueba_opt = (
    modelo_arbol_optimizado.predict(
        X_probar_normalizado
    )
)

exactitud_entrenamiento_opt = accuracy_score(
    y_entrenar,
    predicciones_entrenamiento_opt
)

exactitud_prueba_opt = accuracy_score(
    y_probar,
    predicciones_prueba_opt
)

error_entrenamiento_opt = (
    1 - exactitud_entrenamiento_opt
)

error_generalizacion_opt = (
    1 - exactitud_prueba_opt
)

precision_macro_opt = precision_score(
    y_probar,
    predicciones_prueba_opt,
    average="macro",
    zero_division=0
)

sensibilidad_macro_opt = recall_score(
    y_probar,
    predicciones_prueba_opt,
    average="macro",
    zero_division=0
)

f1_macro_opt = f1_score(
    y_probar,
    predicciones_prueba_opt,
    average="macro",
    zero_division=0
)

matriz_confusion_opt = confusion_matrix(
    y_probar,
    predicciones_prueba_opt,
    labels=clases
)

reporte_diccionario_opt = (
    classification_report(
        y_probar,
        predicciones_prueba_opt,
        labels=clases,
        output_dict=True,
        zero_division=0
    )
)

especificidades_opt = []

for i in range(len(clases)):

    tp = matriz_confusion_opt[i, i]

    fn = (
        matriz_confusion_opt[i, :].sum()
        - tp
    )

    fp = (
        matriz_confusion_opt[:, i].sum()
        - tp
    )

    tn = (
        matriz_confusion_opt.sum()
        - tp
        - fn
        - fp
    )

    if (tn + fp) != 0:

        especificidad = (
            tn / (tn + fp)
        )

    else:

        especificidad = 0

    especificidades_opt.append(
        especificidad
    )

especificidad_macro_opt = np.mean(
    especificidades_opt
)

print("\n=== VALIDACIÓN DEL ÁRBOL OPTIMIZADO ===")

print(
    f"Exactitud entrenamiento: "
    f"{exactitud_entrenamiento_opt * 100:.2f}%"
)

print(
    f"Exactitud prueba: "
    f"{exactitud_prueba_opt * 100:.2f}%"
)

print(
    f"Error entrenamiento: "
    f"{error_entrenamiento_opt * 100:.2f}%"
)

print(
    f"Error generalización: "
    f"{error_generalizacion_opt * 100:.2f}%"
)

print(
    f"Precisión promedio: "
    f"{precision_macro_opt * 100:.2f}%"
)

print(
    f"Recall promedio: "
    f"{sensibilidad_macro_opt * 100:.2f}%"
)

print(
    f"Especificidad promedio: "
    f"{especificidad_macro_opt * 100:.2f}%"
)

print(
    f"F1 promedio: "
    f"{f1_macro_opt * 100:.2f}%"
)

print("\nMatriz de Confusión:")
print(matriz_confusion_opt)

print("\nReporte de Clasificación:")

print(
    classification_report(
        y_probar,
        predicciones_prueba_opt,
        labels=clases,
        zero_division=0
    )
)

print("\n==============================================")
print(" COMPARACIÓN GENERAL")
print("==============================================")

print(
    f"Exactitud prueba: "
    f"{exactitud_prueba * 100:.2f}% -> "
    f"{exactitud_prueba_opt * 100:.2f}%"
)

print(
    f"Precisión: "
    f"{precision_macro * 100:.2f}% -> "
    f"{precision_macro_opt * 100:.2f}%"
)

print(
    f"Recall: "
    f"{sensibilidad_macro * 100:.2f}% -> "
    f"{sensibilidad_macro_opt * 100:.2f}%"
)

print(
    f"Especificidad: "
    f"{especificidad_macro * 100:.2f}% -> "
    f"{especificidad_macro_opt * 100:.2f}%"
)

print(
    f"F1: "
    f"{f1_macro * 100:.2f}% -> "
    f"{f1_macro_opt * 100:.2f}%"
)

importancias = pd.DataFrame({

    "Variable":
        variables_entrada.columns,

    "Importancia":
        modelo_arbol.feature_importances_

}).sort_values(
    by="Importancia",
    ascending=False
)

print("\n=== IMPORTANCIA DE VARIABLES ÁRBOL BASE ===")

print(
    importancias[
        importancias["Importancia"] > 0
    ].to_string(index=False)
)

importancias_opt = pd.DataFrame({

    "Variable":
        variables_entrada.columns,

    "Importancia":
        modelo_arbol_optimizado.feature_importances_

}).sort_values(
    by="Importancia",
    ascending=False
)

print(
    "\n=== IMPORTANCIA DE VARIABLES "
    "ÁRBOL OPTIMIZADO ==="
)

print(
    importancias_opt[
        importancias_opt["Importancia"] > 0
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
    "Matriz de Confusión - Árbol Base",
    fontsize=12,
    fontweight="bold"
)

ax1.set_xlabel(
    "Calidad predicha"
)

ax1.set_ylabel(
    "Calidad real"
)

ax1.set_xticks(
    range(len(clases))
)

ax1.set_yticks(
    range(len(clases))
)

ax1.set_xticklabels(
    clases
)

ax1.set_yticklabels(
    clases
)

for i in range(
    matriz_confusion.shape[0]
):

    for j in range(
        matriz_confusion.shape[1]
    ):

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

x_min = (
    X_entrenar_pca[:, 0].min()
    - 1
)

x_max = (
    X_entrenar_pca[:, 0].max()
    + 1
)

y_min = (
    X_entrenar_pca[:, 1].min()
    - 1
)

y_max = (
    X_entrenar_pca[:, 1].max()
    + 1
)

step_x = (
    x_max - x_min
) / 200

step_y = (
    y_max - y_min
) / 200

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

    for indice, clase
    in enumerate(clases)
}

Z_indices = np.vectorize(
    mapa_clases.get
)(Z)

y_indices = np.array([

    mapa_clases[valor]

    for valor
    in y_entrenar
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
    "Frontera de Decisión - Árbol Base",
    fontsize=12,
    fontweight="bold"
)

ax2.set_xlabel(
    "Componente Principal 1"
)

ax2.set_ylabel(
    "Componente Principal 2"
)

handles, _ = (
    scatter.legend_elements()
)

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
        for clase
        in modelo_arbol.classes_
    ],
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=False,
    impurity=True,
    ax=ax3
)

ax3.set_title(
    "Árbol de Decisión Base",
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

    modelo_temporal = (
        DecisionTreeClassifier(
            criterion="gini",
            max_depth=profundidad,
            random_state=77
        )
    )

    modelo_temporal.fit(
        X_entrenar_normalizado,
        y_entrenar
    )

    pred_entrenamiento_temporal = (
        modelo_temporal.predict(
            X_entrenar_normalizado
        )
    )

    pred_prueba_temporal = (
        modelo_temporal.predict(
            X_probar_normalizado
        )
    )

    errores_entrenamiento.append(
        1 - accuracy_score(
            y_entrenar,
            pred_entrenamiento_temporal
        )
    )

    errores_prueba.append(
        1 - accuracy_score(
            y_probar,
            pred_prueba_temporal
        )
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
    label="Profundidad del árbol base = 4"
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

ax4.grid(
    alpha=0.3
)

fig3.tight_layout()

fig4, (ax5, ax6) = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

imagen_opt = ax5.imshow(
    matriz_confusion_opt,
    interpolation="nearest",
    cmap="Blues"
)

fig4.colorbar(
    imagen_opt,
    ax=ax5
)

ax5.set_title(
    "Matriz de Confusión - Árbol Optimizado",
    fontsize=12,
    fontweight="bold"
)

ax5.set_xlabel(
    "Calidad predicha"
)

ax5.set_ylabel(
    "Calidad real"
)

ax5.set_xticks(
    range(len(clases))
)

ax5.set_yticks(
    range(len(clases))
)

ax5.set_xticklabels(
    clases
)

ax5.set_yticklabels(
    clases
)

for i in range(
    matriz_confusion_opt.shape[0]
):

    for j in range(
        matriz_confusion_opt.shape[1]
    ):

        ax5.text(
            j,
            i,
            matriz_confusion_opt[i, j],
            ha="center",
            va="center"
        )

modelo_arbol_optimizado_2d = (
    DecisionTreeClassifier(
        random_state=77,
        **busqueda.best_params_
    )
)

modelo_arbol_optimizado_2d.fit(
    X_entrenar_pca,
    y_entrenar
)

Z_opt = (
    modelo_arbol_optimizado_2d.predict(
        np.c_[
            xx.ravel(),
            yy.ravel()
        ]
    )
)

Z_opt = Z_opt.reshape(
    xx.shape
)

Z_indices_opt = np.vectorize(
    mapa_clases.get
)(Z_opt)

ax6.contourf(
    xx,
    yy,
    Z_indices_opt,
    alpha=0.3,
    cmap=plt.cm.viridis
)

scatter_opt = ax6.scatter(
    X_entrenar_pca[:, 0],
    X_entrenar_pca[:, 1],
    c=y_indices,
    cmap=plt.cm.viridis,
    edgecolors="k",
    s=30
)

ax6.set_title(
    "Frontera de Decisión - Árbol Optimizado",
    fontsize=12,
    fontweight="bold"
)

ax6.set_xlabel(
    "Componente Principal 1"
)

ax6.set_ylabel(
    "Componente Principal 2"
)

handles_opt, _ = (
    scatter_opt.legend_elements()
)

ax6.legend(
    handles_opt,
    [
        f"Calidad {clase}"
        for clase in clases
    ],
    title="Quality"
)

fig4.tight_layout()

fig5, ax7 = plt.subplots(
    figsize=(28, 14)
)

plot_tree(
    modelo_arbol_optimizado,
    feature_names=list(
        variables_entrada.columns
    ),
    class_names=[
        str(clase)
        for clase
        in modelo_arbol_optimizado.classes_
    ],
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=False,
    impurity=True,
    ax=ax7
)

ax7.set_title(
    "Árbol de Decisión Optimizado",
    fontsize=16,
    fontweight="bold"
)

fig5.tight_layout(
    pad=2.0
)


def mostrar_ventana_con_pestanas(
    figuras_con_titulos
):

    raiz = tk.Tk()

    raiz.title(
        "Comparación de Árboles de Decisión - Calidad del Vino"
    )

    raiz.geometry(
        "1250x850"
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
        text="Comparación"
    )

    titulo_validacion = ttk.Label(
        pestana_validacion,
        text="Comparación de Métricas de Desempeño",
        font=("Arial", 18, "bold")
    )

    titulo_validacion.pack(
        pady=15
    )

    tabla_general = ttk.Treeview(
        pestana_validacion,
        columns=(
            "metrica",
            "base",
            "optimizado"
        ),
        show="headings",
        height=9
    )

    tabla_general.heading(
        "metrica",
        text="Métrica"
    )

    tabla_general.heading(
        "base",
        text="Árbol Base"
    )

    tabla_general.heading(
        "optimizado",
        text="Árbol Optimizado"
    )

    tabla_general.column(
        "metrica",
        width=350
    )

    tabla_general.column(
        "base",
        width=200,
        anchor="center"
    )

    tabla_general.column(
        "optimizado",
        width=200,
        anchor="center"
    )

    metricas = [

        (
            "Exactitud de entrenamiento",
            f"{exactitud_entrenamiento * 100:.2f}%",
            f"{exactitud_entrenamiento_opt * 100:.2f}%"
        ),

        (
            "Exactitud de prueba",
            f"{exactitud_prueba * 100:.2f}%",
            f"{exactitud_prueba_opt * 100:.2f}%"
        ),

        (
            "Error de entrenamiento",
            f"{error_entrenamiento * 100:.2f}%",
            f"{error_entrenamiento_opt * 100:.2f}%"
        ),

        (
            "Error de generalización",
            f"{error_generalizacion * 100:.2f}%",
            f"{error_generalizacion_opt * 100:.2f}%"
        ),

        (
            "Precisión promedio",
            f"{precision_macro * 100:.2f}%",
            f"{precision_macro_opt * 100:.2f}%"
        ),

        (
            "Sensibilidad / Recall",
            f"{sensibilidad_macro * 100:.2f}%",
            f"{sensibilidad_macro_opt * 100:.2f}%"
        ),

        (
            "Especificidad promedio",
            f"{especificidad_macro * 100:.2f}%",
            f"{especificidad_macro_opt * 100:.2f}%"
        ),

        (
            "F1 promedio",
            f"{f1_macro * 100:.2f}%",
            f"{f1_macro_opt * 100:.2f}%"
        ),

        (
            "Diferencia de error",
            f"{(error_generalizacion - error_entrenamiento) * 100:.2f} puntos",
            f"{(error_generalizacion_opt - error_entrenamiento_opt) * 100:.2f} puntos"
        )
    ]

    for (
        metrica,
        base,
        optimizado
    ) in metricas:

        tabla_general.insert(
            "",
            "end",
            values=(
                metrica,
                base,
                optimizado
            )
        )

    tabla_general.pack(
        pady=10
    )

    titulo_parametros = ttk.Label(
        pestana_validacion,
        text="Parámetros del Árbol Optimizado",
        font=("Arial", 14, "bold")
    )

    titulo_parametros.pack(
        pady=10
    )

    texto_parametros = tk.Text(
        pestana_validacion,
        height=5,
        width=90
    )

    texto_parametros.insert(
        "1.0",
        f"Mejores parámetros: {busqueda.best_params_}\n"
        f"F1 macro en validación cruzada: "
        f"{busqueda.best_score_ * 100:.2f}%"
    )

    texto_parametros.config(
        state="disabled"
    )

    texto_parametros.pack(
        pady=5
    )

    titulo_clases = ttk.Label(
        pestana_validacion,
        text="Comparación por clase",
        font=("Arial", 14, "bold")
    )

    titulo_clases.pack(
        pady=10
    )

    tabla_clases = ttk.Treeview(
        pestana_validacion,
        columns=(
            "calidad",
            "precision_base",
            "precision_opt",
            "recall_base",
            "recall_opt",
            "f1_base",
            "f1_opt"
        ),
        show="headings",
        height=len(clases)
    )

    encabezados = {
        "calidad": "Calidad",
        "precision_base": "Precisión Base",
        "precision_opt": "Precisión Opt.",
        "recall_base": "Recall Base",
        "recall_opt": "Recall Opt.",
        "f1_base": "F1 Base",
        "f1_opt": "F1 Opt."
    }

    for columna, texto in encabezados.items():

        tabla_clases.heading(
            columna,
            text=texto
        )

        tabla_clases.column(
            columna,
            width=140,
            anchor="center"
        )

    for clase in clases:

        datos_base = (
            reporte_diccionario[
                str(clase)
            ]
        )

        datos_opt = (
            reporte_diccionario_opt[
                str(clase)
            ]
        )

        tabla_clases.insert(
            "",
            "end",
            values=(
                clase,
                f"{datos_base['precision'] * 100:.2f}%",
                f"{datos_opt['precision'] * 100:.2f}%",
                f"{datos_base['recall'] * 100:.2f}%",
                f"{datos_opt['recall'] * 100:.2f}%",
                f"{datos_base['f1-score'] * 100:.2f}%",
                f"{datos_opt['f1-score'] * 100:.2f}%"
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

        barra_herramientas = (
            NavigationToolbar2Tk(
                canvas,
                pestana
            )
        )

        barra_herramientas.update()

        canvases.append(
            canvas
        )

    raiz.mainloop()


print(
    "\n[INFO] Desplegando resultados comparativos..."
)

mostrar_ventana_con_pestanas([

    (
        fig1,
        "Base: Matriz + Frontera"
    ),

    (
        fig4,
        "Optimizado: Matriz + Frontera"
    ),

    (
        fig2,
        "Árbol Base"
    ),

    (
        fig5,
        "Árbol Optimizado"
    ),

    (
        fig3,
        "Complejidad y Error"
    )
])
