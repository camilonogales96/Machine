import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def cargar_datos_vinos_arbol():
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    curr = directorio_actual
    ruta_origen = None

    while curr:
        candidato = os.path.join(curr, 'redwine', 'winequality-red.csv')
        candidato2 = os.path.join(curr, 'winequality-red.csv')
        if os.path.exists(candidato):
            ruta_origen = candidato
            break
        elif os.path.exists(candidato2):
            ruta_origen = candidato2
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    if not ruta_origen:
        ruta_origen = os.path.join(directorio_actual, 'winequality-red.csv')

    if not os.path.exists(ruta_origen):
        return None

    datos_vino = pd.read_csv(ruta_origen).drop_duplicates().dropna()
    return datos_vino

def crear_interfaz_vinos_arbol(parent_widget):
    datos_vino = cargar_datos_vinos_arbol()
    if datos_vino is None:
        lbl = ttk.Label(parent_widget, text="No se encontró winequality-red.csv", font=('Segoe UI', 12, 'bold'), foreground='red')
        lbl.pack(pady=20)
        return

    objetivo = datos_vino["quality"]
    variables_entrada = datos_vino.drop("quality", axis=1)

    X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
        variables_entrada, objetivo, test_size=0.20, random_state=77
    )

    normalizador = StandardScaler()
    X_entrenar_normalizado = normalizador.fit_transform(X_entrenar)
    X_probar_normalizado = normalizador.transform(X_probar)

    modelo_arbol = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=77)
    modelo_arbol.fit(X_entrenar_normalizado, y_entrenar)

    predicciones = modelo_arbol.predict(X_probar_normalizado)
    exactitud = accuracy_score(y_probar, predicciones)
    matriz_confusion = confusion_matrix(y_probar, predicciones)
    clases = sorted(y_probar.unique())

    reporte_diccionario = classification_report(y_probar, predicciones, output_dict=True, zero_division=0)

    especificidades = []
    for i, clase in enumerate(clases):
        matriz_binaria = np.zeros((2, 2), dtype=int)
        matriz_binaria[0, 0] = matriz_confusion[i, i]
        matriz_binaria[0, 1] = matriz_confusion[i, :].sum() - matriz_confusion[i, i]
        matriz_binaria[1, 0] = matriz_confusion[:, i].sum() - matriz_confusion[i, i]
        matriz_binaria[1, 1] = matriz_confusion.sum() - (matriz_binaria[0, 0] + matriz_binaria[0, 1] + matriz_binaria[1, 0])
        tn = matriz_binaria[1, 1]
        fp = matriz_binaria[1, 0]
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        especificidades.append(especificidad)


    fig1 = mfigure.Figure(figsize=(11, 5))
    canvas1_tmp = FigureCanvasTkAgg(fig1, master=parent_widget)
    canvas1_tmp.draw()

    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)

    im1 = ax1.imshow(matriz_confusion, cmap="Blues")
    ax1.set_title("Matriz de Confusión - Árbol Vinos", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(len(clases)))
    ax1.set_yticks(range(len(clases)))
    ax1.set_xticklabels(clases)
    ax1.set_yticklabels(clases)
    ax1.set_xlabel("Predicción")
    ax1.set_ylabel("Valor Real")

    for i in range(len(clases)):
        for j in range(len(clases)):
            ax1.text(j, i, str(matriz_confusion[i, j]), ha="center", va="center", color="black")

    pca = PCA(n_components=2)
    X_entrenar_pca = pca.fit_transform(X_entrenar_normalizado)
    modelo_arbol_2d = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=77)
    modelo_arbol_2d.fit(X_entrenar_pca, y_entrenar)

    x_min, x_max = X_entrenar_pca[:, 0].min() - 1, X_entrenar_pca[:, 0].max() + 1
    y_min, y_max = X_entrenar_pca[:, 1].min() - 1, X_entrenar_pca[:, 1].max() + 1
    step_x = (x_max - x_min) / 150
    step_y = (y_max - y_min) / 150

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x), np.arange(y_min, y_max, step_y))
    Z = modelo_arbol_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax2.scatter(X_entrenar_pca[:, 0], X_entrenar_pca[:, 1], c=y_entrenar, cmap=plt.cm.coolwarm, edgecolors="k", s=30)
    ax2.set_title("Frontera de Decisión (Árbol 2D PCA)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Componente Principal 1")
    ax2.set_ylabel("Componente Principal 2")
    fig1.tight_layout()


    fig2 = mfigure.Figure(figsize=(16, 8))
    canvas2_tmp = FigureCanvasTkAgg(fig2, master=parent_widget)
    canvas2_tmp.draw()

    ax3 = fig2.add_subplot(111)
    plot_tree(modelo_arbol, feature_names=list(variables_entrada.columns), class_names=[str(c) for c in clases],
              filled=True, rounded=True, fontsize=8, ax=ax3)
    ax3.set_title("Estructura del Árbol de Decisión (Calidad del Vino)", fontsize=13, fontweight="bold")
    fig2.tight_layout()


    modelo_completo = DecisionTreeClassifier(criterion="gini", random_state=77)
    modelo_completo.fit(X_entrenar_normalizado, y_entrenar)
    ruta_poda = modelo_completo.cost_complexity_pruning_path(X_entrenar_normalizado, y_entrenar)
    ccp_alphas = ruta_poda.ccp_alphas

    exactitudes_train, exactitudes_test, num_nodos = [], [], []
    for alpha in ccp_alphas:
        arbol_alpha = DecisionTreeClassifier(criterion="gini", random_state=77, ccp_alpha=alpha)
        arbol_alpha.fit(X_entrenar_normalizado, y_entrenar)
        exactitudes_train.append(arbol_alpha.score(X_entrenar_normalizado, y_entrenar))
        exactitudes_test.append(arbol_alpha.score(X_probar_normalizado, y_probar))
        num_nodos.append(arbol_alpha.tree_.node_count)

    mejor_exactitud = max(exactitudes_test)
    candidatos_alpha = [(a, n, e) for a, e, n in zip(ccp_alphas, exactitudes_test, num_nodos) if e >= mejor_exactitud - 0.005]
    alpha_rec, nodos_rec, _ = max(candidatos_alpha, key=lambda t: t[0])

    modelo_podado = DecisionTreeClassifier(criterion="gini", random_state=77, ccp_alpha=alpha_rec)
    modelo_podado.fit(X_entrenar_normalizado, y_entrenar)

    fig3 = mfigure.Figure(figsize=(12, 5.5))
    canvas3_tmp = FigureCanvasTkAgg(fig3, master=parent_widget)
    canvas3_tmp.draw()

    axA = fig3.add_subplot(1, 2, 1)
    axB = fig3.add_subplot(1, 2, 2)

    axA.plot(ccp_alphas, exactitudes_train, marker="o", markersize=5, label="Train", color="#2a6ebb")
    axA.plot(ccp_alphas, exactitudes_test, marker="o", markersize=5, label="Test", color="#c0392b")
    axA.axvline(alpha_rec, color="#2e7d32", linestyle="--", label=f"α sugerido ({alpha_rec:.3f})")
    axA.set_xlabel("ccp_alpha")
    axA.set_ylabel("Exactitud")
    axA.set_title("Exactitud vs. Nivel de Poda")
    axA.legend()

    axB.plot(ccp_alphas, num_nodos, marker="o", markersize=5, color="#6a3d9a")
    axB.axvline(alpha_rec, color="#2e7d32", linestyle="--", label=f"Nodos sugeridos ({nodos_rec})")
    axB.set_xlabel("ccp_alpha")
    axB.set_ylabel("Cantidad de Nodos")
    axB.set_title("Nodos vs. Nivel de Poda")
    axB.legend()
    fig3.tight_layout()


    fig4 = mfigure.Figure(figsize=(16, 8))
    canvas4_tmp = FigureCanvasTkAgg(fig4, master=parent_widget)
    canvas4_tmp.draw()

    ax4 = fig4.add_subplot(111)
    plot_tree(modelo_podado, feature_names=list(variables_entrada.columns), class_names=[str(c) for c in clases],
              filled=True, rounded=True, fontsize=8, ax=ax4)
    ax4.set_title(f"Estructura del Árbol Podado Optimizado (ccp_alpha={alpha_rec:.4f})", fontsize=12, fontweight="bold")
    fig4.tight_layout()


    frame_tabla = ttk.LabelFrame(parent_widget, text=f" Reporte Multiclase por Calidad (Exactitud Global: {exactitud*100:.2f}%) ")
    frame_tabla.pack(fill="x", padx=10, pady=5)

    columnas_tabla = ("calidad", "precision", "sensibilidad", "especificidad", "f1", "muestras")
    tabla_clases = ttk.Treeview(frame_tabla, columns=columnas_tabla, show="headings", height=5)
    tabla_clases.heading("calidad", text="Calidad (Clase)")
    tabla_clases.heading("precision", text="Precisión")
    tabla_clases.heading("sensibilidad", text="Sensibilidad")
    tabla_clases.heading("especificidad", text="Especificidad")
    tabla_clases.heading("f1", text="F1-Score")
    tabla_clases.heading("muestras", text="Muestras")

    for col in columnas_tabla:
        tabla_clases.column(col, width=120, anchor="center")

    for i, clase in enumerate(clases):
        datos_clase = reporte_diccionario[str(clase)]
        tabla_clases.insert("", "end", values=(
            clase,
            f"{datos_clase['precision'] * 100:.2f}%",
            f"{datos_clase['recall'] * 100:.2f}%",
            f"{especificidades[i] * 100:.2f}%",
            f"{datos_clase['f1-score'] * 100:.2f}%",
            int(datos_clase["support"])
        ))
    tabla_clases.pack(fill="x", padx=5, pady=5)

    notebook = ttk.Notebook(parent_widget)
    notebook.pack(fill="both", expand=True, padx=10, pady=5)

    figuras_con_titulos = [
        (fig1, "Resumen (Matriz + Frontera)"),
        (fig2, "Árbol completo"),
        (fig3, "Complejidad y Error"),
        (fig4, "Árbol Podado Optimizado"),
    ]

    for figura, titulo in figuras_con_titulos:
        pestana = ttk.Frame(notebook)
        notebook.add(pestana, text=titulo)

        canvas = FigureCanvasTkAgg(figura, master=pestana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, pestana)
        toolbar.update()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Árbol de Decisión Vinos")
    root.geometry("1200x850")
    crear_interfaz_vinos_arbol(root)
    root.mainloop()
