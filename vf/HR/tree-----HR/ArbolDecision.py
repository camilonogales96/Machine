import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def cargar_datos_hr_arbol():
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    curr = directorio_actual
    ruta_origen = None
    
    while curr:
        candidato = os.path.join(curr, 'HRDataset_v14.csv')
        if os.path.exists(candidato):
            ruta_origen = candidato
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    if not ruta_origen:
        ruta_origen = os.path.join(directorio_actual, 'HRDataset_v14.csv')

    if not os.path.exists(ruta_origen):
        return None, None

    datos_hr = pd.read_csv(ruta_origen).drop_duplicates().dropna(subset=['Termd'])
    return datos_hr, ruta_origen

def crear_interfaz_hr_arbol(parent_widget):
    datos_hr, ruta_origen = cargar_datos_hr_arbol()
    if datos_hr is None:
        lbl = ttk.Label(parent_widget, text="No se encontró HRDataset_v14.csv", font=('Segoe UI', 12, 'bold'), foreground='red')
        lbl.pack(pady=20)
        return

    variables_numericas = datos_hr.select_dtypes(include=['int64', 'float64']).columns
    datos_hr_numerico = datos_hr[variables_numericas].fillna(0)

    objetivo = datos_hr_numerico['Termd']
    columnas_a_descartar = ['Termd']
    if 'EmpID' in datos_hr_numerico.columns:
        columnas_a_descartar.append('EmpID')

    variables_entrada = datos_hr_numerico.drop(columns=columnas_a_descartar)

    X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
        variables_entrada, objetivo, test_size=0.2, random_state=77
    )

    modelo_arbol = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=77)
    modelo_arbol.fit(X_entrenar, y_entrenar)

    predicciones = modelo_arbol.predict(X_probar)
    exactitud = accuracy_score(y_probar, predicciones)
    matriz_confusion = confusion_matrix(y_probar, predicciones)

    # FIGURA 1: Matriz de Confusión + Frontera (PCA 2D)
    fig1 = mfigure.Figure(figsize=(12, 5.5))
    canvas1_tmp = FigureCanvasTkAgg(fig1, master=parent_widget)
    canvas1_tmp.draw()

    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)

    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Activo (0)', 'Terminado (1)'],
                yticklabels=['Activo (0)', 'Terminado (1)'])
    ax1.set_title('Matriz de Confusión - Árbol de Decisión', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Valor Real')

    scaler = StandardScaler()
    X_entrenar_escalado = scaler.fit_transform(X_entrenar)
    pca = PCA(n_components=2)
    X_entrenar_pca = pca.fit_transform(X_entrenar_escalado)

    modelo_arbol_2d = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=77)
    modelo_arbol_2d.fit(X_entrenar_pca, y_entrenar)

    x_min, x_max = X_entrenar_pca[:, 0].min() - 1, X_entrenar_pca[:, 0].max() + 1
    y_min, y_max = X_entrenar_pca[:, 1].min() - 1, X_entrenar_pca[:, 1].max() + 1
    step_x = (x_max - x_min) / 200
    step_y = (y_max - y_min) / 200

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x), np.arange(y_min, y_max, step_y))
    Z = modelo_arbol_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax2.scatter(X_entrenar_pca[:, 0], X_entrenar_pca[:, 1], c=y_entrenar, cmap=plt.cm.coolwarm, edgecolors='k')
    ax2.set_title('Frontera de Decisión (Árbol 2D vía PCA)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Componente Principal 1')
    ax2.set_ylabel('Componente Principal 2')
    fig1.tight_layout()

    # FIGURA 2: Estructura del Árbol de Decisión
    fig2 = mfigure.Figure(figsize=(16, 8))
    canvas2_tmp = FigureCanvasTkAgg(fig2, master=parent_widget)
    canvas2_tmp.draw()

    ax3 = fig2.add_subplot(111)
    plot_tree(modelo_arbol, feature_names=list(variables_entrada.columns), class_names=['Activo', 'Terminado'],
              filled=True, rounded=True, fontsize=8, ax=ax3)
    ax3.set_title('Estructura Visual del Árbol de Decisión (Reglas de División)', fontsize=13, fontweight='bold')
    fig2.tight_layout()

    # FIGURA 3: Gráfico de Dispersión
    fig3 = mfigure.Figure(figsize=(9, 5.5))
    canvas3_tmp = FigureCanvasTkAgg(fig3, master=parent_widget)
    canvas3_tmp.draw()

    ax4 = fig3.add_subplot(111)
    sns.regplot(data=datos_hr_numerico, x='PerfScoreID', y='EngagementSurvey', x_jitter=0.15,
                scatter_kws=dict(alpha=0.6, s=45, color='#2a6ebb', edgecolor='white'),
                line_kws=dict(color='#c0392b', linewidth=2.5), ax=ax4)
    ax4.set_xlabel('PerfScoreID (1=PIP, 2=Necesita mejorar, 3=Cumple, 4=Excede)', fontsize=9.5)
    ax4.set_ylabel('EngagementSurvey (encuesta de compromiso, 1 a 5)', fontsize=9.5)
    ax4.set_title('Relación entre Desempeño y Compromiso del Empleado', fontsize=11, fontweight='bold')
    fig3.tight_layout()

    # FIGURA 4: Poda del Árbol (CCP)
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

    mejor_exactitud_prueba = max(exactitudes_test)
    candidatos_alpha = [(a, n, e) for a, e, n in zip(ccp_alphas, exactitudes_test, num_nodos)
                         if e >= mejor_exactitud_prueba - 0.005]
    alpha_recomendado, nodos_recomendado, exactitud_recomendada = max(candidatos_alpha, key=lambda t: t[0])

    modelo_arbol_podado = DecisionTreeClassifier(criterion='gini', random_state=77, ccp_alpha=alpha_recomendado)
    modelo_arbol_podado.fit(X_entrenar, y_entrenar)

    fig4 = mfigure.Figure(figsize=(12, 5.5))
    canvas4_tmp = FigureCanvasTkAgg(fig4, master=parent_widget)
    canvas4_tmp.draw()

    axA = fig4.add_subplot(1, 2, 1)
    axB = fig4.add_subplot(1, 2, 2)

    axA.plot(ccp_alphas, exactitudes_train, marker='o', markersize=6, label='Entrenamiento', color='#2a6ebb')
    axA.plot(ccp_alphas, exactitudes_test, marker='o', markersize=6, label='Prueba', color='#c0392b')
    axA.axvline(alpha_recomendado, color='#2e7d32', linestyle='--', linewidth=1.8, label=f'α sugerido ({alpha_recomendado:.3f})')
    axA.set_xlabel('ccp_alpha (nivel de poda)')
    axA.set_ylabel('Exactitud')
    axA.set_title('Exactitud vs. Nivel de Poda')
    axA.legend()

    axB.plot(ccp_alphas, num_nodos, marker='o', markersize=6, color='#6a3d9a')
    axB.axvline(alpha_recomendado, color='#2e7d32', linestyle='--', linewidth=1.8, label=f'Alpha sugerido ({nodos_recomendado} nodos)')
    axB.set_xlabel('ccp_alpha (nivel de poda)')
    axB.set_ylabel('Cantidad de nodos del árbol')
    axB.set_title('Complejidad del Árbol vs. Nivel de Poda')
    axB.legend()

    fig4.suptitle(f'Poda del Árbol (Cost-Complexity Pruning)\nÁrbol Base: {modelo_arbol.tree_.node_count} nodos ({exactitud*100:.1f}%) | Árbol Podado: {modelo_arbol_podado.tree_.node_count} nodos ({modelo_arbol_podado.score(X_probar, y_probar)*100:.1f}%)', fontsize=11, fontweight='bold')
    fig4.tight_layout()

    # FIGURA 5: Árbol Podado Optimizado
    fig5 = mfigure.Figure(figsize=(16, 8))
    canvas5_tmp = FigureCanvasTkAgg(fig5, master=parent_widget)
    canvas5_tmp.draw()

    ax5 = fig5.add_subplot(111)
    plot_tree(modelo_arbol_podado, feature_names=list(variables_entrada.columns), class_names=['Activo', 'Terminado'],
              filled=True, rounded=True, fontsize=8, ax=ax5)
    ax5.set_title(f'Estructura del Árbol Podado Optimizado (ccp_alpha = {alpha_recomendado:.4f}, Nodos = {modelo_arbol_podado.tree_.node_count})', fontsize=12, fontweight='bold')
    fig5.tight_layout()

    # MONTAJE DE PESTAÑAS EN NOTEBOOK
    notebook = ttk.Notebook(parent_widget)
    notebook.pack(fill='both', expand=True)

    figuras_con_titulos = [
        (fig1, "Resumen (Matriz + Frontera)"),
        (fig2, "Árbol completo"),
        (fig3, "Gráfico de Dispersión"),
        (fig4, "Poda del Árbol"),
        (fig5, "Árbol Podado Optimizado"),
    ]

    for figura, titulo in figuras_con_titulos:
        pestana = ttk.Frame(notebook)
        notebook.add(pestana, text=titulo)

        canvas = FigureCanvasTkAgg(figura, master=pestana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, pestana)
        toolbar.update()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Árbol de Decisión HR")
    root.geometry("1200x800")
    crear_interfaz_hr_arbol(root)
    root.mainloop()