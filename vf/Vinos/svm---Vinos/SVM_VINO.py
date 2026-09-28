import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

def cargar_datos_vinos_svm():
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    curr = directorio_actual
    ruta_origen = None
    
    while curr:
        candidato = os.path.join(curr, 'redwine', 'winequality-red.csv')
        candidato2 = os.path.join(curr, 'winequality-red.csv')
        candidato_limpio = os.path.join(curr, 'dataset_vino_limpio.csv')
        if os.path.exists(candidato_limpio):
            ruta_origen = candidato_limpio
            break
        elif os.path.exists(candidato):
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

    datos_vino = pd.read_csv(ruta_origen).drop_duplicates()
    return datos_vino

def crear_interfaz_vinos_svm_lineal(parent_widget):
    datos_vino = cargar_datos_vinos_svm()
    if datos_vino is None:
        lbl = ttk.Label(parent_widget, text="No se encontró winequality-red.csv / dataset_vino_limpio.csv", font=('Segoe UI', 12, 'bold'), foreground='red')
        lbl.pack(pady=20)
        return

    objetivo = datos_vino['quality']
    variables_entrada = datos_vino.drop('quality', axis=1)

    normalizador = StandardScaler()
    valores_escalados = normalizador.fit_transform(variables_entrada)
    datos_normalizados = pd.DataFrame(valores_escalados, columns=variables_entrada.columns)

    X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
        datos_normalizados, objetivo, test_size=0.2, random_state=77, stratify=objetivo
    )

    modelo_svm_lineal = SVC(kernel='linear', C=1.0, random_state=77)
    modelo_svm_lineal.fit(X_entrenar, y_entrenar)

    predicciones = modelo_svm_lineal.predict(X_probar)
    acc_orig = accuracy_score(y_probar, predicciones)
    err_orig = 1.0 - acc_orig
    matriz_confusion = confusion_matrix(y_probar, predicciones)

    report_orig = classification_report(y_probar, predicciones, output_dict=True, zero_division=0)
    prec_orig = report_orig['macro avg']['precision']
    rec_orig = report_orig['macro avg']['recall']
    f1_orig = report_orig['macro avg']['f1-score']

    # Especificidad Original
    clases_orig = sorted(objetivo.unique())
    spec_list_orig = []
    for i, c in enumerate(clases_orig):
        tn = matriz_confusion.sum() - (matriz_confusion[i,:].sum() + matriz_confusion[:,i].sum() - matriz_confusion[i,i])
        fp = matriz_confusion[:,i].sum() - matriz_confusion[i,i]
        spec_list_orig.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    spec_orig = np.mean(spec_list_orig)

    acc_train_orig = accuracy_score(y_entrenar, modelo_svm_lineal.predict(X_entrenar))

    pca = PCA(n_components=2)
    X_probar_pca = pca.fit_transform(X_probar)
    X_entrenar_pca = pca.fit_transform(X_entrenar)
    varianza = pca.explained_variance_ratio_

    modelo_svm_2d = SVC(kernel='linear', C=1.0, random_state=77)
    modelo_svm_2d.fit(X_entrenar_pca, y_entrenar)

    coef_matrix = np.abs(modelo_svm_lineal.coef_)
    coef_mean = coef_matrix.mean(axis=0)
    coefs = pd.Series(coef_mean, index=variables_entrada.columns)

    UMBRAL_COEF = 0.05
    variables_conservadas = coefs[coefs >= UMBRAL_COEF].index.tolist()
    variables_a_eliminar = coefs[coefs < UMBRAL_COEF].index.tolist()

    if len(variables_conservadas) < 2:
        variables_conservadas = coefs.sort_values(ascending=False).head(5).index.tolist()
        variables_a_eliminar = [c for c in variables_entrada.columns if c not in variables_conservadas]

    X_entrenar_mej = X_entrenar[variables_conservadas]
    X_probar_mej = X_probar[variables_conservadas]

    modelo_mej = SVC(kernel='linear', C=1.0, random_state=77)
    modelo_mej.fit(X_entrenar_mej, y_entrenar)

    pred_mej = modelo_mej.predict(X_probar_mej)
    acc_mej = accuracy_score(y_probar, pred_mej)
    err_mej = 1.0 - acc_mej
    matriz_confusion_mejorado = confusion_matrix(y_probar, pred_mej)

    report_mej = classification_report(y_probar, pred_mej, output_dict=True, zero_division=0)
    prec_mej = report_mej['macro avg']['precision']
    rec_mej = report_mej['macro avg']['recall']
    f1_mej = report_mej['macro avg']['f1-score']

    spec_list_mej = []
    for i, c in enumerate(clases_orig):
        tn = matriz_confusion_mejorado.sum() - (matriz_confusion_mejorado[i,:].sum() + matriz_confusion_mejorado[:,i].sum() - matriz_confusion_mejorado[i,i])
        fp = matriz_confusion_mejorado[:,i].sum() - matriz_confusion_mejorado[i,i]
        spec_list_mej.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    spec_mej = np.mean(spec_list_mej)

    acc_train_mej = accuracy_score(y_entrenar, modelo_mej.predict(X_entrenar_mej))

    notebook = ttk.Notebook(parent_widget)
    notebook.pack(fill='both', expand=True, padx=5, pady=5)

    def embed_fig(parent, fig):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        return canvas

    # PESTAÑA 1
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text='  🎯  Frontera PCA Original  ')
    fig1 = mfigure.Figure(figsize=(10, 5), facecolor='#1e1e2e')
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    for ax in (ax1, ax2): ax.set_facecolor('#181825')

    x_min, x_max = X_entrenar_pca[:, 0].min() - 1, X_entrenar_pca[:, 0].max() + 1
    y_min, y_max = X_entrenar_pca[:, 1].min() - 1, X_entrenar_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    Z = modelo_svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax1.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
    sc = ax1.scatter(X_probar_pca[:, 0], X_probar_pca[:, 1], c=y_probar, cmap=plt.cm.coolwarm, edgecolors='#cdd6f4', linewidths=0.3, zorder=3)
    ax1.set_title('Frontera 2D PCA - Datos de Prueba', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax1.set_xlabel('PC1', color='#cdd6f4', fontsize=8)
    ax1.set_ylabel('PC2', color='#cdd6f4', fontsize=8)
    ax1.tick_params(colors='#cdd6f4', labelsize=8)

    clases_labels = [str(c) for c in clases_orig]
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=clases_labels, yticklabels=clases_labels, square=True)
    ax2.set_title('Matriz de Confusión Original', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Predicción', color='#cdd6f4', fontsize=8)
    ax2.set_ylabel('Valor Real', color='#cdd6f4', fontsize=8)
    ax2.tick_params(colors='#cdd6f4', labelsize=8)
    ax2.set_box_aspect(1)
    fig1.tight_layout()
    embed_fig(tab1, fig1)

    # PESTAÑA 2
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text='  📊  Coeficiente e Importancia  ')
    fig2 = mfigure.Figure(figsize=(10, 5), facecolor='#1e1e2e')
    ax2_1 = fig2.add_subplot(1, 2, 1)
    ax2_2 = fig2.add_subplot(1, 2, 2)
    for ax in (ax2_1, ax2_2): ax.set_facecolor('#181825')

    coefs_sorted = coefs.sort_values(ascending=True)
    colors = ['#f38ba8' if v < UMBRAL_COEF else '#89b4fa' for v in coefs_sorted.values]
    ax2_1.barh(coefs_sorted.index, coefs_sorted.values, color=colors)
    ax2_1.axvline(UMBRAL_COEF, color='#f9e2af', linestyle='--', label=f'Umbral={UMBRAL_COEF}')
    ax2_1.set_title('Importancia Promedio de Variables (Coeff)', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax2_1.tick_params(colors='#cdd6f4', labelsize=8)
    ax2_1.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)

    ax2_2.bar(['PC1', 'PC2'], varianza * 100, color=['#89b4fa', '#a6e3a1'])
    ax2_2.set_title(f'Varianza Explicada PCA (Total: {varianza.sum()*100:.1f}%)', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax2_2.set_ylabel('% Varianza', color='#cdd6f4', fontsize=8)
    ax2_2.tick_params(colors='#cdd6f4', labelsize=8)
    fig2.tight_layout()
    embed_fig(tab2, fig2)

    # PESTAÑA 3
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text='  ⚡  Frontera PCA Mejorado  ')
    frame_ctrl = tk.Frame(tab3, bg='#1e1e2e')
    frame_ctrl.pack(fill='x', padx=10, pady=5)
    tk.Label(frame_ctrl, text="Multiplicador de Separación de Clusters:", bg='#1e1e2e', fg='#cdd6f4', font=('Segoe UI', 9)).pack(side='left', padx=5)

    mult_var = tk.DoubleVar(value=1.5)
    frame_fig3 = tk.Frame(tab3, bg='#1e1e2e')
    frame_fig3.pack(fill='both', expand=True)

    fig3 = mfigure.Figure(figsize=(9, 5), facecolor='#1e1e2e')
    canvas3 = FigureCanvasTkAgg(fig3, master=frame_fig3)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill='both', expand=True)

    def actualizar_pca_mejorado(*args):
        fig3.clear()
        mult = mult_var.get()
        pca_m = PCA(n_components=2)
        X_tr_m_pca = pca_m.fit_transform(X_entrenar_mej)
        X_te_m_pca = pca_m.transform(X_probar_mej)

        X_tr_sep = X_tr_m_pca.copy()
        X_te_sep = X_te_m_pca.copy()

        for c in np.unique(y_entrenar):
            mask_tr = (y_entrenar.values == c)
            centro = X_tr_sep[mask_tr].mean(axis=0) if mask_tr.sum() > 0 else 0
            X_tr_sep[mask_tr] = centro + (X_tr_sep[mask_tr] - centro) * mult
            mask_te = (y_probar.values == c)
            X_te_sep[mask_te] = centro + (X_te_sep[mask_te] - centro) * mult

        m_2d_mej = SVC(kernel='linear', C=1.0, random_state=77).fit(X_tr_sep, y_entrenar)

        ax = fig3.add_subplot(111)
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=8)

        xm, xM = X_tr_sep[:, 0].min() - 2, X_tr_sep[:, 0].max() + 2
        ym, yM = X_tr_sep[:, 1].min() - 2, X_tr_sep[:, 1].max() + 2
        xx_m, yy_m = np.meshgrid(np.arange(xm, xM, 0.2), np.arange(ym, yM, 0.2))
        Z_m = m_2d_mej.predict(np.c_[xx_m.ravel(), yy_m.ravel()]).reshape(xx_m.shape)

        ax.contourf(xx_m, yy_m, Z_m, alpha=0.35, cmap=plt.cm.coolwarm)
        sc_m = ax.scatter(X_te_sep[:, 0], X_te_sep[:, 1], c=y_probar, cmap=plt.cm.coolwarm, edgecolors='#cdd6f4', linewidths=0.4, zorder=3)
        ax.set_title(f'Frontera SVM Lineal Mejorado (Variables Conservadas, Multiplicador={mult}x)', color='#cdd6f4', fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1', color='#cdd6f4', fontsize=9)
        ax.set_ylabel('PC2', color='#cdd6f4', fontsize=9)
        fig3.tight_layout()
        canvas3.draw()

    mult_combo = ttk.Combobox(frame_ctrl, textvariable=mult_var, values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], width=5, state='readonly')
    mult_combo.pack(side='left', padx=5)
    mult_combo.bind('<<ComboboxSelected>>', actualizar_pca_mejorado)
    actualizar_pca_mejorado()

    # PESTAÑA 4
    tab4 = ttk.Frame(notebook)
    notebook.add(tab4, text='  📈  Importancia (Mejorado)  ')
    fig4 = mfigure.Figure(figsize=(9, 5), facecolor='#1e1e2e')
    ax4 = fig4.add_subplot(111)
    ax4.set_facecolor('#181825')
    ax4.tick_params(colors='#cdd6f4', labelsize=8)

    coef_mej_matrix = np.abs(modelo_mej.coef_).mean(axis=0)
    coef_mej_ser = pd.Series(coef_mej_matrix, index=variables_conservadas).sort_values(ascending=True)
    ax4.barh(coef_mej_ser.index, coef_mej_ser.values, color='#a6e3a1', edgecolor='#1e1e2e')
    ax4.set_title('Importancia de Variables Retenidas en Modelo Mejorado', color='#cdd6f4', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Coeficiente Absoluto Promedio', color='#cdd6f4', fontsize=9)
    fig4.tight_layout()
    embed_fig(tab4, fig4)

    # PESTAÑA 5
    tab5 = ttk.Frame(notebook)
    notebook.add(tab5, text='  ⚖️  Validación y Comparación  ')

    canvas_tab5 = tk.Canvas(tab5, bg='#1e1e2e', highlightthickness=0)
    scrollbar5 = ttk.Scrollbar(tab5, orient="vertical", command=canvas_tab5.yview)
    scrollable_frame5 = ttk.Frame(canvas_tab5)

    scrollable_frame5.bind("<Configure>", lambda e: canvas_tab5.configure(scrollregion=canvas_tab5.bbox("all")))
    canvas_tab5.create_window((0, 0), window=scrollable_frame5, anchor="nw")
    canvas_tab5.configure(yscrollcommand=scrollbar5.set)
    canvas_tab5.pack(side="left", fill="both", expand=True)
    scrollbar5.pack(side="right", fill="y")

    frame_matrices = tk.Frame(scrollable_frame5, bg='#1e1e2e')
    frame_matrices.pack(fill='x', pady=10)

    fig_cm = mfigure.Figure(figsize=(10, 4.5), facecolor='#1e1e2e')
    ax_cm1 = fig_cm.add_subplot(1, 2, 1)
    ax_cm2 = fig_cm.add_subplot(1, 2, 2)
    for ax_cm in (ax_cm1, ax_cm2): ax_cm.set_facecolor('#1e1e2e')

    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax_cm1, xticklabels=clases_labels, yticklabels=clases_labels, square=True)
    ax_cm1.set_title('Matriz Original', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax_cm1.tick_params(colors='#cdd6f4')
    ax_cm1.set_box_aspect(1)

    sns.heatmap(matriz_confusion_mejorado, annot=True, fmt='d', cmap='Greens', ax=ax_cm2, xticklabels=clases_labels, yticklabels=clases_labels, square=True)
    ax_cm2.set_title('Matriz Mejorado', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax_cm2.tick_params(colors='#cdd6f4')
    ax_cm2.set_box_aspect(1)
    fig_cm.tight_layout()
    embed_fig(frame_matrices, fig_cm)

    frame_metricas = tk.Frame(scrollable_frame5, bg='#1e1e2e')
    frame_metricas.pack(fill='x', pady=10)

    tabla_texto = f"""MÉTRICA                 ORIGINAL       MEJORADO
------------------------------------------------------
Accuracy                {acc_orig*100:6.2f}%       {acc_mej*100:6.2f}%
Error                   {err_orig*100:6.2f}%       {err_mej*100:6.2f}%
Precision               {prec_orig*100:6.2f}%       {prec_mej*100:6.2f}%
Recall                  {rec_orig*100:6.2f}%       {rec_mej*100:6.2f}%
Specificity             {spec_orig*100:6.2f}%       {spec_mej*100:6.2f}%
F1                      {f1_orig*100:6.2f}%       {f1_mej*100:6.2f}%

ERROR DE ENTRENAMIENTO Y GENERALIZACIÓN
------------------------------------------------------
Accuracy entrenamiento  {acc_train_orig*100:6.2f}%       {acc_train_mej*100:6.2f}%
Error entrenamiento     {(1-acc_train_orig)*100:6.2f}%       {(1-acc_train_mej)*100:6.2f}%
Accuracy prueba         {acc_orig*100:6.2f}%       {acc_mej*100:6.2f}%
Error generalización    {err_orig*100:6.2f}%       {err_mej*100:6.2f}%

BRECHA DE GENERALIZACIÓN
------------------------------------------------------
Brecha Original: {(acc_train_orig - acc_orig)*100:6.2f}%
Brecha Mejorado: {(acc_train_mej - acc_mej)*100:6.2f}%

RESUMEN FINAL
------------------------------------------------------
Variables Eliminadas:  {', '.join(variables_a_eliminar)}
Variables Conservadas: {', '.join(variables_conservadas)}
Dif. Accuracy: {(acc_mej - acc_orig)*100:+.2f}%
"""
    lbl_metricas = tk.Label(frame_metricas, text=tabla_texto, bg='#1e1e2e', fg='#cdd6f4', font=('Consolas', 10), justify='left')
    lbl_metricas.pack(padx=20, pady=10)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("SVM Lineal Vinos")
    root.geometry("1200x850")
    crear_interfaz_vinos_svm_lineal(root)
    root.mainloop()
