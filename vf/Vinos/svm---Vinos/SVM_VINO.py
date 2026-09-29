import os
import sys

# Configuración automática TCL/TK si es necesario
if 'TCL_LIBRARY' not in os.environ or 'TK_LIBRARY' not in os.environ:
    base_prefix = getattr(sys, 'base_prefix', sys.prefix)
    tcl_cand = os.path.join(base_prefix, 'tcl', 'tcl8.6')
    tk_cand = os.path.join(base_prefix, 'tcl', 'tk8.6')
    if os.path.exists(tcl_cand): os.environ['TCL_LIBRARY'] = tcl_cand
    if os.path.exists(tk_cand): os.environ['TK_LIBRARY'] = tk_cand

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

def cargar_datos_vino():
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    curr = directorio_actual
    ruta_origen = None
    while curr:
        candidato_limpio = os.path.join(curr, 'dataset_vino_limpio.csv')
        candidato_raw    = os.path.join(curr, 'winequality-red.csv')
        if os.path.exists(candidato_limpio):
            ruta_origen = candidato_limpio
            break
        elif os.path.exists(candidato_raw):
            ruta_origen = candidato_raw
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    if not ruta_origen:
        ruta_origen = os.path.normpath(os.path.join(directorio_actual, '..', '..', '..', 'dataset_vino_limpio.csv'))

    if not os.path.exists(ruta_origen):
        raise FileNotFoundError(f"No se encontró el archivo de datos de vinos en: {ruta_origen}")

    return pd.read_csv(ruta_origen)

def embed_figure(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas

def crear_interfaz_vinos_svm_lineal(parent_widget):
    datos_vino = cargar_datos_vino()
    objetivo         = datos_vino['quality']
    variables_entrada = datos_vino.drop('quality', axis=1)

    normalizador     = StandardScaler()
    valores_escalados = normalizador.fit_transform(variables_entrada)
    datos_normalizados = pd.DataFrame(valores_escalados, columns=variables_entrada.columns)

    X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
        datos_normalizados, objetivo,
        test_size=0.2, random_state=77, stratify=objetivo
    )

    modelo_svm_lineal = SVC(kernel='linear', C=1.0, random_state=77)
    modelo_svm_lineal.fit(X_entrenar, y_entrenar)

    predicciones     = modelo_svm_lineal.predict(X_probar)
    exactitud        = accuracy_score(y_probar, predicciones)
    matriz_confusion = confusion_matrix(y_probar, predicciones)

    pca = PCA(n_components=2)
    X_probar_pca = pca.fit_transform(X_probar)
    varianza = pca.explained_variance_ratio_

    pca2 = PCA(n_components=2)
    X_entrenar_pca = pca2.fit_transform(X_entrenar)
    modelo_svm_2d = SVC(kernel='linear', C=1.0, random_state=77)
    modelo_svm_2d.fit(X_entrenar_pca, y_entrenar)

    coef_matrix = np.abs(modelo_svm_lineal.coef_)
    coef_mean   = coef_matrix.mean(axis=0)
    coefs        = pd.Series(coef_mean, index=variables_entrada.columns)
    top_n        = min(15, len(coefs))
    top_features    = coefs.sort_values(ascending=False).head(top_n)
    bottom_features = coefs.sort_values(ascending=True).head(top_n)

    # ==========================================
    # CÁLCULO MODELO MEJORADO
    # ==========================================
    num_eliminar = 3
    variables_a_eliminar = coefs.sort_values(ascending=True).head(num_eliminar).index.tolist()
    variables_conservadas = [c for c in variables_entrada.columns if c not in variables_a_eliminar]

    X_entrenar_mejorado = X_entrenar[variables_conservadas]
    X_probar_mejorado = X_probar[variables_conservadas]

    modelo_svm_mejorado = SVC(kernel='linear', C=1.0, random_state=77)
    modelo_svm_mejorado.fit(X_entrenar_mejorado, y_entrenar)

    pred_mej_test = modelo_svm_mejorado.predict(X_probar_mejorado)
    pred_mej_train = modelo_svm_mejorado.predict(X_entrenar_mejorado)
    pred_orig_train = modelo_svm_lineal.predict(X_entrenar)

    matriz_confusion_mejorado = confusion_matrix(y_probar, pred_mej_test)

    def calcular_metricas(y_true, y_pred, cm):
        acc = accuracy_score(y_true, y_pred)
        err = 1.0 - acc
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        prec_macro = report['macro avg']['precision']
        rec_macro = report['macro avg']['recall']
        f1_macro = report['macro avg']['f1-score']
        
        spec_list = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            spec_list.append(spec)
        spec_macro = np.mean(spec_list)
        return acc, err, prec_macro, rec_macro, spec_macro, f1_macro

    acc_orig, err_orig, prec_orig, rec_orig, spec_orig, f1_orig = calcular_metricas(y_probar, predicciones, matriz_confusion)
    acc_mej, err_mej, prec_mej, rec_mej, spec_mej, f1_mej = calcular_metricas(y_probar, pred_mej_test, matriz_confusion_mejorado)

    acc_train_orig = accuracy_score(y_entrenar, pred_orig_train)
    acc_train_mej = accuracy_score(y_entrenar, pred_mej_train)

    pca_mej = PCA(n_components=2)
    X_probar_pca_mej = pca_mej.fit_transform(X_probar_mejorado)
    varianza_mej = pca_mej.explained_variance_ratio_
    pca2_mej = PCA(n_components=2)
    X_entrenar_pca_mej = pca2_mej.fit_transform(X_entrenar_mejorado)
    modelo_svm_2d_mej = SVC(kernel='linear', C=1.0, random_state=77)
    modelo_svm_2d_mej.fit(X_entrenar_pca_mej, y_entrenar)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook',     background='#1e1e2e', borderwidth=0)
    style.configure('TNotebook.Tab', background='#313244', foreground='#cdd6f4',
                    padding=[14, 6], font=('Segoe UI', 10, 'bold'))
    style.map('TNotebook.Tab',
              background=[('selected', '#a6e3a1')],
              foreground=[('selected', '#1e1e2e')])
    style.configure('TFrame', background='#1e1e2e')

    notebook = ttk.Notebook(parent_widget)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # ── PESTAÑA 1: Matriz de Confusión ─────────────────────────────────────────
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text='  📊  Matriz de Confusión  ')

    clases_labels = [str(c) for c in sorted(objetivo.unique())]
    fig1 = mfigure.Figure(figsize=(7, 5.5), facecolor='#1e1e2e')
    ax1 = fig1.add_subplot(111)
    ax1.set_facecolor('#1e1e2e')
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=clases_labels, yticklabels=clases_labels,
                annot_kws={'size': 11, 'weight': 'bold'}, linewidths=0.5)
    ax1.set_title(f'Matriz de Confusión — SVM Lineal (Vinos)\nExactitud: {exactitud*100:.2f}%',
                  fontsize=13, fontweight='bold', color='#cdd6f4', pad=14)
    ax1.set_xlabel('Calidad Predicha', color='#cdd6f4', fontsize=11)
    ax1.set_ylabel('Calidad Real',     color='#cdd6f4', fontsize=11)
    ax1.tick_params(colors='#cdd6f4')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#313244')
    fig1.tight_layout(pad=2)
    embed_figure(tab1, fig1)

    # ── PESTAÑA 2: Visualización PCA ───────────────────────────────────────────
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text='  📐  Visualización PCA  ')

    fig2 = mfigure.Figure(figsize=(8, 6), facecolor='#1e1e2e')
    ax2 = fig2.add_subplot(111)
    ax2.set_facecolor('#1e1e2e')

    clases = sorted(y_probar.unique())
    cmap_vino = plt.cm.RdYlGn
    for i, clase in enumerate(clases):
        idx = y_probar.values == clase
        ax2.scatter(X_probar_pca[idx, 0], X_probar_pca[idx, 1],
                    label=f'Calidad {clase}', alpha=0.75,
                    color=cmap_vino(i / max(1, len(clases)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)

    xp = np.linspace(ax2.get_xlim()[0] if ax2.get_xlim()[0] != 0 else X_probar_pca[:, 0].min()-1,
                        X_probar_pca[:, 0].max()+1, 400)
    normas = np.linalg.norm(modelo_svm_2d.coef_, axis=1)
    idx_principal = int(np.argmax(normas))

    for k, (w2, b2) in enumerate(zip(modelo_svm_2d.coef_, modelo_svm_2d.intercept_)):
        if abs(w2[1]) < 1e-10:
            continue
        yp = (-w2[0]*xp - b2) / w2[1]
        if k == idx_principal:
            ax2.plot(xp, yp, 'w-', linewidth=2.2, alpha=0.95, zorder=5,
                     label='Hiperplano principal')
        else:
            ax2.plot(xp, yp, color='white', linewidth=0.8, alpha=0.22, zorder=4)

    ax2.set_xlim(X_probar_pca[:, 0].min()-1, X_probar_pca[:, 0].max()+1)
    ax2.set_ylim(X_probar_pca[:, 1].min()-1, X_probar_pca[:, 1].max()+1)
    ax2.set_title(f'SVM Lineal — Visualización PCA de Calidad del Vino\nPC1: {varianza[0]*100:.1f}%  |  PC2: {varianza[1]*100:.1f}%',
                  fontsize=12, fontweight='bold', color='#cdd6f4', pad=12)
    ax2.set_xlabel(f'Componente Principal 1 ({varianza[0]*100:.2f}% varianza)', color='#cdd6f4')
    ax2.set_ylabel(f'Componente Principal 2 ({varianza[1]*100:.2f}% varianza)', color='#cdd6f4')
    ax2.tick_params(colors='#cdd6f4')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#313244')
    legend = ax2.legend(loc='best', fontsize=9, facecolor='#313244', labelcolor='#cdd6f4')
    ax2.grid(True, color='#313244', linewidth=0.5, alpha=0.6)
    fig2.tight_layout(pad=2)
    embed_figure(tab2, fig2)

    # ── PESTAÑA 3: Características Representativas ─────────────────────────────
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text='  🏆  Características Representativas  ')

    fig3 = mfigure.Figure(figsize=(11, 6), facecolor='#1e1e2e')
    ax3a = fig3.add_subplot(1, 2, 1)
    ax3b = fig3.add_subplot(1, 2, 2)

    for ax in (ax3a, ax3b):
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#313244')

    ax3a.barh(top_features.index[::-1], top_features.values[::-1],
              color='#89b4fa', edgecolor='#1e1e2e', height=0.65)
    ax3a.set_title(f'Top {top_n} Variables Más Representativas\n(Media |coef w| entre clases)',
                   fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
    ax3a.set_xlabel('Media |coeficiente w|', color='#cdd6f4', fontsize=9)
    for i, (feat, val) in enumerate(zip(top_features.index[::-1], top_features.values[::-1])):
        ax3a.text(val + max(top_features.values)*0.01, i, f'{val:.4f}',
                  va='center', ha='left', color='#cdd6f4', fontsize=8)

    ax3b.barh(bottom_features.index[::-1], bottom_features.values[::-1],
              color='#a6e3a1', edgecolor='#1e1e2e', height=0.65)
    ax3b.set_title(f'Top {top_n} Variables Menos Representativas\n(Media |coef w| entre clases)',
                   fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
    ax3b.set_xlabel('Media |coeficiente w|', color='#cdd6f4', fontsize=9)
    for i, (feat, val) in enumerate(zip(bottom_features.index[::-1], bottom_features.values[::-1])):
        ax3b.text(val + max(bottom_features.values)*0.01 if max(bottom_features.values) > 0 else 0.0001,
                  i, f'{val:.4f}', va='center', ha='left', color='#cdd6f4', fontsize=8)

    fig3.tight_layout(pad=2.5)
    embed_figure(tab3, fig3)

    # ── PESTAÑA 4: Modelo Mejorado ────────────────────────────────────────────
    tab4 = ttk.Frame(notebook)
    notebook.add(tab4, text='  ✨  Modelo Mejorado  ')

    frame_vars = tk.Frame(tab4, bg='#1e1e2e')
    frame_vars.pack(side='top', fill='x', pady=5)

    lbl_vars_orig = tk.Label(frame_vars, text=f"VARIABLES ORIGINALES\nNúmero total: {len(variables_entrada.columns)}", bg='#1e1e2e', fg='#cdd6f4', font=('Segoe UI', 10, 'bold'))
    lbl_vars_orig.pack(side='left', padx=20)

    txt_conservadas = "\n".join([f"- {v}" for v in variables_conservadas])
    lbl_vars_cons = tk.Label(frame_vars, text=f"VARIABLES CONSERVADAS\n{txt_conservadas}", bg='#1e1e2e', fg='#a6e3a1', font=('Segoe UI', 9), justify='left')
    lbl_vars_cons.pack(side='left', padx=20)

    txt_eliminadas = "\n".join([f"- {v}" for v in variables_a_eliminar])
    lbl_vars_elim = tk.Label(frame_vars, text=f"VARIABLES ELIMINADAS\n{txt_eliminadas}", bg='#1e1e2e', fg='#f38ba8', font=('Segoe UI', 9), justify='left')
    lbl_vars_elim.pack(side='left', padx=20)

    frame_graf = tk.Frame(tab4, bg='#1e1e2e')
    frame_graf.pack(side='top', fill='both', expand=True)

    frame_ctrl = tk.Frame(frame_graf, bg='#1e1e2e')
    frame_ctrl.pack(side='top', fill='x', pady=5)
    tk.Label(frame_ctrl, text="Multiplicador de visualización:", bg='#1e1e2e', fg='#cdd6f4').pack(side='left', padx=5)

    mult_var = tk.DoubleVar(value=1.0)

    fig4 = mfigure.Figure(figsize=(7, 5), facecolor='#1e1e2e')
    ax4 = fig4.add_subplot(111)
    ax4.set_facecolor('#1e1e2e')
    canvas4 = FigureCanvasTkAgg(fig4, master=frame_graf)
    canvas4.get_tk_widget().pack(fill='both', expand=True)

    def actualizar_pca_mejorado(*args):
        ax4.clear()
        mult = mult_var.get()
        ax_m = ax4
        ax_m.set_facecolor('#1e1e2e')
        
        mult_max = 3.0
        graf_max_x = np.max(np.abs(X_probar_pca_mej[:, 0])) * mult_max + 2.0
        graf_max_y = np.max(np.abs(X_probar_pca_mej[:, 1])) * mult_max + 2.0
        
        xx_visual, yy_visual = np.meshgrid(np.arange(-graf_max_x, graf_max_x, 0.1),
                                           np.arange(-graf_max_y, graf_max_y, 0.1))
        
        Z = modelo_svm_2d_mej.predict(np.c_[xx_visual.ravel() / mult, yy_visual.ravel() / mult]).reshape(xx_visual.shape)
        
        clases_sorted = sorted(y_probar.unique())
        cmap_vino = plt.cm.RdYlGn
        
        ax_m.contourf(xx_visual, yy_visual, Z, alpha=0.12, cmap='RdYlGn',
                      levels=np.arange(min(clases_sorted)-0.5, max(clases_sorted)+1, 1))

        for i, clase in enumerate(clases_sorted):
            idx = y_probar.values == clase
            ax_m.scatter(X_probar_pca_mej[idx, 0] * mult, X_probar_pca_mej[idx, 1] * mult,
                        label=f'Calidad {clase}', alpha=0.75,
                        color=cmap_vino(i / max(1, len(clases_sorted)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)
        
        xp_visual = np.linspace(-graf_max_x, graf_max_x, 400)
        normas_m = np.linalg.norm(modelo_svm_2d_mej.coef_, axis=1)
        idx_principal_m = int(np.argmax(normas_m))
        for k, (w2, b2) in enumerate(zip(modelo_svm_2d_mej.coef_, modelo_svm_2d_mej.intercept_)):
            if abs(w2[1]) < 1e-10: continue
            yp_visual = (-w2[0]*(xp_visual/mult) - b2) / w2[1]
            yp_visual = yp_visual * mult
            if k == idx_principal_m:
                ax_m.plot(xp_visual, yp_visual, 'w-', linewidth=2.5, alpha=1.0, zorder=5, label='Hiperplano principal')
            else:
                ax_m.plot(xp_visual, yp_visual, color='white', linewidth=1.2, alpha=0.6, linestyle='--', zorder=4)

        ax_m.set_xlim(-graf_max_x, graf_max_x)
        ax_m.set_ylim(-graf_max_y, graf_max_y)

        ax_m.set_title(f'SVM Lineal — Visualización PCA Mejorado (Mult: {mult})\nPC1: {varianza_mej[0]*100:.1f}%  |  PC2: {varianza_mej[1]*100:.1f}%',
                      fontsize=12, fontweight='bold', color='#cdd6f4', pad=12)
        ax_m.set_xlabel(f'Componente Principal 1 ({varianza_mej[0]*100:.2f}% varianza)', color='#cdd6f4')
        ax_m.set_ylabel(f'Componente Principal 2 ({varianza_mej[1]*100:.2f}% varianza)', color='#cdd6f4')
        ax_m.tick_params(colors='#cdd6f4')
        for spine in ax_m.spines.values():
            spine.set_edgecolor('#313244')
        ax_m.legend(loc='best', fontsize=9, facecolor='#313244', labelcolor='#cdd6f4')
        ax_m.grid(True, color='#313244', linewidth=0.5, alpha=0.6)
        fig4.tight_layout(pad=2)

        canvas4.draw()

    mult_combo = ttk.Combobox(frame_ctrl, textvariable=mult_var, values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], width=5, state='readonly')
    mult_combo.pack(side='left', padx=5)
    mult_combo.bind('<<ComboboxSelected>>', actualizar_pca_mejorado)

    actualizar_pca_mejorado()

    # ── PESTAÑA 5: Validación y Comparación ────────────────────────────────────
    tab5 = ttk.Frame(notebook)
    notebook.add(tab5, text='  ⚖️  Validación y Comparación  ')

    canvas_tab5 = tk.Canvas(tab5, bg='#1e1e2e', highlightthickness=0)
    scrollbar5 = ttk.Scrollbar(tab5, orient="vertical", command=canvas_tab5.yview)
    scrollable_frame5 = ttk.Frame(canvas_tab5)

    scrollable_frame5.bind(
        "<Configure>",
        lambda e: canvas_tab5.configure(
            scrollregion=canvas_tab5.bbox("all")
        )
    )

    canvas_tab5.create_window((0, 0), window=scrollable_frame5, anchor="nw")
    canvas_tab5.configure(yscrollcommand=scrollbar5.set)
    canvas_tab5.pack(side="left", fill="both", expand=True)
    scrollbar5.pack(side="right", fill="y")

    frame_matrices = tk.Frame(scrollable_frame5, bg='#1e1e2e')
    frame_matrices.pack(fill='x', pady=10)

    fig_cm = mfigure.Figure(figsize=(10, 4), facecolor='#1e1e2e')
    ax_cm1 = fig_cm.add_subplot(1, 2, 1)
    ax_cm2 = fig_cm.add_subplot(1, 2, 2)
    for ax_cm in (ax_cm1, ax_cm2): ax_cm.set_facecolor('#1e1e2e')

    clases_labels = [str(c) for c in sorted(objetivo.unique())]
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax_cm1, xticklabels=clases_labels, yticklabels=clases_labels)
    ax_cm1.set_title('Matriz Original', color='#cdd6f4')
    ax_cm1.tick_params(colors='#cdd6f4')

    sns.heatmap(matriz_confusion_mejorado, annot=True, fmt='d', cmap='Greens', ax=ax_cm2, xticklabels=clases_labels, yticklabels=clases_labels)
    ax_cm2.set_title('Matriz Mejorado', color='#cdd6f4')
    ax_cm2.tick_params(colors='#cdd6f4')
    fig_cm.tight_layout()
    embed_figure(frame_matrices, fig_cm)

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
Interpretación: Valores altos indican sobreajuste.

RESUMEN FINAL
------------------------------------------------------
Variables               {len(variables_entrada.columns):6d}         {len(variables_conservadas):6d}
Variables eliminadas: {', '.join(variables_a_eliminar)}
Variables conservadas: {', '.join(variables_conservadas)}
Dif. Accuracy: {(acc_mej - acc_orig)*100:+.2f}%
Dif. Error:    {(err_mej - err_orig)*100:+.2f}%
Dif. F1:       {(f1_mej - f1_orig)*100:+.2f}%
Dif. Recall:   {(rec_mej - rec_orig)*100:+.2f}%
Dif. Precision:{(prec_mej - prec_orig)*100:+.2f}%
"""
    lbl_metricas = tk.Label(frame_metricas, text=tabla_texto, bg='#1e1e2e', fg='#cdd6f4', font=('Consolas', 10), justify='left')
    lbl_metricas.pack(padx=20, pady=10)

    return notebook

if __name__ == '__main__':
    root = tk.Tk()
    root.title("SVM Lineal — Vinos")
    root.geometry("1100x680")
    root.configure(bg='#1e1e2e')
    crear_interfaz_vinos_svm_lineal(root)
    root.mainloop()
