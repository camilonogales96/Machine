import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

directorio_actual = os.path.dirname(os.path.abspath(__file__))
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
    ruta_origen = os.path.normpath(os.path.join(directorio_actual, '..', 'dataset_vino_limpio.csv'))

datos_vino = pd.read_csv(ruta_origen)
objetivo          = datos_vino['quality']
variables_entrada = datos_vino.drop('quality', axis=1)

normalizador      = StandardScaler()
valores_escalados = normalizador.fit_transform(variables_entrada)
datos_normalizados = pd.DataFrame(valores_escalados, columns=variables_entrada.columns)

X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
    datos_normalizados, objetivo,
    test_size=0.2, random_state=77, stratify=objetivo
)

print("==============================================")
print("     ENTRENAMIENTO DEL MODELO SVM NO LINEAL")
print("==============================================")
print(f"Registros totales:    {len(datos_vino)}")
print(f"Volumen de entrenamiento: {len(X_entrenar)}")
print(f"Volumen de prueba:        {len(X_probar)}")
print(f"Clases de calidad: {sorted(objetivo.unique())}")

modelo_svm_no_lineal = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_svm_no_lineal.fit(X_entrenar, y_entrenar)

predicciones     = modelo_svm_no_lineal.predict(X_probar)
exactitud        = accuracy_score(y_probar, predicciones)
matriz_confusion = confusion_matrix(y_probar, predicciones)

print(f"\nExactitud (Accuracy): {exactitud * 100:.2f}%")
print("\nMatriz de Confusión:")
print(matriz_confusion)
print("\nReporte de Clasificación:")
print(classification_report(y_probar, predicciones, zero_division=0))

pca = PCA(n_components=2)
X_probar_pca = pca.fit_transform(X_probar)
varianza = pca.explained_variance_ratio_
print(f"\nVarianza PC1: {varianza[0]*100:.2f}%  |  PC2: {varianza[1]*100:.2f}%  |  Total: {varianza.sum()*100:.2f}%")
pca2 = PCA(n_components=2)
X_entrenar_pca = pca2.fit_transform(X_entrenar)
modelo_svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_svm_2d.fit(X_entrenar_pca, y_entrenar)

margen = 12.0
x_min2d = X_entrenar_pca[:, 0].min() - margen
x_max2d = X_entrenar_pca[:, 0].max() + margen
y_min2d = X_entrenar_pca[:, 1].min() - margen
y_max2d = X_entrenar_pca[:, 1].max() + margen
xx, yy  = np.meshgrid(np.arange(x_min2d, x_max2d, 0.12),
                      np.arange(y_min2d, y_max2d, 0.12))
Z = modelo_svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

print("\n[INFO] Calculando importancia por permutación (puede tardar unos segundos)...")
perm_imp = permutation_importance(
    modelo_svm_no_lineal, X_probar, y_probar,
    n_repeats=10, random_state=77, scoring='accuracy'
)
importancias    = pd.Series(perm_imp.importances_mean, index=variables_entrada.columns)
top_n           = min(15, len(importancias))
top_features    = importancias.sort_values(ascending=False).head(top_n)
bottom_features = importancias.sort_values(ascending=True).head(top_n)


# ==========================================
# CALCULO MODELO MEJORADO
# ==========================================
num_eliminar = 3
variables_a_eliminar = importancias.sort_values(ascending=True).head(num_eliminar).index.tolist()
variables_conservadas = [c for c in variables_entrada.columns if c not in variables_a_eliminar]

X_entrenar_mejorado = X_entrenar[variables_conservadas]
X_probar_mejorado = X_probar[variables_conservadas]

modelo_svm_mejorado = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_svm_mejorado.fit(X_entrenar_mejorado, y_entrenar)

pred_mej_test = modelo_svm_mejorado.predict(X_probar_mejorado)
pred_mej_train = modelo_svm_mejorado.predict(X_entrenar_mejorado)
pred_orig_train = modelo_svm_no_lineal.predict(X_entrenar)

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
modelo_svm_2d_mej = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_svm_2d_mej.fit(X_entrenar_pca_mej, y_entrenar)

x_min2d_mej = X_entrenar_pca_mej[:, 0].min() - margen
x_max2d_mej = X_entrenar_pca_mej[:, 0].max() + margen
y_min2d_mej = X_entrenar_pca_mej[:, 1].min() - margen
y_max2d_mej = X_entrenar_pca_mej[:, 1].max() + margen
xx_mej, yy_mej = np.meshgrid(np.arange(x_min2d_mej, x_max2d_mej, 0.12),
                             np.arange(y_min2d_mej, y_max2d_mej, 0.12))
Z_mej = modelo_svm_2d_mej.predict(np.c_[xx_mej.ravel(), yy_mej.ravel()]).reshape(xx_mej.shape)

root = tk.Tk()
root.title("SVM No Lineal (RBF) — Vinos")
root.geometry("1100x680")
root.configure(bg='#1e1e2e')

style = ttk.Style()
style.theme_use('clam')
style.configure('TNotebook',     background='#1e1e2e', borderwidth=0)
style.configure('TNotebook.Tab', background='#313244', foreground='#cdd6f4',
                padding=[14, 6], font=('Segoe UI', 10, 'bold'))
style.map('TNotebook.Tab',
          background=[('selected', '#cba6f7')],
          foreground=[('selected', '#1e1e2e')])
style.configure('TFrame', background='#1e1e2e')

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=10)


def embed_figure(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas


tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='  📊  Matriz de Confusión  ')

clases_labels = [str(c) for c in sorted(objetivo.unique())]
fig1 = mfigure.Figure(figsize=(7, 5.5), facecolor='#1e1e2e')
ax1  = fig1.add_subplot(111)
ax1.set_facecolor('#1e1e2e')
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Oranges', ax=ax1,
            xticklabels=clases_labels, yticklabels=clases_labels,
            annot_kws={'size': 11, 'weight': 'bold'}, linewidths=0.5)
ax1.set_title(f'Matriz de Confusión — SVM No Lineal RBF (Vinos)\nExactitud: {exactitud*100:.2f}%',
              fontsize=13, fontweight='bold', color='#cdd6f4', pad=14)
ax1.set_xlabel('Calidad Predicha', color='#cdd6f4', fontsize=11)
ax1.set_ylabel('Calidad Real',     color='#cdd6f4', fontsize=11)
ax1.tick_params(colors='#cdd6f4')
for spine in ax1.spines.values():
    spine.set_edgecolor('#313244')
fig1.tight_layout(pad=2)
embed_figure(tab1, fig1)


tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='  🔵  Frontera RBF (PCA)  ')

fig2 = mfigure.Figure(figsize=(8, 6), facecolor='#1e1e2e')
ax2  = fig2.add_subplot(111)
ax2.set_facecolor('#1e1e2e')

clases_sorted = sorted(objetivo.unique())
cmap_vino = plt.cm.RdYlGn

ax2.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlGn',
             levels=np.arange(min(clases_sorted)-0.5, max(clases_sorted)+1, 1))

for i, clase in enumerate(clases_sorted):
    idx = y_probar.values == clase
    ax2.scatter(X_probar_pca[idx, 0], X_probar_pca[idx, 1],
                label=f'Calidad {clase}', alpha=0.8,
                color=cmap_vino(i / max(1, len(clases_sorted)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)

ax2.set_xlim(X_probar_pca[:, 0].min()-12, X_probar_pca[:, 0].max()+12)
ax2.set_ylim(X_probar_pca[:, 1].min()-12, X_probar_pca[:, 1].max()+12)
ax2.set_title(f'SVM No Lineal (RBF) — Frontera de Decisión vía PCA\nPC1: {varianza[0]*100:.1f}%  |  PC2: {varianza[1]*100:.1f}%',
              fontsize=12, fontweight='bold', color='#cdd6f4', pad=12)
ax2.set_xlabel(f'Componente Principal 1 ({varianza[0]*100:.2f}% varianza)', color='#cdd6f4')
ax2.set_ylabel(f'Componente Principal 2 ({varianza[1]*100:.2f}% varianza)', color='#cdd6f4')
ax2.tick_params(colors='#cdd6f4')
for spine in ax2.spines.values():
    spine.set_edgecolor('#313244')
legend = ax2.legend(loc='best', fontsize=9, facecolor='#313244', labelcolor='#cdd6f4')
ax2.grid(True, color='#313244', linewidth=0.5, alpha=0.5)
fig2.tight_layout(pad=2)

canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
canvas2.draw()
toolbar_frame2 = tk.Frame(tab2, bg='#313244')
toolbar_frame2.pack(side='bottom', fill='x')
toolbar2 = NavigationToolbar2Tk(canvas2, toolbar_frame2)
toolbar2.update()
canvas2.get_tk_widget().pack(fill='both', expand=True)


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

colors_top = ['#cba6f7' if v >= 0 else '#f38ba8' for v in top_features.values[::-1]]
ax3a.barh(top_features.index[::-1], top_features.values[::-1],
          color=colors_top, edgecolor='#1e1e2e', height=0.65)
ax3a.set_title(f'Top {top_n} Variables Más Representativas\n(Importancia por Permutación)',
               fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
ax3a.set_xlabel('Reducción media de exactitud', color='#cdd6f4', fontsize=9)
ax3a.axvline(0, color='#6c7086', linewidth=0.8, linestyle='--')
for i, val in enumerate(top_features.values[::-1]):
    ax3a.text(val + abs(top_features.values).max()*0.01, i,
              f'{val:.4f}', va='center', ha='left', color='#cdd6f4', fontsize=8)

colors_bot = ['#a6e3a1' if v >= 0 else '#fab387' for v in bottom_features.values[::-1]]
ax3b.barh(bottom_features.index[::-1], bottom_features.values[::-1],
          color=colors_bot, edgecolor='#1e1e2e', height=0.65)
ax3b.set_title(f'Top {top_n} Variables Menos Representativas\n(Importancia por Permutación)',
               fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
ax3b.set_xlabel('Reducción media de exactitud', color='#cdd6f4', fontsize=9)
ax3b.axvline(0, color='#6c7086', linewidth=0.8, linestyle='--')
for i, val in enumerate(bottom_features.values[::-1]):
    ax3b.text(val + 0.0001, i, f'{val:.4f}',
              va='center', ha='left', color='#cdd6f4', fontsize=8)

fig3.tight_layout(pad=2.5)
embed_figure(tab3, fig3)

print("\n[INFO] Abriendo ventana interactiva SVM No Lineal Vinos...")

# ==========================================
# PESTAÑA 4: MODELO MEJORADO
# ==========================================
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
    
    # Límites fijos para evitar auto-zoom
    mult_max = 3.0
    graf_max_x = np.max(np.abs(X_probar_pca_mej[:, 0])) * mult_max + 2.0
    graf_max_y = np.max(np.abs(X_probar_pca_mej[:, 1])) * mult_max + 2.0
    
    xx_visual, yy_visual = np.meshgrid(np.arange(-graf_max_x, graf_max_x, 0.1),
                                       np.arange(-graf_max_y, graf_max_y, 0.1))
    
    # Inversa para predecir regiones
    Z = modelo_svm_2d_mej.predict(np.c_[xx_visual.ravel() / mult, yy_visual.ravel() / mult]).reshape(xx_visual.shape)
    
    clases_sorted = sorted(objetivo.unique())
    cmap_vino = plt.cm.RdYlGn
    
    ax_m.contourf(xx_visual, yy_visual, Z, alpha=0.12, cmap='RdYlGn',
                 levels=np.arange(min(clases_sorted)-0.5, max(clases_sorted)+1, 1))

    for i, clase in enumerate(clases_sorted):
        idx = y_probar.values == clase
        ax_m.scatter(X_probar_pca_mej[idx, 0] * mult, X_probar_pca_mej[idx, 1] * mult,
                    label=f'Calidad {clase}', alpha=0.8,
                    color=cmap_vino(i / max(1, len(clases_sorted)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)

    ax_m.set_xlim(-graf_max_x, graf_max_x)
    ax_m.set_ylim(-graf_max_y, graf_max_y)

    ax_m.set_title(f'SVM No Lineal (RBF) — Visualización PCA Mejorado (Mult: {mult})\nPC1: {varianza_mej[0]*100:.1f}%  |  PC2: {varianza_mej[1]*100:.1f}%',
                  fontsize=12, fontweight='bold', color='#cdd6f4', pad=12)
    ax_m.set_xlabel(f'Componente Principal 1 ({varianza_mej[0]*100:.2f}% varianza)', color='#cdd6f4')
    ax_m.set_ylabel(f'Componente Principal 2 ({varianza_mej[1]*100:.2f}% varianza)', color='#cdd6f4')
    ax_m.tick_params(colors='#cdd6f4')
    for spine in ax_m.spines.values():
        spine.set_edgecolor('#313244')
    ax_m.legend(loc='best', fontsize=9, facecolor='#313244', labelcolor='#cdd6f4')
    ax_m.grid(True, color='#313244', linewidth=0.5, alpha=0.5)
    fig4.tight_layout(pad=2)

    canvas4.draw()

mult_combo = ttk.Combobox(frame_ctrl, textvariable=mult_var, values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], width=5, state='readonly')
mult_combo.pack(side='left', padx=5)
mult_combo.bind('<<ComboboxSelected>>', actualizar_pca_mejorado)

actualizar_pca_mejorado()

# ==========================================
# PESTAÑA 5: VALIDACIÓN Y COMPARACIÓN
# ==========================================
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

root.mainloop()