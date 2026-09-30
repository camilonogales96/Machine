import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

datos_hr = pd.read_csv(ruta_origen).drop_duplicates().dropna(subset=['Termd'])
variables_numericas = datos_hr.select_dtypes(include=['int64', 'float64']).columns
datos_hr_numerico = datos_hr[variables_numericas].fillna(0)

objetivo = datos_hr_numerico['Termd']
variables_entrada = datos_hr_numerico.drop('Termd', axis=1)

normalizador = StandardScaler()
valores_escalados = normalizador.fit_transform(variables_entrada)
datos_normalizados = pd.DataFrame(valores_escalados, columns=variables_entrada.columns)

X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
    datos_normalizados, objetivo, test_size=0.2, random_state=77
)

print("=== ENTRENAMIENTO DEL MODELO SVM NO LINEAL (KERNEL RBF / RRHH) ===")
print(f"Volumen de datos de entrenamiento: {len(X_entrenar)}")
print(f"Volumen de datos de prueba: {len(X_probar)}")

modelo_svm_nolineal = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_svm_nolineal.fit(X_entrenar, y_entrenar)

predicciones     = modelo_svm_nolineal.predict(X_probar)
exactitud        = accuracy_score(y_probar, predicciones)
matriz_confusion = confusion_matrix(y_probar, predicciones)

print(f"\n=== REPORTE DE EVALUACION - SVM NO LINEAL (RBF) ===")
print(f"Exactitud (Accuracy): {exactitud * 100:.2f}%")
print("\nMatriz de Confusion:")
print(matriz_confusion)
print("\nReporte de Clasificacion:")
print(classification_report(y_probar, predicciones))

pca = PCA(n_components=2)
X_entrenar_pca = pca.fit_transform(X_entrenar)

modelo_svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_svm_2d.fit(X_entrenar_pca, y_entrenar)

margen = 12.0
x_min, x_max = X_entrenar_pca[:, 0].min() - margen, X_entrenar_pca[:, 0].max() + margen
y_min, y_max = X_entrenar_pca[:, 1].min() - margen, X_entrenar_pca[:, 1].max() + margen
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.15),
                     np.arange(y_min, y_max, 0.15))
Z = modelo_svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

print("\n[INFO] Calculando importancia por permutacion...")
perm_imp = permutation_importance(
    modelo_svm_nolineal, X_probar, y_probar,
    n_repeats=15, random_state=77, scoring='accuracy'
)
importancias    = pd.Series(perm_imp.importances_mean, index=variables_entrada.columns)
importancias    = importancias.sort_values(ascending=False)
top_n           = 15
top_features    = importancias.head(top_n)
bottom_features = importancias.tail(top_n)


umbral_significancia = 0.0
vars_significativas  = importancias[importancias > umbral_significancia].index.tolist()
if len(vars_significativas) < 2:
    vars_significativas = importancias.head(5).index.tolist()

pca_filt            = PCA(n_components=2)
X_entrenar_filt_pca = pca_filt.fit_transform(X_entrenar[vars_significativas])

SEPARACION_MULT = 4.0
X_sep = X_entrenar_filt_pca.copy()
for clase in np.unique(y_entrenar):
    mask      = (y_entrenar.values == clase)
    centroide = X_sep[mask].mean(axis=0)
    X_sep[mask] = centroide + (X_sep[mask] - centroide) * SEPARACION_MULT

modelo_filt_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_filt_2d.fit(X_sep, y_entrenar)

margen_f = 25.0
xf_min, xf_max = X_sep[:, 0].min() - margen_f, X_sep[:, 0].max() + margen_f
yf_min, yf_max = X_sep[:, 1].min() - margen_f, X_sep[:, 1].max() + margen_f
xx_f, yy_f = np.meshgrid(np.arange(xf_min, xf_max, 0.3),
                          np.arange(yf_min, yf_max, 0.3))
Z_f = modelo_filt_2d.predict(np.c_[xx_f.ravel(), yy_f.ravel()]).reshape(xx_f.shape)


tn, fp, fn, tp = matriz_confusion.ravel() if matriz_confusion.shape == (2, 2) else (
    matriz_confusion[1, 1], matriz_confusion[0, 1],
    matriz_confusion[1, 0], matriz_confusion[0, 0])
P_total = int(tp + fn)
N_total = int(tn + fp)
total   = P_total + N_total

exactitud_val     = (tp + tn) / total if total else 0
tasa_error_val    = (fp + fn) / total if total else 0
sensibilidad_val  = tp / P_total if P_total else 0
especificidad_val = tn / N_total if N_total else 0
precision_val     = tp / (tp + fp) if (tp + fp) else 0
f1_val            = (2 * precision_val * sensibilidad_val) / (precision_val + sensibilidad_val)\
                    if (precision_val + sensibilidad_val) else 0
beta              = 0.5
fbeta_val         = ((1 + beta**2) * precision_val * sensibilidad_val) /\
                    (beta**2 * precision_val + sensibilidad_val)\
                    if (beta**2 * precision_val + sensibilidad_val) else 0


root = tk.Tk()
root.title("SVM No Lineal (RBF) - RRHH")
root.geometry("1100x680")
root.configure(bg='#1e1e2e')

style = ttk.Style()
style.theme_use('clam')
style.configure('TNotebook',     background='#1e1e2e', borderwidth=0)
style.configure('TNotebook.Tab', background='#313244', foreground='#cdd6f4',
                padding=[14, 6], font=('Segoe UI', 10, 'bold'))
style.map('TNotebook.Tab',
          background=[('selected', '#89b4fa')],
          foreground=[('selected', '#1e1e2e')])
style.configure('TFrame', background='#1e1e2e')

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=10)


def embed_figure(parent, fig):
    """Incrusta una figura matplotlib en un frame tkinter."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas


tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='  Matriz de Confusion  ')

fig1 = mfigure.Figure(figsize=(6, 5), facecolor='#1e1e2e')
ax1  = fig1.add_subplot(111)
ax1.set_facecolor('#1e1e2e')
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Oranges', ax=ax1,
            xticklabels=['Activo (0)', 'Terminado (1)'],
            yticklabels=['Activo (0)', 'Terminado (1)'],
            annot_kws={'size': 16, 'weight': 'bold'}, linewidths=0.5)
ax1.set_title(f'Matriz de Confusion - SVM No Lineal (RBF)\nExactitud: {exactitud*100:.2f}%',
              fontsize=13, fontweight='bold', color='#cdd6f4', pad=14)
ax1.set_xlabel('Prediccion', color='#cdd6f4', fontsize=11)
ax1.set_ylabel('Valor Real',  color='#cdd6f4', fontsize=11)
ax1.tick_params(colors='#cdd6f4')
for spine in ax1.spines.values():
    spine.set_edgecolor('#313244')
fig1.tight_layout(pad=2)
embed_figure(tab1, fig1)


tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='  Frontera de Decision (RBF)  ')

fig2 = mfigure.Figure(figsize=(8, 6), facecolor='#1e1e2e')
ax2  = fig2.add_subplot(111)
ax2.set_facecolor('#1e1e2e')
ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
scatter = ax2.scatter(X_entrenar_pca[:, 0], X_entrenar_pca[:, 1],
                      c=y_entrenar, cmap=plt.cm.coolwarm, edgecolors='#cdd6f4',
                      linewidths=0.4, zorder=3)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_title('Frontera de Decision No Lineal (Kernel RBF en 2D via PCA)',
              fontsize=12, fontweight='bold', color='#cdd6f4', pad=12)
ax2.set_xlabel('Componente Principal 1', color='#cdd6f4')
ax2.set_ylabel('Componente Principal 2', color='#cdd6f4')
ax2.tick_params(colors='#cdd6f4')
for spine in ax2.spines.values():
    spine.set_edgecolor('#313244')
legend = ax2.legend(*scatter.legend_elements(), title='Termd',
                    facecolor='#313244', labelcolor='#cdd6f4',
                    title_fontsize=9, fontsize=9)
legend.get_title().set_color('#cdd6f4')
fig2.tight_layout(pad=2)
embed_figure(tab2, fig2)


tab3 = ttk.Frame(notebook)
notebook.add(tab3, text='  Caracteristicas Representativas  ')

fig3 = mfigure.Figure(figsize=(11, 6), facecolor='#1e1e2e')
ax3a = fig3.add_subplot(1, 2, 1)
ax3b = fig3.add_subplot(1, 2, 2)

for ax in (ax3a, ax3b):
    ax.set_facecolor('#181825')
    ax.tick_params(colors='#cdd6f4', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#313244')

colors_top = ['#89b4fa' if v >= 0 else '#f38ba8' for v in top_features.values]
bars_top = ax3a.barh(top_features.index[::-1], top_features.values[::-1],
                     color=colors_top[::-1], edgecolor='#1e1e2e', height=0.65)
ax3a.set_title(f'Top {top_n} Variables Mas Representativas\n(Importancia por Permutacion)',
               fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
ax3a.set_xlabel('Reduccion media de exactitud', color='#cdd6f4', fontsize=9)
ax3a.axvline(0, color='#6c7086', linewidth=0.8, linestyle='--')
for bar, val in zip(bars_top, top_features.values[::-1]):
    ax3a.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
              f'{val:.4f}', va='center', ha='left', color='#cdd6f4', fontsize=8)

colors_bot = ['#a6e3a1' if v >= 0 else '#fab387' for v in bottom_features.values]
bars_bot = ax3b.barh(bottom_features.index[::-1], bottom_features.values[::-1],
                     color=colors_bot[::-1], edgecolor='#1e1e2e', height=0.65)
ax3b.set_title(f'Top {top_n} Variables Menos Representativas\n(Importancia por Permutacion)',
               fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
ax3b.set_xlabel('Reduccion media de exactitud', color='#cdd6f4', fontsize=9)
ax3b.axvline(0, color='#6c7086', linewidth=0.8, linestyle='--')
for bar, val in zip(bars_bot, bottom_features.values[::-1]):
    ax3b.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
              f'{val:.4f}', va='center', ha='left', color='#cdd6f4', fontsize=8)

fig3.tight_layout(pad=2.5)
embed_figure(tab3, fig3)


tab4 = ttk.Frame(notebook)
notebook.add(tab4, text='  Frontera de Decisión (Significativos)  ')

fig4 = mfigure.Figure(figsize=(10, 6), facecolor='#1e1e2e')
ax4a = fig4.add_subplot(1, 2, 1)
ax4b = fig4.add_subplot(1, 2, 2)

for ax in (ax4a, ax4b):
    ax.set_facecolor('#1e1e2e')
    ax.tick_params(colors='#cdd6f4', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#313244')


y_pos = np.arange(len(vars_significativas))
colores_sig = ['#89b4fa' if v >= 0 else '#f38ba8'
               for v in importancias[vars_significativas].values]
ax4a.barh(y_pos, importancias[vars_significativas].values,
          color=colores_sig, edgecolor='#1e1e2e', height=0.7)
ax4a.set_yticks(y_pos)
ax4a.set_yticklabels(vars_significativas, fontsize=7, color='#cdd6f4')
elim = [c for c in variables_entrada.columns if c not in vars_significativas]
elim_txt = (', '.join(elim[:8]) + (f'... +{len(elim)-8}' if len(elim) > 8 else '')) if elim else 'Ninguna'
ax4a.set_title(
    f'Variables Significativas Retenidas\n'
    f'({len(vars_significativas)} de {len(variables_entrada.columns)}, umbral > {umbral_significancia})',
    fontsize=10, fontweight='bold', color='#cdd6f4', pad=8)
ax4a.set_xlabel(
    f'Importancia por Permutacion\nEliminadas ({len(elim)}): {elim_txt}',
    color='#f38ba8' if elim else '#cdd6f4', fontsize=7)
ax4a.axvline(0, color='#6c7086', linewidth=0.8, linestyle='--')


ax4b.contourf(xx_f, yy_f, Z_f, alpha=0.35, cmap=plt.cm.coolwarm)
sc_sep = ax4b.scatter(X_sep[:, 0], X_sep[:, 1],
                      c=y_entrenar, cmap=plt.cm.coolwarm,
                      edgecolors='#cdd6f4', linewidths=0.4, alpha=0.9, zorder=3)
ax4b.set_xlim(xf_min, xf_max)
ax4b.set_ylim(yf_min, yf_max)
ax4b.set_title(
    f'Frontera de Decision (Significativos)\nSeparacion x{SEPARACION_MULT} (PCA 2D)',
    fontsize=10, fontweight='bold', color='#cdd6f4', pad=8)
ax4b.set_xlabel('Componente Principal 1', color='#cdd6f4', fontsize=8)
ax4b.set_ylabel('Componente Principal 2', color='#cdd6f4', fontsize=8)
leg4 = ax4b.legend(*sc_sep.legend_elements(), title='Termd',
                   facecolor='#313244', labelcolor='#cdd6f4',
                   title_fontsize=8, fontsize=8)
leg4.get_title().set_color('#cdd6f4')
fig4.tight_layout(pad=2.5)
embed_figure(tab4, fig4)


tab5 = ttk.Frame(notebook)
notebook.add(tab5, text='  Metricas de Rendimiento  ')

outer5     = tk.Frame(tab5, bg='#1e1e2e')
outer5.pack(fill='both', expand=True)
canvas5    = tk.Canvas(outer5, bg='#1e1e2e', highlightthickness=0)
scroll_y5  = ttk.Scrollbar(outer5, orient='vertical', command=canvas5.yview)
canvas5.configure(yscrollcommand=scroll_y5.set)
scroll_y5.pack(side='right', fill='y')
canvas5.pack(side='left', fill='both', expand=True)
inner5 = tk.Frame(canvas5, bg='#1e1e2e')
win5   = canvas5.create_window((0, 0), window=inner5, anchor='nw')

inner5.bind('<Configure>', lambda e: canvas5.configure(scrollregion=canvas5.bbox('all')))
canvas5.bind('<Configure>', lambda e: canvas5.itemconfig(win5, width=e.width))
canvas5.bind_all('<MouseWheel>',
    lambda e: canvas5.yview_scroll(int(-1 * (e.delta / 120)), 'units'))

tk.Label(inner5, text='Evaluacion del Rendimiento - Clasificacion',
         bg='#1e1e2e', fg='#89b4fa',
         font=('Segoe UI', 15, 'bold')).pack(pady=(18, 4))
tk.Label(inner5, text='SVM No Lineal (Kernel RBF) - RRHH Dataset',
         bg='#1e1e2e', fg='#6c7086',
         font=('Segoe UI', 10)).pack(pady=(0, 16))


def seccion5(titulo):
    tk.Label(inner5, text=titulo, bg='#1e1e2e', fg='#cba6f7',
             font=('Segoe UI', 12, 'bold')).pack(anchor='w', padx=24, pady=(14, 4))
    tk.Frame(inner5, bg='#313244', height=1).pack(fill='x', padx=24, pady=(0, 8))


seccion5('Valores de la Matriz de Confusion')
for lbl, val in [
    ('TP  (True Positive  - positivos correctos)',  int(tp)),
    ('TN  (True Negative  - negativos correctos)',  int(tn)),
    ('FP  (False Positive - falsos positivos)',     int(fp)),
    ('FN  (False Negative - falsos negativos)',     int(fn)),
    ('P   (Total positivos reales)',                P_total),
    ('N   (Total negativos reales)',                N_total),
    ('P+N (Total muestras de prueba)',              total),
]:
    row = tk.Frame(inner5, bg='#181825')
    row.pack(fill='x', padx=24, pady=2, ipady=5)
    tk.Label(row, text=lbl, bg='#181825', fg='#cdd6f4',
             font=('Consolas', 10), anchor='w').pack(side='left', padx=12)
    tk.Label(row, text=f'{val:>6d}', bg='#181825', fg='#f9e2af',
             font=('Consolas', 11, 'bold')).pack(side='right', padx=14)


seccion5('Metricas Calculadas')
for nombre, formula, valor, color in [
    ('Exactitud  (Accuracy)',        '(TP + TN) / (P + N)',                    exactitud_val,     '#a6e3a1'),
    ('Tasa de Error  (Error Rate)',  '(FP + FN) / (P + N)',                    tasa_error_val,    '#f38ba8'),
    ('Sensibilidad  (Recall / TPR)', 'TP / P',                                 sensibilidad_val,  '#89b4fa'),
    ('Especificidad  (TNR)',         'TN / N',                                 especificidad_val, '#89dceb'),
    ('Precision',                   'TP / (TP + FP)',                          precision_val,     '#cba6f7'),
    ('F1  (Harmonic Mean)',          '2 x Prec x Recall / (Prec + Recall)',    f1_val,            '#fab387'),
    (f'F-beta  (b={beta})',          '(1+b^2) x Prec x Recall / (b^2xPrec+Recall)', fbeta_val,   '#94e2d5'),
]:
    card = tk.Frame(inner5, bg='#181825')
    card.pack(fill='x', padx=24, pady=3, ipady=7)
    left = tk.Frame(card, bg='#181825')
    left.pack(side='left', fill='both', expand=True, padx=12)
    tk.Label(left, text=nombre,  bg='#181825', fg=color,
             font=('Segoe UI', 10, 'bold'), anchor='w').pack(anchor='w')
    tk.Label(left, text=formula, bg='#181825', fg='#6c7086',
             font=('Consolas', 8), anchor='w').pack(anchor='w')
    tk.Label(card, text=f'{valor*100:.2f}%', bg='#181825', fg=color,
             font=('Segoe UI', 14, 'bold')).pack(side='right', padx=18)


seccion5('Matriz de Confusion - Detalle Visual')
fig5_cm = mfigure.Figure(figsize=(4.5, 3.2), facecolor='#181825')
ax5c = fig5_cm.add_subplot(111)
ax5c.set_facecolor('#181825')
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Oranges', ax=ax5c,
            xticklabels=['Activo (0)', 'Terminado (1)'],
            yticklabels=['Activo (0)', 'Terminado (1)'],
            annot_kws={'size': 15, 'weight': 'bold'}, linewidths=0.5)
ax5c.set_title('Matriz de Confusion', fontsize=10, fontweight='bold', color='#cdd6f4', pad=8)
ax5c.set_xlabel('Prediccion', color='#cdd6f4', fontsize=9)
ax5c.set_ylabel('Valor Real',  color='#cdd6f4', fontsize=9)
ax5c.tick_params(colors='#cdd6f4', labelsize=8)
for spine in ax5c.spines.values():
    spine.set_edgecolor('#313244')
fig5_cm.tight_layout(pad=1.5)
cm_frame5 = tk.Frame(inner5, bg='#1e1e2e')
cm_frame5.pack(pady=(4, 16))
canvas_cm5 = FigureCanvasTkAgg(fig5_cm, master=cm_frame5)
canvas_cm5.draw()
canvas_cm5.get_tk_widget().pack()


seccion5('Terminos y Definiciones')
for term, desc in [
    ('P',  'Positivo - numero de casos reales positivos en los datos'),
    ('N',  'Negativo - numero de casos reales negativos en los datos'),
    ('TP', 'True Positive  - positivos clasificados correctamente por el clasificador'),
    ('TN', 'True Negative  - negativos clasificados correctamente por el clasificador'),
    ('FP', 'False Positive - negativos clasificados incorrectamente como positivos'),
    ('FN', 'False Negative - positivos clasificados incorrectamente como negativos'),
]:
    row_t = tk.Frame(inner5, bg='#1e1e2e')
    row_t.pack(fill='x', padx=24, pady=2)
    tk.Label(row_t, text=f' {term} ', bg='#313244', fg='#f9e2af',
             font=('Consolas', 10, 'bold'), width=5).pack(side='left', padx=(0, 10))
    tk.Label(row_t, text=desc, bg='#1e1e2e', fg='#cdd6f4',
             font=('Segoe UI', 9), anchor='w').pack(side='left')


seccion5('Tabla de Formulas de Rendimiento')
tabla_f = tk.Frame(inner5, bg='#181825')
tabla_f.pack(fill='x', padx=24, pady=(0, 16), ipady=4)
for col, (h, w) in enumerate([('Medicion', 28), ('Formula', 38), ('Valor', 10)]):
    tk.Label(tabla_f, text=h, bg='#313244', fg='#89b4fa',
             font=('Segoe UI', 9, 'bold'), width=w, anchor='center').grid(
             row=0, column=col, padx=1, pady=1, sticky='nsew')
for r, (med, form, val) in enumerate([
    ('Exactitud / Reconocimiento',  '(TP+TN)/(P+N)',               f'{exactitud_val*100:.2f}%'),
    ('Tasa de Error',               '(FP+FN)/(P+N)',               f'{tasa_error_val*100:.2f}%'),
    ('Sensibilidad / Recall / TPR', 'TP/P',                        f'{sensibilidad_val*100:.2f}%'),
    ('Especificidad / TNR',         'TN/N',                        f'{especificidad_val*100:.2f}%'),
    ('Precision',                   'TP/(TP+FP)',                   f'{precision_val*100:.2f}%'),
    ('F1 - Media Armonica',         '2xPrecxRecall/(Prec+Recall)', f'{f1_val*100:.2f}%'),
    (f'F-beta (b={beta})',          '(1+b2)xPxR/(b2xP+R)',         f'{fbeta_val*100:.2f}%'),
], start=1):
    bg = '#1e1e2e' if r % 2 == 0 else '#181825'
    tk.Label(tabla_f, text=med,  bg=bg, fg='#cdd6f4', font=('Segoe UI', 9),
             anchor='w', wraplength=220).grid(row=r, column=0, padx=1, pady=1, sticky='nsew')
    tk.Label(tabla_f, text=form, bg=bg, fg='#6c7086', font=('Consolas', 8),
             anchor='w').grid(row=r, column=1, padx=1, pady=1, sticky='nsew')
    tk.Label(tabla_f, text=val,  bg=bg, fg='#f9e2af', font=('Consolas', 10, 'bold'),
             anchor='center').grid(row=r, column=2, padx=1, pady=1, sticky='nsew')

tk.Label(inner5, text='', bg='#1e1e2e').pack(pady=12)

print("\n[INFO] Abriendo ventana interactiva SVM No Lineal...")
root.mainloop()
