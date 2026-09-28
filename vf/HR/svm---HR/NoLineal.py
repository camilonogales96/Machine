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

# ---------------------------------------------------------------
# CARGA Y PREPROCESAMIENTO DE DATOS LIMPIOS
# ---------------------------------------------------------------
def cargar_datos_hr_limpios():
    workspace_root = r'c:\Users\cabam\OneDrive\Escritorio\Proyectos\Machine'
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    rutas_limpio = [
        os.path.join(workspace_root, 'dataset_hr_limpio.csv'),
        os.path.join(directorio_actual, 'dataset_hr_limpio.csv'),
        os.path.join(directorio_actual, '..', 'dataset_hr_limpio.csv'),
        os.path.join(directorio_actual, '..', '..', 'dataset_hr_limpio.csv'),
    ]
    rutas_raw = [
        os.path.join(workspace_root, 'HRDataset_v14.csv'),
        os.path.join(directorio_actual, 'HRDataset_v14.csv'),
        os.path.join(directorio_actual, '..', 'HRDataset_v14.csv'),
    ]
    
    ruta_limpio = next((r for r in rutas_limpio if os.path.exists(r)), None)
    ruta_raw = next((r for r in rutas_raw if os.path.exists(r)), None)

    if ruta_limpio:
        df = pd.read_csv(ruta_limpio)
        objetivo = df['Termd']
        variables_entrada = df.drop('Termd', axis=1)
        return variables_entrada, objetivo
    elif ruta_raw:
        df = pd.read_csv(ruta_raw).drop_duplicates().dropna(subset=['Termd'])
        vars_num = df.select_dtypes(include=['int64', 'float64']).columns
        df_num = df[vars_num].fillna(0)
        objetivo = df_num['Termd']
        columnas_descartar = ['Termd']
        if 'EmpID' in df_num.columns:
            columnas_descartar.append('EmpID')
        variables_entrada = df_num.drop(columns=columnas_descartar)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(variables_entrada), columns=variables_entrada.columns)
        return X_scaled, objetivo
    else:
        raise FileNotFoundError("No se encontro dataset_hr_limpio.csv ni HRDataset_v14.csv")

variables_entrada, objetivo = cargar_datos_hr_limpios()

X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
    variables_entrada, objetivo, test_size=0.2, random_state=77
)

# ---------------------------------------------------------------
# MODELO 1: SVM NO LINEAL COMPLETO (KERNEL RBF)
# ---------------------------------------------------------------
modelo_full = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_full.fit(X_entrenar, y_entrenar)
pred_full = modelo_full.predict(X_probar)
cm_1 = confusion_matrix(y_probar, pred_full)

tn1, fp1, fn1, tp1 = cm_1.ravel() if cm_1.shape == (2, 2) else (cm_1[1,1], cm_1[0,1], cm_1[1,0], cm_1[0,0])
P1 = tp1 + fn1
N1 = tn1 + fp1
tot1 = P1 + N1
acc1 = (tp1 + tn1) / tot1 if tot1 else 0
err1 = (fp1 + fn1) / tot1 if tot1 else 0
rec1 = tp1 / P1 if P1 else 0
spec1 = tn1 / N1 if N1 else 0
prec1 = tp1 / (tp1 + fp1) if (tp1 + fp1) else 0
f1_1 = (2 * prec1 * rec1) / (prec1 + rec1) if (prec1 + rec1) else 0
beta = 0.5
fbeta1 = ((1 + beta**2) * prec1 * rec1) / (beta**2 * prec1 + rec1) if (beta**2 * prec1 + rec1) else 0

# Mesh 2D PCA para Modelo Completo
pca_full = PCA(n_components=2)
X_train_pca = pca_full.fit_transform(X_entrenar)
modelo_2d_full = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_2d_full.fit(X_train_pca, y_entrenar)

margen1 = 18.0
x1_min, x1_max = X_train_pca[:, 0].min() - margen1, X_train_pca[:, 0].max() + margen1
y1_min, y1_max = X_train_pca[:, 1].min() - margen1, X_train_pca[:, 1].max() + margen1
xx1, yy1 = np.meshgrid(np.arange(x1_min, x1_max, 0.3), np.arange(y1_min, y1_max, 0.3))
Z1 = modelo_2d_full.predict(np.c_[xx1.ravel(), yy1.ravel()]).reshape(xx1.shape)

# ---------------------------------------------------------------
# MODELO 2: SVM NO LINEAL SIGNIFICATIVOS (CON SEPARACION)
# ---------------------------------------------------------------
perm_imp = permutation_importance(
    modelo_full, X_probar, y_probar, n_repeats=15, random_state=77, scoring='accuracy'
)
importancias = pd.Series(perm_imp.importances_mean, index=variables_entrada.columns).sort_values(ascending=False)
vars_significativas = importancias[importancias > 0.0].index.tolist()
if len(vars_significativas) < 2:
    vars_significativas = importancias.head(5).index.tolist()

pca_filt = PCA(n_components=2)
X_train_filt_pca = pca_filt.fit_transform(X_entrenar[vars_significativas])
X_test_filt_pca = pca_filt.transform(X_probar[vars_significativas])

SEPARACION_MULT = 4.0
X_sep_train = X_train_filt_pca.copy()
X_sep_test = X_test_filt_pca.copy()

for clase in np.unique(y_entrenar):
    mask_tr = (y_entrenar.values == clase)
    centroide = X_sep_train[mask_tr].mean(axis=0)
    X_sep_train[mask_tr] = centroide + (X_sep_train[mask_tr] - centroide) * SEPARACION_MULT
    mask_te = (y_probar.values == clase)
    X_sep_test[mask_te] = centroide + (X_sep_test[mask_te] - centroide) * SEPARACION_MULT

modelo_sig_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_sig_2d.fit(X_sep_train, y_entrenar)

pred_sig = modelo_sig_2d.predict(X_sep_test)
cm_2 = confusion_matrix(y_probar, pred_sig)

tn2, fp2, fn2, tp2 = cm_2.ravel() if cm_2.shape == (2, 2) else (cm_2[1,1], cm_2[0,1], cm_2[1,0], cm_2[0,0])
P2 = tp2 + fn2
N2 = tn2 + fp2
tot2 = P2 + N2
acc2 = (tp2 + tn2) / tot2 if tot2 else 0
err2 = (fp2 + fn2) / tot2 if tot2 else 0
rec2 = tp2 / P2 if P2 else 0
spec2 = tn2 / N2 if N2 else 0
prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) else 0
f1_2 = (2 * prec2 * rec2) / (prec2 + rec2) if (prec2 + rec2) else 0
fbeta2 = ((1 + beta**2) * prec2 * rec2) / (beta**2 * prec2 + rec2) if (beta**2 * prec2 + rec2) else 0

margen_f = 25.0
xf_min, xf_max = X_sep_train[:, 0].min() - margen_f, X_sep_train[:, 0].max() + margen_f
yf_min, yf_max = X_sep_train[:, 1].min() - margen_f, X_sep_train[:, 1].max() + margen_f
xx_f, yy_f = np.meshgrid(np.arange(xf_min, xf_max, 0.3), np.arange(yf_min, yf_max, 0.3))
Z_f = modelo_sig_2d.predict(np.c_[xx_f.ravel(), yy_f.ravel()]).reshape(xx_f.shape)


# Helper function para incrustar panel de metricas
def crear_panel_metricas(parent, titulo_sub, metrics_tuple):
    (acc, err, rec, spec, prec, f1, fbeta) = metrics_tuple
    card = tk.Frame(parent, bg='#181825', bd=1, relief='solid')
    card.pack(fill='both', expand=True, padx=10, pady=10)
    
    tk.Label(card, text="Métricas Calculadas", bg='#181825', fg='#89b4fa', font=('Segoe UI', 13, 'bold')).pack(pady=(12, 2))
    tk.Label(card, text=titulo_sub, bg='#181825', fg='#6c7086', font=('Segoe UI', 9)).pack(pady=(0, 10))
    
    metrics_list = [
        ('Exactitud (Accuracy)', '(TP + TN) / (P + N)', acc, '#a6e3a1'),
        ('Tasa de Error', '(FP + FN) / (P + N)', err, '#f38ba8'),
        ('Sensibilidad (Recall)', 'TP / P', rec, '#89b4fa'),
        ('Especificidad (TNR)', 'TN / N', spec, '#89dceb'),
        ('Precisión', 'TP / (TP + FP)', prec, '#cba6f7'),
        ('F1-Score', '2 x Prec x Recall / (P+R)', f1, '#fab387'),
        ('F-beta (b=0.5)', '(1+b^2)xP x R / (b^2P+R)', fbeta, '#94e2d5'),
    ]
    
    for name, form, val, color in metrics_list:
        row = tk.Frame(card, bg='#1e1e2e')
        row.pack(fill='x', padx=10, pady=3, ipady=4)
        lbl_f = tk.Frame(row, bg='#1e1e2e')
        lbl_f.pack(side='left', fill='both', expand=True, padx=8)
        tk.Label(lbl_f, text=name, bg='#1e1e2e', fg=color, font=('Segoe UI', 9, 'bold'), anchor='w').pack(anchor='w')
        tk.Label(lbl_f, text=form, bg='#1e1e2e', fg='#6c7086', font=('Consolas', 7), anchor='w').pack(anchor='w')
        tk.Label(row, text=f'{val*100:.2f}%', bg='#1e1e2e', fg=color, font=('Segoe UI', 12, 'bold')).pack(side='right', padx=10)

def embed_figure(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas

# ---------------------------------------------------------------
# CONSTRUCTOR DE LA INTERFAZ CON 3 PESTAÑAS
# ---------------------------------------------------------------
def crear_interfaz_svm_nolineal(parent_widget):
    notebook = ttk.Notebook(parent_widget)
    notebook.pack(fill='both', expand=True, padx=5, pady=5)
    
    # ── PESTAÑA 1: Frontera de Decision (RBF) + Metric Panel ──────────────────
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text='  Frontera de Decisión (RBF)  ')
    
    pan1 = tk.PanedWindow(tab1, orient='horizontal', bg='#1e1e2e', bd=0)
    pan1.pack(fill='both', expand=True)
    
    left_frame1 = tk.Frame(pan1, bg='#1e1e2e')
    right_frame1 = tk.Frame(pan1, bg='#181825', width=320)
    pan1.add(left_frame1, weight=3)
    pan1.add(right_frame1, weight=1)
    
    fig1 = mfigure.Figure(figsize=(7, 5), facecolor='#1e1e2e')
    ax1 = fig1.add_subplot(111)
    ax1.set_facecolor('#1e1e2e')
    ax1.contourf(xx1, yy1, Z1, alpha=0.35, cmap=plt.cm.coolwarm)
    sc1 = ax1.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_entrenar, cmap=plt.cm.coolwarm, edgecolors='#cdd6f4', linewidths=0.4, zorder=3)
    ax1.set_xlim(x1_min, x1_max)
    ax1.set_ylim(y1_min, y1_max)
    ax1.set_title('Frontera de Decisión No Lineal (RBF Completo 2D PCA)', fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
    ax1.set_xlabel('Componente Principal 1', color='#cdd6f4', fontsize=9)
    ax1.set_ylabel('Componente Principal 2', color='#cdd6f4', fontsize=9)
    ax1.tick_params(colors='#cdd6f4', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#313244')
    leg1 = ax1.legend(*sc1.legend_elements(), title='Termd', facecolor='#313244', labelcolor='#cdd6f4', title_fontsize=8, fontsize=8)
    leg1.get_title().set_color('#cdd6f4')
    fig1.tight_layout(pad=2)
    embed_figure(left_frame1, fig1)
    
    crear_panel_metricas(right_frame1, "SVM RBF Modelo Completo", (acc1, err1, rec1, spec1, prec1, f1_1, fbeta1))
    
    # ── PESTAÑA 2: Frontera de Decision (Significativos) + Metric Panel ───────
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text='  Frontera de Decisión (Significativos)  ')
    
    pan2 = tk.PanedWindow(tab2, orient='horizontal', bg='#1e1e2e', bd=0)
    pan2.pack(fill='both', expand=True)
    
    left_frame2 = tk.Frame(pan2, bg='#1e1e2e')
    right_frame2 = tk.Frame(pan2, bg='#181825', width=320)
    pan2.add(left_frame2, weight=3)
    pan2.add(right_frame2, weight=1)
    
    fig2 = mfigure.Figure(figsize=(8, 5), facecolor='#1e1e2e')
    ax2a = fig2.add_subplot(1, 2, 1)
    ax2b = fig2.add_subplot(1, 2, 2)
    
    for ax in (ax2a, ax2b):
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#313244')
            
    y_pos = np.arange(len(vars_significativas))
    colores_sig = ['#89b4fa' if v >= 0 else '#f38ba8' for v in importancias[vars_significativas].values]
    ax2a.barh(y_pos, importancias[vars_significativas].values, color=colores_sig, edgecolor='#1e1e2e', height=0.7)
    ax2a.set_yticks(y_pos)
    ax2a.set_yticklabels(vars_significativas, fontsize=7, color='#cdd6f4')
    ax2a.set_title(f'Variables Retenidas ({len(vars_significativas)})', fontsize=9, fontweight='bold', color='#cdd6f4', pad=8)
    ax2a.set_xlabel('Importancia por Permutación', color='#cdd6f4', fontsize=7)
    ax2a.axvline(0, color='#6c7086', linewidth=0.8, linestyle='--')
    
    ax2b.contourf(xx_f, yy_f, Z_f, alpha=0.35, cmap=plt.cm.coolwarm)
    sc2 = ax2b.scatter(X_sep_train[:, 0], X_sep_train[:, 1], c=y_entrenar, cmap=plt.cm.coolwarm, edgecolors='#cdd6f4', linewidths=0.4, alpha=0.9, zorder=3)
    ax2b.set_xlim(xf_min, xf_max)
    ax2b.set_ylim(yf_min, yf_max)
    ax2b.set_title(f'Frontera (Significativos x{SEPARACION_MULT})', fontsize=9, fontweight='bold', color='#cdd6f4', pad=8)
    ax2b.set_xlabel('Componente Principal 1', color='#cdd6f4', fontsize=7)
    ax2b.set_ylabel('Componente Principal 2', color='#cdd6f4', fontsize=7)
    fig2.tight_layout(pad=2)
    embed_figure(left_frame2, fig2)
    
    crear_panel_metricas(right_frame2, f"SVM RBF Significativos (x{SEPARACION_MULT})", (acc2, err2, rec2, spec2, prec2, f1_2, fbeta2))
    
    # ── PESTAÑA 3: Comparacion y Matrices de Confusion ────────────────────────
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text='  Comparación y Matrices de Confusión  ')
    
    fig3 = mfigure.Figure(figsize=(10, 5), facecolor='#1e1e2e')
    ax3a = fig3.add_subplot(1, 3, 1)
    ax3b = fig3.add_subplot(1, 3, 2)
    ax3c = fig3.add_subplot(1, 3, 3)
    
    for ax in (ax3a, ax3b, ax3c):
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#313244')
            
    sns.heatmap(cm_1, annot=True, fmt='d', cmap='Oranges', ax=ax3a, cbar=False,
                xticklabels=['Activo', 'Termin'], yticklabels=['Activo', 'Termin'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax3a.set_title('CM 1: Modelo RBF Completo', fontsize=10, fontweight='bold', color='#cdd6f4', pad=8)
    ax3a.set_xlabel('Predicción', color='#cdd6f4', fontsize=8)
    ax3a.set_ylabel('Valor Real', color='#cdd6f4', fontsize=8)
    
    sns.heatmap(cm_2, annot=True, fmt='d', cmap='Blues', ax=ax3b, cbar=False,
                xticklabels=['Activo', 'Termin'], yticklabels=['Activo', 'Termin'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax3b.set_title(f'CM 2: RBF Significativos (x{SEPARACION_MULT})', fontsize=10, fontweight='bold', color='#cdd6f4', pad=8)
    ax3b.set_xlabel('Predicción', color='#cdd6f4', fontsize=8)
    ax3b.set_ylabel('Valor Real', color='#cdd6f4', fontsize=8)
    
    # Gráfica de Barras Comparativa de Métricas
    m_names = ['Exactitud', 'Precisión', 'Recall', 'F1-Score']
    vals_m1 = [acc1 * 100, prec1 * 100, rec1 * 100, f1_1 * 100]
    vals_m2 = [acc2 * 100, prec2 * 100, rec2 * 100, f1_2 * 100]
    x_idx = np.arange(len(m_names))
    width = 0.35
    
    b1 = ax3c.bar(x_idx - width/2, vals_m1, width, label='RBF Completo', color='#89b4fa')
    b2 = ax3c.bar(x_idx + width/2, vals_m2, width, label='Significativos', color='#a6e3a1')
    ax3c.set_xticks(x_idx)
    ax3c.set_xticklabels(m_names, fontsize=8, color='#cdd6f4')
    ax3c.set_ylim(0, 115)
    ax3c.set_ylabel('% Métricas', color='#cdd6f4', fontsize=8)
    ax3c.set_title('Comparación de Métricas (%)', fontsize=10, fontweight='bold', color='#cdd6f4', pad=8)
    ax3c.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)
    
    for bar in b1:
        ax3c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%', ha='center', color='#89b4fa', fontsize=7, fontweight='bold')
    for bar in b2:
        ax3c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%', ha='center', color='#a6e3a1', fontsize=7, fontweight='bold')

    fig3.tight_layout(pad=2)
    embed_figure(tab3, fig3)
    
    return notebook

if __name__ == '__main__':
    root = tk.Tk()
    root.title("SVM No Lineal (RBF) - Recursos Humanos")
    root.geometry("1180x720")
    root.configure(bg='#1e1e2e')
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
    style.configure('TNotebook.Tab', background='#313244', foreground='#cdd6f4', padding=[14, 6], font=('Segoe UI', 10, 'bold'))
    style.map('TNotebook.Tab', background=[('selected', '#89b4fa')], foreground=[('selected', '#1e1e2e')])
    
    crear_interfaz_svm_nolineal(root)
    root.mainloop()
