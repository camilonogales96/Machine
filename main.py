import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# ---------------------------------------------------------------
# IMPORTAR O EJECUTAR LIMPIEZA AUTOMATICA
# ---------------------------------------------------------------
try:
    import data_cleaner
    data_cleaner.ejecutar_limpieza_completa()
except Exception as e:
    print(f"[ADVERTENCIA] No se pudo ejecutar data_cleaner directamente: {e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_hr_data():
    workspace_root = r'c:\Users\cabam\OneDrive\Escritorio\Proyectos\Machine'
    ruta = os.path.join(workspace_root, 'dataset_hr_limpio.csv')
    if not os.path.exists(ruta):
        ruta_raw = os.path.join(workspace_root, 'HRDataset_v14.csv')
        df = pd.read_csv(ruta_raw).drop_duplicates().dropna(subset=['Termd'])
        vars_num = df.select_dtypes(include=['int64', 'float64']).columns
        df_num = df[vars_num].fillna(0)
        obj = df_num['Termd']
        cols_desc = ['Termd']
        if 'EmpID' in df_num.columns:
            cols_desc.append('EmpID')
        X_in = df_num.drop(columns=cols_desc)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_in), columns=X_in.columns)
        return X_scaled, obj
    df = pd.read_csv(ruta)
    y = df['Termd']
    X = df.drop('Termd', axis=1)
    return X, y

def get_vino_data():
    workspace_root = r'c:\Users\cabam\OneDrive\Escritorio\Proyectos\Machine'
    ruta = os.path.join(workspace_root, 'dataset_vino_limpio.csv')
    if not os.path.exists(ruta):
        ruta_raw = os.path.join(workspace_root, 'winequality-red.csv')
        df = pd.read_csv(ruta_raw).drop_duplicates().dropna()
        y = df['quality']
        X_in = df.drop('quality', axis=1)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_in), columns=X_in.columns)
        return X_scaled, y
    df = pd.read_csv(ruta)
    y = df['quality']
    X = df.drop('quality', axis=1)
    return X, y

def helper_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = cm[0, 0]
        fp = cm[0, 1:].sum() if cm.shape[1] > 1 else 0
        fn = cm[1:, 0].sum() if cm.shape[0] > 1 else 0
        tn = cm[1:, 1:].sum() if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    P = tp + fn
    N = tn + fp
    tot = P + N
    acc = (tp + tn) / tot if tot else accuracy_score(y_true, y_pred)
    err = 1.0 - acc
    rec = tp / P if P else 0
    spec = tn / N if N else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
    beta = 0.5
    fbeta = ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec) if (beta**2 * prec + rec) else 0
    return {
        'acc': acc, 'err': err, 'rec': rec, 'spec': spec,
        'prec': prec, 'f1': f1, 'fbeta': fbeta, 'cm': cm
    }

def embed_figure(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas

def crear_panel_metricas_card(parent, titulo_sub, m_dict):
    card = tk.Frame(parent, bg='#181825', bd=1, relief='solid')
    card.pack(fill='both', expand=True, padx=8, pady=8)
    
    tk.Label(card, text="Métricas Calculadas", bg='#181825', fg='#89b4fa', font=('Segoe UI', 12, 'bold')).pack(pady=(10, 2))
    tk.Label(card, text=titulo_sub, bg='#181825', fg='#6c7086', font=('Segoe UI', 8)).pack(pady=(0, 8))
    
    metrics_list = [
        ('Exactitud (Accuracy)', '(TP + TN) / (P + N)', m_dict['acc'], '#a6e3a1'),
        ('Tasa de Error', '(FP + FN) / (P + N)', m_dict['err'], '#f38ba8'),
        ('Sensibilidad (Recall)', 'TP / P', m_dict['rec'], '#89b4fa'),
        ('Especificidad (TNR)', 'TN / N', m_dict['spec'], '#89dceb'),
        ('Precisión', 'TP / (TP + FP)', m_dict['prec'], '#cba6f7'),
        ('F1-Score', '2 x Prec x Recall / (P+R)', m_dict['f1'], '#fab387'),
        ('F-beta (b=0.5)', '(1+b^2)xP x R / (b^2P+R)', m_dict['fbeta'], '#94e2d5'),
    ]
    
    for name, form, val, color in metrics_list:
        row = tk.Frame(card, bg='#1e1e2e')
        row.pack(fill='x', padx=8, pady=2, ipady=3)
        lbl_f = tk.Frame(row, bg='#1e1e2e')
        lbl_f.pack(side='left', fill='both', expand=True, padx=6)
        tk.Label(lbl_f, text=name, bg='#1e1e2e', fg=color, font=('Segoe UI', 8, 'bold'), anchor='w').pack(anchor='w')
        tk.Label(lbl_f, text=form, bg='#1e1e2e', fg='#6c7086', font=('Consolas', 7), anchor='w').pack(anchor='w')
        tk.Label(row, text=f'{val*100:.2f}%', bg='#1e1e2e', fg=color, font=('Segoe UI', 11, 'bold')).pack(side='right', padx=8)

# ===============================================================
# VISTAS DE VINOS
# ===============================================================
def vista_vinos_regresion(parent):
    X, y = get_vino_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=77)
    
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    fig = mfigure.Figure(figsize=(9, 5), facecolor='#1e1e2e')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for ax in (ax1, ax2):
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#313244')
        
    ax1.scatter(X_te['alcohol'], y_te, color='#89b4fa', alpha=0.6, label='Valores Reales')
    ax1.set_title('Alcohol vs Calidad del Vino', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Alcohol (Normalizado)', color='#cdd6f4', fontsize=8)
    ax1.set_ylabel('Calidad (3 a 8)', color='#cdd6f4', fontsize=8)
    ax1.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)
    
    ax2.scatter(y_te, preds, color='#cba6f7', alpha=0.6, label='Predicciones')
    ax2.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', label='Ajuste Perfecto')
    ax2.set_title('Calidad Real vs Predicha (Regresión Multivariable)', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Calidad Real', color='#cdd6f4', fontsize=8)
    ax2.set_ylabel('Calidad Predicha', color='#cdd6f4', fontsize=8)
    ax2.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)
    
    fig.tight_layout(pad=2)
    embed_figure(parent, fig)

def vista_vinos_arbol(parent):
    X, y = get_vino_data()
    y_bin = (y >= 6).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.2, random_state=77)
    
    model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=77)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    
    fig = mfigure.Figure(figsize=(9, 5), facecolor='#1e1e2e')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for ax in (ax1, ax2):
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#313244')
        
    plot_tree(model, feature_names=X.columns, class_names=['Baja/Media', 'Alta'], filled=True, ax=ax1, fontsize=6)
    ax1.set_title('Estructura del Árbol de Decisión (Vinos)', color='#cdd6f4', fontsize=10, fontweight='bold')
    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    ax2.barh(importances.index, importances.values, color='#a6e3a1', edgecolor='#1e1e2e')
    ax2.set_title('Importancia de Variables (Vinos)', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Gini Importance', color='#cdd6f4', fontsize=8)
    
    fig.tight_layout(pad=2)
    embed_figure(parent, fig)

def vista_vinos_svm(parent):
    from sklearn.decomposition import PCA
    X, y = get_vino_data()
    y_bin = (y >= 6).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.2, random_state=77)
    
    pca = PCA(n_components=2)
    X_tr_pca = pca.fit_transform(X_tr)
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
    model.fit(X_tr_pca, y_tr)
    
    x_min, x_max = X_tr_pca[:, 0].min() - 2, X_tr_pca[:, 0].max() + 2
    y_min, y_max = X_tr_pca[:, 1].min() - 2, X_tr_pca[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig = mfigure.Figure(figsize=(8, 5), facecolor='#1e1e2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#181825')
    ax.tick_params(colors='#cdd6f4', labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor('#313244')
    
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
    sc = ax.scatter(X_tr_pca[:, 0], X_tr_pca[:, 1], c=y_tr, cmap=plt.cm.coolwarm, edgecolors='#cdd6f4', linewidths=0.4, zorder=3)
    ax.set_title('SVM RBF 2D PCA - Calidad de Vinos (Baja vs Alta)', color='#cdd6f4', fontsize=11, fontweight='bold')
    ax.set_xlabel('Componente Principal 1', color='#cdd6f4', fontsize=9)
    ax.set_ylabel('Componente Principal 2', color='#cdd6f4', fontsize=9)
    leg = ax.legend(*sc.legend_elements(), title='Calidad >= 6', facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)
    leg.get_title().set_color('#cdd6f4')
    
    fig.tight_layout(pad=2)
    embed_figure(parent, fig)


# ===============================================================
# VISTAS DE RECURSOS HUMANOS (HR)
# ===============================================================
def vista_hr_regresion(parent):
    X, y = get_hr_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=77)
    
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    fig = mfigure.Figure(figsize=(9, 5), facecolor='#1e1e2e')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for ax in (ax1, ax2):
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#313244')
        
    ax1.scatter(X_te.iloc[:, 0], y_te, color='#f38ba8', alpha=0.7, label='Muestras Reales')
    ax1.set_title('Variable Predictora Principal vs Termd (HR)', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax1.set_xlabel(X.columns[0], color='#cdd6f4', fontsize=8)
    ax1.set_ylabel('Termd (0=Activo, 1=Terminado)', color='#cdd6f4', fontsize=8)
    ax1.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)
    
    ax2.scatter(y_te, preds, color='#89b4fa', alpha=0.7, label='Predicciones')
    ax2.set_title('Ajuste Lineal de Clasificación (HR)', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Termd Real', color='#cdd6f4', fontsize=8)
    ax2.set_ylabel('Valor Predicho', color='#cdd6f4', fontsize=8)
    ax2.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)
    
    fig.tight_layout(pad=2)
    embed_figure(parent, fig)

def vista_hr_arbol(parent):
    X, y = get_hr_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=77)
    
    model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=77)
    model.fit(X_tr, y_tr)
    
    fig = mfigure.Figure(figsize=(9, 5), facecolor='#1e1e2e')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for ax in (ax1, ax2):
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#313244')
        
    plot_tree(model, feature_names=X.columns, class_names=['Activo', 'Terminado'], filled=True, ax=ax1, fontsize=5)
    ax1.set_title('Árbol de Decisión RRHH (max_depth=4)', color='#cdd6f4', fontsize=10, fontweight='bold')
    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    ax2.barh(importances.index, importances.values, color='#fab387', edgecolor='#1e1e2e')
    ax2.set_title('Importancia de Variables (HR)', color='#cdd6f4', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Gini Importance', color='#cdd6f4', fontsize=8)
    
    fig.tight_layout(pad=2)
    embed_figure(parent, fig)

def vista_hr_svm_3tabs(parent):
    """Implementa las 3 pestañas requeridas en el Punto 4 para SVM HR."""
    try:
        from scratch.nolineal_3tabs import crear_interfaz_svm_nolineal
        crear_interfaz_svm_nolineal(parent)
    except Exception as e:
        X, y = get_hr_data()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=77)
        m = SVC(kernel='rbf', C=1.0, random_state=77).fit(X_tr, y_tr)
        p = m.predict(X_te)
        dict_m = helper_metrics(y_te, p)
        crear_panel_metricas_card(parent, "SVM RBF Modelo Completo (HR)", dict_m)

# ===============================================================
# VISTA RESULTADOS (COMPARACION GLOBAL)
# ===============================================================
def vista_resultados(parent):
    notebook = ttk.Notebook(parent)
    notebook.pack(fill='both', expand=True, padx=5, pady=5)
    
    # ── PESTAÑA 1: RESULTADOS VINOS ───────────────────────────────────────────
    tab_vinos = ttk.Frame(notebook)
    notebook.add(tab_vinos, text='  Resultados - Vinos  ')
    
    Xv, yv = get_vino_data()
    yv_bin = (yv >= 6).astype(int)
    Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(Xv, yv_bin, test_size=0.2, random_state=77)
    
    # Model 1: Regresion
    m_reg_v = LinearRegression().fit(Xv_tr, yv_tr)
    p_reg_v = (m_reg_v.predict(Xv_te) >= 0.5).astype(int)
    met_reg_v = helper_metrics(yv_te, p_reg_v)
    
    # Model 2: Arbol
    m_arb_v = DecisionTreeClassifier(max_depth=4, random_state=77).fit(Xv_tr, yv_tr)
    p_arb_v = m_arb_v.predict(Xv_te)
    met_arb_v = helper_metrics(yv_te, p_arb_v)
    
    # Model 3: SVM
    m_svm_v = SVC(kernel='rbf', C=1.0, random_state=77).fit(Xv_tr, yv_tr)
    p_svm_v = m_svm_v.predict(Xv_te)
    met_svm_v = helper_metrics(yv_te, p_svm_v)
    
    top_f_v = tk.Frame(tab_vinos, bg='#1e1e2e')
    top_f_v.pack(fill='both', expand=True, padx=10, pady=10)
    
    tk.Label(top_f_v, text="Tabla Comparativa de Métricas - Dataset Vinos", bg='#1e1e2e', fg='#89b4fa', font=('Segoe UI', 13, 'bold')).pack(anchor='w', pady=(0, 6))
    
    cols = ['Modelo', 'Exactitud', 'Tasa Error', 'Sensibilidad', 'Especificidad', 'Precisión', 'F1-Score']
    table_f_v = tk.Frame(top_f_v, bg='#181825', bd=1, relief='solid')
    table_f_v.pack(fill='x', pady=5)
    
    h_row = tk.Frame(table_f_v, bg='#313244')
    h_row.pack(fill='x', ipady=4)
    for c in cols:
        tk.Label(h_row, text=c, bg='#313244', fg='#cdd6f4', font=('Segoe UI', 9, 'bold'), width=14, anchor='center').pack(side='left', expand=True)
        
    data_v_rows = [
        ('Regresión Lineal', met_reg_v),
        ('Árbol de Decisión', met_arb_v),
        ('SVM (RBF)', met_svm_v)
    ]
    for m_name, m_d in data_v_rows:
        r_f = tk.Frame(table_f_v, bg='#181825')
        r_f.pack(fill='x', ipady=4)
        tk.Label(r_f, text=m_name, bg='#181825', fg='#89b4fa', font=('Segoe UI', 9, 'bold'), width=14, anchor='center').pack(side='left', expand=True)
        for val in [m_d['acc'], m_d['err'], m_d['rec'], m_d['spec'], m_d['prec'], m_d['f1']]:
            tk.Label(r_f, text=f"{val*100:.2f}%", bg='#181825', fg='#cdd6f4', font=('Consolas', 9), width=14, anchor='center').pack(side='left', expand=True)
            
    fig_v = mfigure.Figure(figsize=(9, 4), facecolor='#1e1e2e')
    ax_v = fig_v.add_subplot(111)
    ax_v.set_facecolor('#181825')
    ax_v.tick_params(colors='#cdd6f4', labelsize=8)
    for s in ax_v.spines.values(): s.set_edgecolor('#313244')
    
    m_labels = ['Exactitud', 'Precisión', 'Recall', 'F1-Score']
    x_idx = np.arange(len(m_labels))
    w = 0.25
    
    v_reg = [met_reg_v['acc']*100, met_reg_v['prec']*100, met_reg_v['rec']*100, met_reg_v['f1']*100]
    v_arb = [met_arb_v['acc']*100, met_arb_v['prec']*100, met_arb_v['rec']*100, met_arb_v['f1']*100]
    v_svm = [met_svm_v['acc']*100, met_svm_v['prec']*100, met_svm_v['rec']*100, met_svm_v['f1']*100]
    
    ax_v.bar(x_idx - w, v_reg, w, label='Regresión', color='#89b4fa')
    ax_v.bar(x_idx,     v_arb, w, label='Árbol Decisión', color='#a6e3a1')
    ax_v.bar(x_idx + w, v_svm, w, label='SVM RBF', color='#cba6f7')
    
    ax_v.set_xticks(x_idx)
    ax_v.set_xticklabels(m_labels, color='#cdd6f4', fontsize=9)
    ax_v.set_ylim(0, 115)
    ax_v.set_ylabel('% Rendimiento', color='#cdd6f4', fontsize=8)
    ax_v.set_title('Comparación de Métricas (%) - Modelos Vinos', color='#cdd6f4', fontsize=11, fontweight='bold', pad=10)
    ax_v.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)
    fig_v.tight_layout(pad=2)
    embed_figure(top_f_v, fig_v)
    
    # ── PESTAÑA 2: RESULTADOS RECURSOS HUMANOS ─────────────────────────────────
    tab_hr = ttk.Frame(notebook)
    notebook.add(tab_hr, text='  Resultados - Recursos Humanos (HR)  ')
    
    Xh, yh = get_hr_data()
    Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.2, random_state=77)
    
    # Regresion HR
    m_reg_h = LinearRegression().fit(Xh_tr, yh_tr)
    p_reg_h = (m_reg_h.predict(Xh_te) >= 0.5).astype(int)
    met_reg_h = helper_metrics(yh_te, p_reg_h)
    
    # Arbol HR
    m_arb_h = DecisionTreeClassifier(max_depth=4, random_state=77).fit(Xh_tr, yh_tr)
    p_arb_h = m_arb_h.predict(Xh_te)
    met_arb_h = helper_metrics(yh_te, p_arb_h)
    
    # SVM HR
    m_svm_h = SVC(kernel='rbf', C=1.0, random_state=77).fit(Xh_tr, yh_tr)
    p_svm_h = m_svm_h.predict(Xh_te)
    met_svm_h = helper_metrics(yh_te, p_svm_h)
    
    top_f_h = tk.Frame(tab_hr, bg='#1e1e2e')
    top_f_h.pack(fill='both', expand=True, padx=10, pady=10)
    
    tk.Label(top_f_h, text="Tabla Comparativa de Métricas - Dataset Recursos Humanos (HR)", bg='#1e1e2e', fg='#89b4fa', font=('Segoe UI', 13, 'bold')).pack(anchor='w', pady=(0, 6))
    
    table_f_h = tk.Frame(top_f_h, bg='#181825', bd=1, relief='solid')
    table_f_h.pack(fill='x', pady=5)
    
    h_row2 = tk.Frame(table_f_h, bg='#313244')
    h_row2.pack(fill='x', ipady=4)
    for c in cols:
        tk.Label(h_row2, text=c, bg='#313244', fg='#cdd6f4', font=('Segoe UI', 9, 'bold'), width=14, anchor='center').pack(side='left', expand=True)
        
    data_h_rows = [
        ('Regresión HR', met_reg_h),
        ('Árbol de Decisión HR', met_arb_h),
        ('SVM HR (RBF)', met_svm_h)
    ]
    for m_name, m_d in data_h_rows:
        r_f = tk.Frame(table_f_h, bg='#181825')
        r_f.pack(fill='x', ipady=4)
        tk.Label(r_f, text=m_name, bg='#181825', fg='#89b4fa', font=('Segoe UI', 9, 'bold'), width=14, anchor='center').pack(side='left', expand=True)
        for val in [m_d['acc'], m_d['err'], m_d['rec'], m_d['spec'], m_d['prec'], m_d['f1']]:
            tk.Label(r_f, text=f"{val*100:.2f}%", bg='#181825', fg='#cdd6f4', font=('Consolas', 9), width=14, anchor='center').pack(side='left', expand=True)
            
    fig_h = mfigure.Figure(figsize=(9, 4), facecolor='#1e1e2e')
    ax_h = fig_h.add_subplot(111)
    ax_h.set_facecolor('#181825')
    ax_h.tick_params(colors='#cdd6f4', labelsize=8)
    for s in ax_h.spines.values(): s.set_edgecolor('#313244')
    
    h_reg = [met_reg_h['acc']*100, met_reg_h['prec']*100, met_reg_h['rec']*100, met_reg_h['f1']*100]
    h_arb = [met_arb_h['acc']*100, met_arb_h['prec']*100, met_arb_h['rec']*100, met_arb_h['f1']*100]
    h_svm = [met_svm_h['acc']*100, met_svm_h['prec']*100, met_svm_h['rec']*100, met_svm_h['f1']*100]
    
    ax_h.bar(x_idx - w, h_reg, w, label='Regresión', color='#f38ba8')
    ax_h.bar(x_idx,     h_arb, w, label='Árbol Decisión', color='#fab387')
    ax_h.bar(x_idx + w, h_svm, w, label='SVM RBF', color='#89b4fa')
    
    ax_h.set_xticks(x_idx)
    ax_h.set_xticklabels(m_labels, color='#cdd6f4', fontsize=9)
    ax_h.set_ylim(0, 115)
    ax_h.set_ylabel('% Rendimiento', color='#cdd6f4', fontsize=8)
    ax_h.set_title('Comparación de Métricas (%) - Modelos HR', color='#cdd6f4', fontsize=11, fontweight='bold', pad=10)
    ax_h.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)
    fig_h.tight_layout(pad=2)
    embed_figure(top_f_h, fig_h)

# ===============================================================
# APLICACION PRINCIPAL (MAIN GUI)
# ===============================================================
class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Unificado de Aprendizaje Automático - Vinos & Recursos Humanos")
        self.root.geometry("1280x820")
        self.root.configure(bg='#1e1e2e')
        
        # Style setup
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
        self.style.configure('TNotebook.Tab', background='#313244', foreground='#cdd6f4', padding=[14, 6], font=('Segoe UI', 10, 'bold'))
        self.style.map('TNotebook.Tab', background=[('selected', '#89b4fa')], foreground=[('selected', '#1e1e2e')])
        
        # Main Layout: Sidebar Left, Container Right
        self.sidebar = tk.Frame(root, bg='#181825', width=220)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)
        
        self.container = tk.Frame(root, bg='#1e1e2e')
        self.container.pack(side='right', fill='both', expand=True)
        
        # Sidebar Header
        lbl_title = tk.Label(self.sidebar, text="🤖 Machine\nLearning", bg='#181825', fg='#89b4fa', font=('Segoe UI', 15, 'bold'), justify='center')
        lbl_title.pack(pady=(20, 15))
        
        tk.Frame(self.sidebar, bg='#313244', height=1).pack(fill='x', padx=15, pady=(0, 15))
        
        # Menu Buttons
        self.btn_vinos = self.create_menu_btn("🍷  Vinos", lambda: self.show_view('vinos'))
        self.btn_hr    = self.create_menu_btn("👥  Recursos Humanos", lambda: self.show_view('hr'))
        self.btn_res   = self.create_menu_btn("📊  Resultados", lambda: self.show_view('resultados'))
        
        # Status footer
        status_lbl = tk.Label(self.sidebar, text="[✓] Datasets Limpios\nHR & Vinos Cargados", bg='#181825', fg='#a6e3a1', font=('Consolas', 8), justify='center')
        status_lbl.pack(side='bottom', pady=15)
        
        self.active_btn = None
        self.current_frame = None
        
        # Vista por defecto
        self.show_view('vinos')

    def create_menu_btn(self, text, command):
        btn = tk.Button(
            self.sidebar, text=text, command=command,
            bg='#181825', fg='#cdd6f4', activebackground='#313244', activeforeground='#89b4fa',
            font=('Segoe UI', 11, 'bold'), bd=0, relief='flat', anchor='w', padx=20, pady=12
        )
        btn.pack(fill='x', pady=2)
        return btn

    def set_active_btn(self, target_btn):
        for btn in (self.btn_vinos, self.btn_hr, self.btn_res):
            btn.config(bg='#181825', fg='#cdd6f4')
        target_btn.config(bg='#89b4fa', fg='#1e1e2e')

    def clear_container(self):
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = tk.Frame(self.container, bg='#1e1e2e')
        self.current_frame.pack(fill='both', expand=True)

    def show_view(self, name):
        self.clear_container()
        if name == 'vinos':
            self.set_active_btn(self.btn_vinos)
            nb = ttk.Notebook(self.current_frame)
            nb.pack(fill='both', expand=True, padx=5, pady=5)
            
            t1 = ttk.Frame(nb); nb.add(t1, text='  Regresión  '); vista_vinos_regresion(t1)
            t2 = ttk.Frame(nb); nb.add(t2, text='  Árbol de Decisión  '); vista_vinos_arbol(t2)
            t3 = ttk.Frame(nb); nb.add(t3, text='  SVM Vinos  '); vista_vinos_svm(t3)
            
        elif name == 'hr':
            self.set_active_btn(self.btn_hr)
            nb = ttk.Notebook(self.current_frame)
            nb.pack(fill='both', expand=True, padx=5, pady=5)
            
            t1 = ttk.Frame(nb); nb.add(t1, text='  Regresión  '); vista_hr_regresion(t1)
            t2 = ttk.Frame(nb); nb.add(t2, text='  Árbol de Decisión  '); vista_hr_arbol(t2)
            t3 = ttk.Frame(nb); nb.add(t3, text='  SVM HR (3 Pestañas)  '); vista_hr_svm_3tabs(t3)
            
        elif name == 'resultados':
            self.set_active_btn(self.btn_res)
            vista_resultados(self.current_frame)

if __name__ == '__main__':
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()
