import warnings
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS HR
# ==========================================
def cargar_datos_hr():
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    curr = directorio_actual
    ruta_limpio = None
    ruta_raw = None

    while curr:
        cand_limpio = os.path.join(curr, 'dataset_hr_limpio.csv')
        cand_raw = os.path.join(curr, 'HRDataset_v14.csv')
        if not ruta_limpio and os.path.exists(cand_limpio):
            ruta_limpio = cand_limpio
        if not ruta_raw and os.path.exists(cand_raw):
            ruta_raw = cand_raw
        if ruta_limpio or ruta_raw:
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    if not ruta_limpio and not ruta_raw:
        cand_limpio_cwd = os.path.join(os.getcwd(), 'dataset_hr_limpio.csv')
        cand_raw_cwd = os.path.join(os.getcwd(), 'HRDataset_v14.csv')
        if os.path.exists(cand_limpio_cwd):
            ruta_limpio = cand_limpio_cwd
        elif os.path.exists(cand_raw_cwd):
            ruta_raw = cand_raw_cwd

    if ruta_limpio:
        df = pd.read_csv(ruta_limpio)
        y = df['Termd'].values
        X = df.drop(columns=['Termd'])
        return X, y
    elif ruta_raw:
        df_raw = pd.read_csv(ruta_raw).drop_duplicates().dropna(subset=['Termd'])
        num_cols = df_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
        df_num = df_raw[num_cols].fillna(0)
        y = df_num['Termd'].values
        cols_drop = ['Termd']
        if 'EmpID' in df_num.columns:
            cols_drop.append('EmpID')
        X = df_num.drop(columns=cols_drop)
        return X, y
    else:
        # Fallback dummy dataset si no hay archivo
        X = pd.DataFrame(np.random.randn(300, 5), columns=['Salary', 'EngagementSurvey', 'EmpSatisfaction', 'Absences', 'SpecialProjectsCount'])
        y = np.random.choice([0, 1], size=300)
        return X, y

def embed_figure(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas

def crear_interfaz_hr_rna(parent_widget):
    X_df, y = cargar_datos_hr()
    feature_names = list(X_df.columns)
    target_names = ['Activo (0)', 'Terminado (1)']

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Modelo Base (Sin escalado)
    mlp_base = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)
    mlp_base.fit(X_train, y_train)
    y_pred_base = mlp_base.predict(X_test)

    # 2. Modelo Optimizado (Con StandardScaler + Perceptrón Multicapa Backpropagation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_opt = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    mlp_opt.fit(X_train_scaled, y_train)
    y_pred_opt = mlp_opt.predict(X_test_scaled)

    # Variables significativas usando Importancia por Permutación
    perm_imp = permutation_importance(mlp_opt, X_test_scaled, y_test, n_repeats=10, random_state=42)
    importancias = pd.Series(perm_imp.importances_mean, index=feature_names).sort_values(ascending=False)
    top_vars = importancias.head(5).index.tolist()

    # Cálculo de Métricas de Evaluación
    acc_b = accuracy_score(y_test, y_pred_base)
    acc_o = accuracy_score(y_test, y_pred_opt)

    prec_b = precision_score(y_test, y_pred_base, average='weighted', zero_division=0)
    prec_o = precision_score(y_test, y_pred_opt, average='weighted', zero_division=0)

    rec_b = recall_score(y_test, y_pred_base, average='weighted', zero_division=0)
    rec_o = recall_score(y_test, y_pred_opt, average='weighted', zero_division=0)

    f1_b = f1_score(y_test, y_pred_base, average='weighted', zero_division=0)
    f1_o = f1_score(y_test, y_pred_opt, average='weighted', zero_division=0)

    cm_opt = confusion_matrix(y_test, y_pred_opt)
    cm_base = confusion_matrix(y_test, y_pred_base)

    tn, fp, fn, tp = cm_opt.ravel() if cm_opt.shape == (2, 2) else (cm_opt[1,1], cm_opt[0,1], cm_opt[1,0], cm_opt[0,0])
    spec_o = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    err_o = 1.0 - acc_o
    err_b = 1.0 - acc_b

    # ==========================================
    # CONSTRUCCIÓN DE LA INTERFAZ CON NOTEBOOK
    # ==========================================
    notebook = ttk.Notebook(parent_widget)
    notebook.pack(fill='both', expand=True, padx=5, pady=5)

    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)

    notebook.add(tab1, text="  🎨 Pestaña 1: Predicción e Ingreso de Datos  ")
    notebook.add(tab2, text="  📈 Pestaña 2: Pérdida (Loss) y Variables Significativas  ")
    notebook.add(tab3, text="  📋 Pestaña 3: Métricas y Matriz de Confusión  ")

    # ---------------------------------------------------------------
    # PESTAÑA 1: PREDICCIÓN INTERACTIVA Y DISPERSIÓN DE DATOS
    # ---------------------------------------------------------------
    frame_inputs = ttk.LabelFrame(tab1, text=" Ingreso de Variables del Empleado ")
    frame_inputs.pack(fill="x", padx=15, pady=10)

    entries = {}
    main_inter_vars = ['Salary', 'EngagementSurvey', 'EmpSatisfaction', 'Absences']
    defaults = {"Salary": "65000", "EngagementSurvey": "4.5", "EmpSatisfaction": "4.0", "Absences": "5.0"}

    for i, feature in enumerate(main_inter_vars):
        lbl = ttk.Label(frame_inputs, text=f"{feature}:")
        lbl.grid(row=i, column=0, padx=10, pady=6, sticky="w")
        
        entry = ttk.Entry(frame_inputs, width=15)
        entry.insert(0, defaults.get(feature, "1.0"))
        entry.grid(row=i, column=1, padx=10, pady=6, sticky="w")
        entries[feature] = entry

    lbl_resultado = ttk.Label(frame_inputs, text="Resultado: Esperando entrada...", font=("Segoe UI", 11, "bold"))
    lbl_resultado.grid(row=0, column=2, columnspan=2, padx=30, pady=5, sticky="w")

    txt_probabilidades = tk.Text(frame_inputs, height=5, width=45, font=("Consolas", 9), bg='#181825', fg='#cdd6f4', bd=1)
    txt_probabilidades.grid(row=1, column=2, columnspan=2, rowspan=3, padx=30, pady=5)

    def realizar_prediccion():
        try:
            # Crear vector con promedios y reemplazar los ingresados
            vec = np.mean(X_train, axis=0)
            for f in main_inter_vars:
                if f in feature_names:
                    idx_f = feature_names.index(f)
                    vec[idx_f] = float(entries[f].get())
        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor ingresa valores numéricos válidos en los campos.")
            return

        datos_scaled = scaler.transform([vec])
        pred_class = mlp_opt.predict(datos_scaled)[0]
        probs = mlp_opt.predict_proba(datos_scaled)[0]

        nombre_clase = "PERMANECE (ACTIVO)" if pred_class == 0 else "DESVINCULADO (TERMINADO)"
        color_resultado = "#a6e3a1" if pred_class == 0 else "#f38ba8"
        lbl_resultado.config(text=f"Predicción RNA: {nombre_clase}", foreground=color_resultado)

        txt_probabilidades.config(state="normal")
        txt_probabilidades.delete("1.0", tk.END)
        txt_probabilidades.insert(tk.END, "Probabilidades del Modelo MLP:\n")
        txt_probabilidades.insert(tk.END, "─" * 40 + "\n")
        txt_probabilidades.insert(tk.END, f" • Activo (Clase 0):    {probs[0]*100:6.2f}%\n")
        txt_probabilidades.insert(tk.END, f" • Terminado (Clase 1): {probs[1]*100:6.2f}%\n")
        txt_probabilidades.config(state="disabled")

    btn_predecir = ttk.Button(frame_inputs, text="⚡ Realizar Predicción RNA", command=realizar_prediccion)
    btn_predecir.grid(row=4, column=0, columnspan=2, pady=10, padx=10)

    # Dispersión decorativa
    frame_plot1 = ttk.LabelFrame(tab1, text=" Distribución del Dataset (Salario vs Compromiso Laboral) ")
    frame_plot1.pack(expand=True, fill="both", padx=15, pady=5)

    fig1 = mfigure.Figure(figsize=(8, 4), facecolor='#1e1e2e')
    ax1 = fig1.add_subplot(111)
    ax1.set_facecolor('#1e1e2e')

    idx_sal = feature_names.index('Salary') if 'Salary' in feature_names else 0
    idx_eng = feature_names.index('EngagementSurvey') if 'EngagementSurvey' in feature_names else 1

    sc = ax1.scatter(X_train[:, idx_sal], X_train[:, idx_eng], c=y_train, cmap='coolwarm', edgecolor='#cdd6f4', s=50, alpha=0.85)
    ax1.set_title("Distribución de Empleados (Salario vs Nivel de Compromiso)", fontsize=11, fontweight='bold', color='#cdd6f4')
    ax1.set_xlabel("Salario ($)", fontsize=9, color='#cdd6f4')
    ax1.set_ylabel("Engagement Survey", fontsize=9, color='#cdd6f4')
    ax1.tick_params(colors='#cdd6f4')
    ax1.grid(True, linestyle='--', alpha=0.4, color='#313244')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#313244')

    fig1.tight_layout()
    embed_figure(frame_plot1, fig1)
    realizar_prediccion()

    # ---------------------------------------------------------------
    # PESTAÑA 2: CURVA DE PÉRDIDA Y VARIABLES SIGNIFICATIVAS
    # ---------------------------------------------------------------
    frame_plots_tab2 = ttk.LabelFrame(tab2, text=" Análisis de Convergencia Backpropagation e Importancia de Variables ")
    frame_plots_tab2.pack(fill="both", expand=True, padx=10, pady=5)

    fig2 = mfigure.Figure(figsize=(10, 4.5), facecolor='#1e1e2e')
    ax2_loss = fig2.add_subplot(1, 2, 1)
    ax2_imp = fig2.add_subplot(1, 2, 2)

    for ax in (ax2_loss, ax2_imp):
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#313244')

    # Subplot 1: Curva de Pérdida
    ax2_loss.plot(mlp_opt.loss_curve_, color='#89b4fa', linewidth=2.5, label='Pérdida (Loss)')
    ax2_loss.set_title("Curva de Pérdida (Loss Curve - Backpropagation)", fontsize=10, fontweight='bold', color='#cdd6f4')
    ax2_loss.set_xlabel("Épocas / Iteraciones", fontsize=8, color='#cdd6f4')
    ax2_loss.set_ylabel("Pérdida (Cross-Entropy Loss)", fontsize=8, color='#cdd6f4')
    ax2_loss.grid(True, linestyle='--', alpha=0.4, color='#313244')
    ax2_loss.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)

    # Subplot 2: Variables Significativas
    y_pos = np.arange(len(top_vars))
    vals_imp = importancias[top_vars].values
    ax2_imp.barh(y_pos, vals_imp, color='#a6e3a1', edgecolor='#1e1e2e', height=0.6)
    ax2_imp.set_yticks(y_pos)
    ax2_imp.set_yticklabels(top_vars, fontsize=8, color='#cdd6f4')
    ax2_imp.set_title("Top 5 Variables Significativas (Permutation Importance)", fontsize=10, fontweight='bold', color='#cdd6f4')
    ax2_imp.set_xlabel("Importancia Relativa", fontsize=8, color='#cdd6f4')
    ax2_imp.axvline(0, color='#6c7086', linewidth=0.8, linestyle='--')

    fig2.tight_layout(pad=2)
    embed_figure(frame_plots_tab2, fig2)

    # ---------------------------------------------------------------
    # PESTAÑA 3: MATRIZ DE CONFUSIÓN Y MÉTRICAS
    # ---------------------------------------------------------------
    pan3 = ttk.PanedWindow(tab3, orient='horizontal')
    pan3.pack(fill='both', expand=True, padx=5, pady=5)

    left_frame3 = tk.Frame(pan3, bg='#1e1e2e')
    right_frame3 = tk.Frame(pan3, bg='#181825', width=360)
    pan3.add(left_frame3, weight=3)
    pan3.add(right_frame3, weight=2)

    fig3 = mfigure.Figure(figsize=(6, 5), facecolor='#1e1e2e')
    ax3_cm = fig3.add_subplot(111)
    ax3_cm.set_facecolor('#1e1e2e')

    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues', ax=ax3_cm,
                xticklabels=['Activo', 'Terminado'], yticklabels=['Activo', 'Terminado'],
                cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    ax3_cm.set_title("Matriz de Confusión (RNA Optimizada MLP)", fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
    ax3_cm.set_xlabel("Clase Predicha", fontsize=9, color='#cdd6f4')
    ax3_cm.set_ylabel("Clase Real", fontsize=9, color='#cdd6f4')
    ax3_cm.tick_params(colors='#cdd6f4')
    for spine in ax3_cm.spines.values():
        spine.set_edgecolor('#313244')

    fig3.tight_layout()
    embed_figure(left_frame3, fig3)

    # Panel de métricas a la derecha
    card = tk.Frame(right_frame3, bg='#181825', bd=1, relief='solid')
    card.pack(fill='both', expand=True, padx=10, pady=10)

    tk.Label(card, text="Métricas Calculadas", bg='#181825', fg='#89b4fa', font=('Segoe UI', 13, 'bold')).pack(pady=(12, 2))
    tk.Label(card, text="Red Neuronal Artificial (Perceptrón Multicapa)", bg='#181825', fg='#6c7086', font=('Segoe UI', 9)).pack(pady=(0, 10))

    metrics_list = [
        ('Exactitud (Accuracy)', '(TP + TN) / Total', acc_o, '#a6e3a1'),
        ('Tasa de Error', '1 - Accuracy', err_o, '#f38ba8'),
        ('Sensibilidad (Recall)', 'TP / (TP + FN)', rec_o, '#89b4fa'),
        ('Especificidad (TNR)', 'TN / (TN + FP)', spec_o, '#89dceb'),
        ('Precisión', 'TP / (TP + FP)', prec_o, '#cba6f7'),
        ('F1-Score', '2 x Prec x Recall / (P+R)', f1_o, '#fab387'),
    ]

    for name, form, val, color in metrics_list:
        row = tk.Frame(card, bg='#1e1e2e')
        row.pack(fill='x', padx=10, pady=4, ipady=4)
        lbl_f = tk.Frame(row, bg='#1e1e2e')
        lbl_f.pack(side='left', fill='both', expand=True, padx=8)
        tk.Label(lbl_f, text=name, bg='#1e1e2e', fg=color, font=('Segoe UI', 9, 'bold'), anchor='w').pack(anchor='w')
        tk.Label(lbl_f, text=form, bg='#1e1e2e', fg='#6c7086', font=('Consolas', 7), anchor='w').pack(anchor='w')
        tk.Label(row, text=f'{val*100:.2f}%', bg='#1e1e2e', fg=color, font=('Segoe UI', 12, 'bold')).pack(side='right', padx=10)

    return notebook

if __name__ == '__main__':
    root = tk.Tk()
    root.title("RNA Perceptrón Multicapa - Recursos Humanos")
    root.geometry("1180x820")
    root.configure(bg='#1e1e2e')
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
    style.configure('TNotebook.Tab', background='#313244', foreground='#cdd6f4', padding=[14, 6], font=('Segoe UI', 10, 'bold'))
    style.map('TNotebook.Tab', background=[('selected', '#89b4fa')], foreground=[('selected', '#1e1e2e')])

    crear_interfaz_hr_rna(root)
    root.mainloop()
