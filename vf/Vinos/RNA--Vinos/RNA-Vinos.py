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
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def cargar_datos_vinos():
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    curr = directorio_actual
    ruta_origen = None
    while curr:
        candidato_limpio = os.path.join(curr, 'dataset_vino_limpio.csv')
        candidato_raw    = os.path.join(curr, 'winequality-red.csv')
        candidato_sub    = os.path.join(curr, 'redwine', 'winequality-red.csv')
        if os.path.exists(candidato_limpio):
            ruta_origen = candidato_limpio
            break
        elif os.path.exists(candidato_raw):
            ruta_origen = candidato_raw
            break
        elif os.path.exists(candidato_sub):
            ruta_origen = candidato_sub
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    if not ruta_origen or not os.path.exists(ruta_origen):
        ruta_origen = os.path.join(directorio_actual, 'winequality-red.csv')

    if os.path.exists(ruta_origen):
        df = pd.read_csv(ruta_origen).drop_duplicates().dropna()
        objetivo = df['quality']
        variables_entrada = df.drop('quality', axis=1)
        return variables_entrada, objetivo
    else:

        cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        df_dummy = pd.DataFrame(np.random.randn(500, len(cols)), columns=cols)
        target_dummy = pd.Series(np.random.choice([3, 4, 5, 6, 7, 8], size=500), name='quality')
        return df_dummy, target_dummy

def embed_figure(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas

def crear_interfaz_vinos_rna(parent_widget):
    X_df, y = cargar_datos_vinos()
    feature_names = list(X_df.columns)
    clases_unicas = sorted(y.unique())
    clases_labels = [str(c) for c in clases_unicas]


    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y.values, test_size=0.20, random_state=77, stratify=y.values
    )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    mlp_base = MLPClassifier(hidden_layer_sizes=(16,), max_iter=400, random_state=77)
    mlp_base.fit(X_train, y_train)
    y_pred_base = mlp_base.predict(X_test)


    mlp_opt = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=800,
        random_state=77
    )
    mlp_opt.fit(X_train_scaled, y_train)
    y_pred_opt = mlp_opt.predict(X_test_scaled)


    perm_imp = permutation_importance(mlp_opt, X_test_scaled, y_test, n_repeats=10, random_state=77)
    importancias = pd.Series(perm_imp.importances_mean, index=feature_names).sort_values(ascending=False)
    top_vars = importancias.head(6).index.tolist()


    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    mlp_2d = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, random_state=77)
    mlp_2d.fit(X_train_pca, y_train)


    acc_base = accuracy_score(y_test, y_pred_base)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    err_opt = 1.0 - acc_opt

    rep_opt = classification_report(y_test, y_pred_opt, output_dict=True, zero_division=0)
    prec_macro = rep_opt['macro avg']['precision']
    rec_macro = rep_opt['macro avg']['recall']
    f1_macro = rep_opt['macro avg']['f1-score']

    cm_opt = confusion_matrix(y_test, y_pred_opt)

    spec_list = []
    for i in range(len(cm_opt)):
        tn = cm_opt.sum() - (cm_opt[i, :].sum() + cm_opt[:, i].sum() - cm_opt[i, i])
        fp = cm_opt[:, i].sum() - cm_opt[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        spec_list.append(spec)
    spec_macro = float(np.mean(spec_list))


    notebook = ttk.Notebook(parent_widget)
    notebook.pack(fill='both', expand=True, padx=5, pady=5)

    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)
    tab4 = ttk.Frame(notebook)

    notebook.add(tab1, text="  🔮 Pestaña 1: Predicción e Ingreso de Datos  ")
    notebook.add(tab2, text="  🌐 Pestaña 2: Curva de Loss & Variables Significativas  ")
    notebook.add(tab3, text="  📐 Pestaña 3: Frontera 2D (PCA - Red Neuronal)  ")
    notebook.add(tab4, text="  📋 Pestaña 4: Matriz de Confusión y Métricas  ")


    frame_inputs = ttk.LabelFrame(tab1, text=" Ingreso de Variables Fisicoquímicas del Vino ")
    frame_inputs.pack(fill="x", padx=15, pady=10)

    key_vars = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'density', 'pH']
    defaults_vino = {"alcohol": "10.5", "volatile acidity": "0.52", "sulphates": "0.65", "citric acid": "0.28", "density": "0.996", "pH": "3.31"}
    entries_vino = {}

    for i, f_name in enumerate(key_vars):
        row_idx = i // 2
        col_idx = (i % 2) * 2

        lbl = ttk.Label(frame_inputs, text=f"{f_name.capitalize()}:")
        lbl.grid(row=row_idx, column=col_idx, padx=10, pady=6, sticky="w")

        entry = ttk.Entry(frame_inputs, width=12)
        entry.insert(0, defaults_vino.get(f_name, "1.0"))
        entry.grid(row=row_idx, column=col_idx+1, padx=10, pady=6, sticky="w")
        entries_vino[f_name] = entry

    lbl_res = ttk.Label(frame_inputs, text="Resultado: Esperando entrada...", font=("Segoe UI", 11, "bold"))
    lbl_res.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="w")

    txt_probs = tk.Text(frame_inputs, height=6, width=45, font=("Consolas", 9), bg='#181825', fg='#cdd6f4', bd=1)
    txt_probs.grid(row=3, column=2, columnspan=2, rowspan=2, padx=15, pady=5)

    def realizar_prediccion_vino():
        try:
            vec = np.mean(X_train, axis=0)
            for f in key_vars:
                if f in feature_names:
                    idx_f = feature_names.index(f)
                    vec[idx_f] = float(entries_vino[f].get())
        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor ingresa valores numéricos válidos en los parámetros del vino.")
            return

        vec_scaled = scaler.transform([vec])
        pred_q = mlp_opt.predict(vec_scaled)[0]
        probs = mlp_opt.predict_proba(vec_scaled)[0]

        color_res = "#a6e3a1" if pred_q >= 6 else "#f9e2af" if pred_q == 5 else "#f38ba8"
        lbl_res.config(text=f"Predicción Calidad RNA: {pred_q} / 10", foreground=color_res)

        txt_probs.config(state="normal")
        txt_probs.delete("1.0", tk.END)
        txt_probs.insert(tk.END, "Probabilidades Estimadas por Clase (MLP):\n")
        txt_probs.insert(tk.END, "─" * 42 + "\n")
        for idx_cls, c_val in enumerate(mlp_opt.classes_):
            txt_probs.insert(tk.END, f" • Calidad {c_val}: {probs[idx_cls]*100:6.2f}%\n")
        txt_probs.config(state="disabled")

    btn_pred = ttk.Button(frame_inputs, text="⚡ Realizar Predicción RNA", command=realizar_prediccion_vino)
    btn_pred.grid(row=4, column=0, columnspan=2, pady=10, padx=10)


    frame_plot1 = ttk.LabelFrame(tab1, text=" Distribución del Dataset (Alcohol vs Acidez Volátil) ")
    frame_plot1.pack(expand=True, fill="both", padx=15, pady=5)

    fig1 = mfigure.Figure(figsize=(8, 4), facecolor='#1e1e2e')
    ax1 = fig1.add_subplot(111)
    ax1.set_facecolor('#1e1e2e')

    idx_alc = feature_names.index('alcohol') if 'alcohol' in feature_names else 0
    idx_va = feature_names.index('volatile acidity') if 'volatile acidity' in feature_names else 1

    sc = ax1.scatter(X_train[:, idx_alc], X_train[:, idx_va], c=y_train, cmap='RdYlGn', edgecolor='#cdd6f4', s=45, alpha=0.85)
    ax1.set_title("Distribución de Vinos (Alcohol vs Acidez Volátil)", fontsize=11, fontweight='bold', color='#cdd6f4')
    ax1.set_xlabel("Alcohol (% vol)", fontsize=9, color='#cdd6f4')
    ax1.set_ylabel("Acidez Volátil (g/dm³)", fontsize=9, color='#cdd6f4')
    ax1.tick_params(colors='#cdd6f4')
    ax1.grid(True, linestyle='--', alpha=0.4, color='#313244')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#313244')

    fig1.tight_layout()
    embed_figure(frame_plot1, fig1)
    realizar_prediccion_vino()


    frame_plots_tab2 = ttk.LabelFrame(tab2, text=" Evaluación de Entrenamiento (Backpropagation & Permutación) ")
    frame_plots_tab2.pack(fill="both", expand=True, padx=10, pady=5)

    fig2 = mfigure.Figure(figsize=(10, 4.5), facecolor='#1e1e2e')
    ax2_loss = fig2.add_subplot(1, 2, 1)
    ax2_imp = fig2.add_subplot(1, 2, 2)

    for ax in (ax2_loss, ax2_imp):
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#313244')


    ax2_loss.plot(mlp_opt.loss_curve_, color='#89b4fa', linewidth=2.5, label='Loss (Entropía Cruzada)')
    ax2_loss.set_title("Curva de Pérdida (Loss Curve - Solver Adam/Backprop)", fontsize=10, fontweight='bold', color='#cdd6f4')
    ax2_loss.set_xlabel("Épocas / Iteraciones", fontsize=8, color='#cdd6f4')
    ax2_loss.set_ylabel("Pérdida (Loss)", fontsize=8, color='#cdd6f4')
    ax2_loss.grid(True, linestyle='--', alpha=0.4, color='#313244')
    ax2_loss.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)


    y_pos = np.arange(len(top_vars))
    vals_imp = importancias[top_vars].values
    ax2_imp.barh(y_pos, vals_imp, color='#fab387', edgecolor='#1e1e2e', height=0.6)
    ax2_imp.set_yticks(y_pos)
    ax2_imp.set_yticklabels(top_vars, fontsize=8, color='#cdd6f4')
    ax2_imp.set_title("Top Variables Significativas (Permutation Importance)", fontsize=10, fontweight='bold', color='#cdd6f4')
    ax2_imp.set_xlabel("Disminución en Accuracy", fontsize=8, color='#cdd6f4')
    ax2_imp.axvline(0, color='#6c7086', linewidth=0.8, linestyle='--')

    fig2.tight_layout(pad=2)
    embed_figure(frame_plots_tab2, fig2)


    frame_pca = ttk.LabelFrame(tab3, text=" Visualización de Regiones Aprendidas por la Red Neuronal (PCA 2D) ")
    frame_pca.pack(fill="both", expand=True, padx=10, pady=5)

    fig3 = mfigure.Figure(figsize=(8, 5), facecolor='#1e1e2e')
    ax3 = fig3.add_subplot(111)
    ax3.set_facecolor('#1e1e2e')

    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = mlp_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax3.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.RdYlGn)
    sc3 = ax3.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.RdYlGn, edgecolors='#cdd6f4', linewidths=0.3, alpha=0.9)
    ax3.set_title("Fronteras No Lineales Aprendidas por Perceptrón Multicapa (PCA 2D)", fontsize=11, fontweight='bold', color='#cdd6f4')
    ax3.set_xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color='#cdd6f4', fontsize=9)
    ax3.set_ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color='#cdd6f4', fontsize=9)
    ax3.tick_params(colors='#cdd6f4')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#313244')

    fig3.tight_layout()
    embed_figure(frame_pca, fig3)


    pan4 = ttk.PanedWindow(tab4, orient='horizontal')
    pan4.pack(fill='both', expand=True, padx=5, pady=5)

    left_frame4 = tk.Frame(pan4, bg='#1e1e2e')
    right_frame4 = tk.Frame(pan4, bg='#181825', width=360)
    pan4.add(left_frame4, weight=3)
    pan4.add(right_frame4, weight=2)

    fig4 = mfigure.Figure(figsize=(6, 5), facecolor='#1e1e2e')
    ax4_cm = fig4.add_subplot(111)
    ax4_cm.set_facecolor('#1e1e2e')

    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=ax4_cm,
                xticklabels=clases_labels, yticklabels=clases_labels,
                cbar=False, annot_kws={'size': 11, 'weight': 'bold'})
    ax4_cm.set_title("Matriz de Confusión — RNA Perceptrón Multicapa", fontsize=11, fontweight='bold', color='#cdd6f4', pad=10)
    ax4_cm.set_xlabel("Calidad Predicha", fontsize=9, color='#cdd6f4')
    ax4_cm.set_ylabel("Calidad Real", fontsize=9, color='#cdd6f4')
    ax4_cm.tick_params(colors='#cdd6f4')
    for spine in ax4_cm.spines.values():
        spine.set_edgecolor('#313244')

    fig4.tight_layout()
    embed_figure(left_frame4, fig4)

    card = tk.Frame(right_frame4, bg='#181825', bd=1, relief='solid')
    card.pack(fill='both', expand=True, padx=10, pady=10)

    tk.Label(card, text="Métricas del Modelo RNA", bg='#181825', fg='#89b4fa', font=('Segoe UI', 13, 'bold')).pack(pady=(12, 2))
    tk.Label(card, text="Perceptrón Multicapa (Capas 64-32)", bg='#181825', fg='#6c7086', font=('Segoe UI', 9)).pack(pady=(0, 10))

    metrics_list = [
        ('Exactitud (Accuracy)', '(Muestras Correctas / Total)', acc_opt, '#a6e3a1'),
        ('Tasa de Error Global', '1 - Accuracy', err_opt, '#f38ba8'),
        ('Precisión (Macro Avg)', 'Promedio Precisión por Clase', prec_macro, '#cba6f7'),
        ('Sensibilidad (Recall)', 'Promedio Recall por Clase', rec_macro, '#89b4fa'),
        ('Especificidad (Macro)', 'Promedio TNR por Clase', spec_macro, '#89dceb'),
        ('F1-Score (Macro Avg)', 'Media Armónica Prec/Rec', f1_macro, '#fab387'),
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
    root.title("RNA Perceptrón Multicapa - Vinos")
    root.geometry("1180x820")
    root.configure(bg='#1e1e2e')
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
    style.configure('TNotebook.Tab', background='#313244', foreground='#cdd6f4', padding=[14, 6], font=('Segoe UI', 10, 'bold'))
    style.map('TNotebook.Tab', background=[('selected', '#89b4fa')], foreground=[('selected', '#1e1e2e')])

    crear_interfaz_vinos_rna(root)
    root.mainloop()
