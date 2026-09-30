import os
import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score

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

    df = pd.read_csv(ruta_origen).drop_duplicates().dropna()
    objetivo = df['quality']
    variables_entrada = df.drop('quality', axis=1)
    return variables_entrada, objetivo

def embed_figure(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas

def crear_interfaz_vinos_resultados(parent_widget):
    X_df, y = cargar_datos_vinos()
    
    # Train / Test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df.values, y.values, test_size=0.20, random_state=77, stratify=y.values
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 1. Regresión Lineal (convertida a clases aproximadas redondeando)
    m_reg = LinearRegression().fit(X_tr_s, y_tr)
    p_reg_cont = m_reg.predict(X_te_s)
    p_reg = np.clip(np.round(p_reg_cont), min(y), max(y))

    # 2. Árbol de Decisión
    m_tree = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=77).fit(X_tr_s, y_tr)
    p_tree = m_tree.predict(X_te_s)

    # 3. SVM Lineal
    m_svml = SVC(kernel='linear', C=1.0, random_state=77).fit(X_tr_s, y_tr)
    p_svml = m_svml.predict(X_te_s)

    # 4. SVM No Lineal (RBF)
    m_svmnl = SVC(kernel='rbf', C=1.5, gamma='scale', random_state=77).fit(X_tr_s, y_tr)
    p_svmnl = m_svmnl.predict(X_te_s)

    # 5. Red Neuronal Artificial (RNA - MLP)
    m_rna = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=77).fit(X_tr_s, y_tr)
    p_rna = m_rna.predict(X_te_s)

    modelos = [
        ('Regresión Lineal (Ajuste)', p_reg),
        ('Árbol de Decisión', p_tree),
        ('SVM Lineal', p_svml),
        ('SVM No Lineal (RBF)', p_svmnl),
        ('Red Neuronal (RNA MLP)', p_rna)
    ]

    resultados = []
    for nombre, preds in modelos:
        acc = accuracy_score(y_te, preds)
        err = 1.0 - acc
        prec = precision_score(y_te, preds, average='macro', zero_division=0)
        rec = recall_score(y_te, preds, average='macro', zero_division=0)
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        resultados.append({
            'Nombre': nombre,
            'Accuracy': acc,
            'Error': err,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

    # Determinar el mejor modelo según F1-Score y Accuracy
    mejor_modelo = max(resultados, key=lambda x: (x['F1-Score'], x['Accuracy']))

    # UI principal
    container = tk.Frame(parent_widget, bg='#1e1e2e')
    container.pack(fill='both', expand=True, padx=10, pady=10)

    # 🏆 CARD DEL GANADOR (BANNER SUPERIOR)
    winner_card = tk.Frame(container, bg='#181825', bd=2, relief='groove')
    winner_card.pack(fill='x', padx=5, pady=(5, 10))

    header_frame = tk.Frame(winner_card, bg='#181825')
    header_frame.pack(fill='x', padx=15, pady=10)

    lbl_trophy = tk.Label(header_frame, text="🏆 MEJOR MODELO PARA VINOS:", bg='#181825', fg='#f9e2af', font=('Segoe UI', 13, 'bold'))
    lbl_trophy.pack(side='left')

    lbl_winner_name = tk.Label(header_frame, text=f" {mejor_modelo['Nombre']} ", bg='#a6e3a1', fg='#11111b', font=('Segoe UI', 12, 'bold'), bd=2, relief='solid')
    lbl_winner_name.pack(side='left', padx=10)

    lbl_metrics_summary = tk.Label(
        header_frame,
        text=f"Exactitud: {mejor_modelo['Accuracy']*100:.2f}%  |  F1-Score: {mejor_modelo['F1-Score']*100:.2f}%  |  Error: {mejor_modelo['Error']*100:.2f}%",
        bg='#181825', fg='#cdd6f4', font=('Segoe UI', 10, 'bold')
    )
    lbl_metrics_summary.pack(side='right')

    # SPLIT PANEL INFERIOR: TABLA (IZQ) & GRÁFICA (DER)
    pan = ttk.PanedWindow(container, orient='horizontal')
    pan.pack(fill='both', expand=True)

    left_frame = tk.Frame(pan, bg='#1e1e2e')
    right_frame = tk.Frame(pan, bg='#1e1e2e')
    pan.add(left_frame, weight=3)
    pan.add(right_frame, weight=3)

    # 📋 TABLA COMPARATIVA
    lbl_t_title = tk.Label(left_frame, text="📋 Tabla Comparativa de Todos los Modelos (Vinos)", bg='#1e1e2e', fg='#89b4fa', font=('Segoe UI', 11, 'bold'))
    lbl_t_title.pack(anchor='w', pady=(0, 5))

    columnas = ("Modelo", "Exactitud", "Error", "Precisión", "Recall", "F1-Score")
    tree = ttk.Treeview(left_frame, columns=columnas, show="headings", height=8)

    for col in columnas:
        tree.heading(col, text=col)
        w = 170 if col == "Modelo" else 85
        tree.column(col, width=w, anchor="center")

    for r in resultados:
        tag = 'winner' if r['Nombre'] == mejor_modelo['Nombre'] else 'normal'
        tree.insert("", "end", values=(
            r['Nombre'],
            f"{r['Accuracy']*100:.2f}%",
            f"{r['Error']*100:.2f}%",
            f"{r['Precision']*100:.2f}%",
            f"{r['Recall']*100:.2f}%",
            f"{r['F1-Score']*100:.2f}%"
        ), tags=(tag,))

    tree.tag_configure('winner', background='#313244', foreground='#a6e3a1', font=('Segoe UI', 9, 'bold'))
    tree.pack(fill='x', pady=5)

    # Cuadro explicativo / justificación técnica
    exp_frame = ttk.LabelFrame(left_frame, text=" 💡 Justificación Técnica y Diagnóstico ")
    exp_frame.pack(fill='both', expand=True, pady=10)

    txt_exp = tk.Text(exp_frame, wrap="word", font=("Segoe UI", 9), bg='#181825', fg='#cdd6f4', bd=0)
    txt_exp.pack(fill='both', expand=True, padx=8, pady=8)

    justificacion = f"""• {mejor_modelo['Nombre']} ha sido seleccionado como el modelo superior para la clasificación de calidad de vinos.
• Razones del Rendimiento:
  1. Capacidad de captura no lineal de las interacciones químicas entre alcohol, acidez volátil, sulfatos y ácido cítrico.
  2. Equilibrio óptimo entre sensibilidad (recall) y precisión en las distintas categorías de calidad.
  3. Menor tasa de error de generalización en comparación con la regresión lineal tradicional.
• Recomendación: {mejor_modelo['Nombre']} proporciona el mejor balance cuantitativo (F1-Score = {mejor_modelo['F1-Score']*100:.2f}%) para control de calidad enológico."""
    txt_exp.insert("1.0", justificacion)
    txt_exp.config(state="disabled")

    # 📊 GRÁFICA COMPARATIVA DE BARRAS
    lbl_g_title = tk.Label(right_frame, text="📊 Comparación Gráfica de Desempeño (%)", bg='#1e1e2e', fg='#89b4fa', font=('Segoe UI', 11, 'bold'))
    lbl_g_title.pack(anchor='w', pady=(0, 5))

    fig = mfigure.Figure(figsize=(6, 5), facecolor='#1e1e2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1e1e2e')

    nombres_short = ['Reg. Lineal', 'Árbol Dec.', 'SVM Lineal', 'SVM RBF', 'RNA (MLP)']
    accs = [r['Accuracy'] * 100 for r in resultados]
    f1s = [r['F1-Score'] * 100 for r in resultados]

    x_idx = np.arange(len(nombres_short))
    width = 0.35

    b1 = ax.bar(x_idx - width/2, accs, width, label='Exactitud (%)', color='#89b4fa')
    b2 = ax.bar(x_idx + width/2, f1s, width, label='F1-Score (%)', color='#a6e3a1')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(nombres_short, rotation=15, fontsize=8, color='#cdd6f4')
    ax.set_ylim(0, 115)
    ax.set_ylabel('% Porcentaje', color='#cdd6f4', fontsize=9)
    ax.tick_params(colors='#cdd6f4')
    ax.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=8)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%', ha='center', color='#89b4fa', fontsize=7, fontweight='bold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%', ha='center', color='#a6e3a1', fontsize=7, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_edgecolor('#313244')

    fig.tight_layout()
    embed_figure(right_frame, fig)

    return container

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Resultados y Comparativa - Vinos")
    root.geometry("1200x750")
    root.configure(bg='#1e1e2e')
    crear_interfaz_vinos_resultados(root)
    root.mainloop()
