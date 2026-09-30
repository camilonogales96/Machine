import warnings
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)


import os

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
    candidato_cwd = os.path.join(os.getcwd(), 'HRDataset_v14.csv')
    if os.path.exists(candidato_cwd):
        ruta_origen = candidato_cwd
    else:
        ruta_origen = os.path.join(directorio_actual, 'HRDataset_v14.csv')

df_raw = pd.read_csv(ruta_origen)


feature_names = ['Salary', 'EngagementSurvey', 'EmpSatisfaction', 'Absences']
target_col = 'Termd'


target_names = ['Activo', 'Terminado']


df = df_raw[feature_names + [target_col]].dropna().reset_index(drop=True)

X = df[feature_names].values
y = df[target_col].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


mlp_base = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)
mlp_base.fit(X_train, y_train)
y_pred_base = mlp_base.predict(X_test)


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


acc_b = accuracy_score(y_test, y_pred_base)
acc_o = accuracy_score(y_test, y_pred_opt)

prec_b = precision_score(y_test, y_pred_base, average='weighted', zero_division=0)
prec_o = precision_score(y_test, y_pred_opt, average='weighted', zero_division=0)

rec_b = recall_score(y_test, y_pred_base, average='weighted', zero_division=0)
rec_o = recall_score(y_test, y_pred_opt, average='weighted', zero_division=0)

f1_b = f1_score(y_test, y_pred_base, average='weighted', zero_division=0)
f1_o = f1_score(y_test, y_pred_opt, average='weighted', zero_division=0)

cm_opt = confusion_matrix(y_test, y_pred_opt)


root = tk.Tk()
root.title("Proyecto RNA - Predicción de Rotación Laboral (MLP)")
root.geometry("1180x920")

style = ttk.Style()
style.theme_use('clam')

notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)

notebook.add(tab1, text=" 🎨 Pestaña 1: Predicción e Ingreso de Datos (Usuario) ")
notebook.add(tab2, text=" 📋 Pestaña 2: Gráficas de Validación, Matriz de Confusión y Métricas ")
notebook.pack(expand=1, fill="both")


frame_inputs = ttk.LabelFrame(tab1, text=" Ingreso de Variables del Empleado ")
frame_inputs.pack(fill="x", padx=15, pady=10)

entries = {}

valores_defecto = ["65000", "4.5", "4", "5"]

for i, feature in enumerate(feature_names):
    lbl = ttk.Label(frame_inputs, text=f"{feature.capitalize()}:")
    lbl.grid(row=i, column=0, padx=10, pady=8, sticky="w")

    entry = ttk.Entry(frame_inputs, width=15)
    entry.insert(0, valores_defecto[i])
    entry.grid(row=i, column=1, padx=10, pady=8, sticky="w")
    entries[feature] = entry

lbl_resultado = ttk.Label(frame_inputs, text="Resultado: Esperando entrada...", font=("Arial", 11, "bold"))
lbl_resultado.grid(row=0, column=2, columnspan=2, padx=30, pady=5, sticky="w")

txt_probabilidades = tk.Text(frame_inputs, height=5, width=45, font=("Arial", 9))
txt_probabilidades.grid(row=1, column=2, columnspan=2, rowspan=3, padx=30, pady=5)

def realizar_prediccion():
    try:
        valores = [float(entries[f].get()) for f in feature_names]
    except ValueError:
        messagebox.showerror("Error de Entrada", "Por favor ingresa valores numéricos válidos en todos los campos.")
        return


    datos_scaled = scaler.transform([valores])
    pred_class = mlp_opt.predict(datos_scaled)[0]
    probs = mlp_opt.predict_proba(datos_scaled)[0]

    nombre_clase = target_names[pred_class].upper()
    color_resultado = "green" if pred_class == 0 else "red"
    lbl_resultado.config(text=f"Predicción de Estado: {nombre_clase}", foreground=color_resultado)

    txt_probabilidades.config(state="normal")
    txt_probabilidades.delete("1.0", tk.END)
    txt_probabilidades.insert(tk.END, "Porcentajes de Confianza:\n")
    txt_probabilidades.insert(tk.END, "-" * 35 + "\n")
    for idx, c_name in enumerate(target_names):
        txt_probabilidades.insert(tk.END, f" • {c_name.capitalize()}: {probs[idx]*100:.2f}%\n")
    txt_probabilidades.config(state="disabled")

btn_predecir = ttk.Button(frame_inputs, text="⚡ Realizar Predicción", command=realizar_prediccion)
btn_predecir.grid(row=4, column=0, columnspan=2, pady=10, padx=10)


frame_plot1 = ttk.LabelFrame(tab1, text=" Visualización del Conjunto de Datos (HR Dataset) ")
frame_plot1.pack(expand=True, fill="both", padx=15, pady=5)

fig1, ax1 = plt.subplots(figsize=(8, 4))
canvas1 = FigureCanvasTkAgg(fig1, master=frame_plot1)
canvas1.get_tk_widget().pack(expand=True, fill="both")


scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50, alpha=0.8)
ax1.set_title("Distribución de Clases (Salario vs Nivel de Compromiso)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Salario ($)", fontsize=9)
ax1.set_ylabel("Engagement Survey (Puntuación)", fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.5)


frame_plots_tab2 = ttk.LabelFrame(tab2, text=" Evaluación Gráfica: Curva de Pérdida y Matriz de Confusión ")
frame_plots_tab2.pack(fill="both", expand=True, padx=10, pady=5)

fig2, (ax2_loss, ax2_cm) = plt.subplots(1, 2, figsize=(10, 3.8))
canvas2 = FigureCanvasTkAgg(fig2, master=frame_plots_tab2)
canvas2.get_tk_widget().pack(expand=True, fill="both")


ax2_loss.plot(mlp_opt.loss_curve_, color='#1f77b4', linewidth=2.5)
ax2_loss.set_title("Curva de Pérdida (Loss Curve - Backpropagation)", fontsize=10, fontweight='bold')
ax2_loss.set_xlabel("Épocas / Iteraciones", fontsize=8)
ax2_loss.set_ylabel("Pérdida (Loss)", fontsize=8)
ax2_loss.grid(True, linestyle='--', alpha=0.5)


sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues', ax=ax2_cm,
            xticklabels=target_names, yticklabels=target_names, cbar=False)
ax2_cm.set_title("Matriz de Confusión (RNA Optimizada)", fontsize=10, fontweight='bold')
ax2_cm.set_xlabel("Clase Predicha", fontsize=8)
ax2_cm.set_ylabel("Clase Real", fontsize=8)

fig2.tight_layout()
canvas2.draw()


frame_t2_middle = ttk.LabelFrame(tab2, text=" Comparación de Métricas de Validación ")
frame_t2_middle.pack(fill="x", padx=10, pady=5)

columnas = ("Métrica", "Modelo Base (Sin Escalar / 1 Capa)", "Modelo Optimizado (StandardScaler / MLP 16-8)")
tree = ttk.Treeview(frame_t2_middle, columns=columnas, show="headings", height=7)

tree.heading("Métrica", text="Métrica de Evaluación")
tree.heading("Modelo Base (Sin Escalar / 1 Capa)", text="Modelo Base")
tree.heading("Modelo Optimizado (StandardScaler / MLP 16-8)", text="Modelo RNA Optimizado")

tree.column("Métrica", width=340)
tree.column("Modelo Base (Sin Escalar / 1 Capa)", width=230, anchor="center")
tree.column("Modelo Optimizado (StandardScaler / MLP 16-8)", width=230, anchor="center")

filas_metricas = [
    ("Exactitud / Accuracy ((TP+TN)/(P+N))", f"{acc_b * 100:.2f}%", f"{acc_o * 100:.2f}%"),
    ("Precisión Ponderada / Precision", f"{prec_b * 100:.2f}%", f"{prec_o * 100:.2f}%"),
    ("Sensibilidad / Recall", f"{rec_b * 100:.2f}%", f"{rec_o * 100:.2f}%"),
    ("Puntuación F1 / F1-Score", f"{f1_b * 100:.2f}%", f"{f1_o * 100:.2f}%"),
    ("Épocas de Entrenamiento", f"{mlp_base.n_iter_}", f"{mlp_opt.n_iter_}"),
    ("Optimizador de Aprendizaje", "Adam", "Adam (Con Backpropagation)"),
    ("Diagnóstico de Convergencia", "Sensible a escalas no estandarizadas", "Convergencia suave y óptima en Loss")
]

for fila in filas_metricas:
    tree.insert("", "end", values=fila)

tree.pack(fill="x", padx=5, pady=5)


frame_info = ttk.LabelFrame(tab2, text=" Explicación del Modelo de Red Neuronal ")
frame_info.pack(fill="x", padx=10, pady=5)

txt_info = tk.Text(frame_info, wrap="word", font=("Arial", 9), height=4)
txt_info.pack(fill="both", expand=True, padx=5, pady=5)

explicacion = """• Perceptrón Multicapa (MLP): Red neuronal Feedforward que predice la probabilidad de que un empleado sea desvinculado (Terminado) basándose en su salario, compromiso, satisfacción y ausencias.
• Algoritmo Backpropagation: Calcula el gradiente del error ajustando los pesos sinápticos de atrás hacia adelante en cada época.
• Impacto del StandardScaler: Las redes neuronales son altamente sensibles a la escala de las variables (como Salario vs Ausencias); la estandarización acelera la convergencia del algoritmo Adam."""

txt_info.insert("1.0", explicacion)
txt_info.config(state="disabled")


root.mainloop()
