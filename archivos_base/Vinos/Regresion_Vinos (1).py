import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
)


def cargar_datos():
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    curr = directorio_actual
    ruta_origen = None

    while curr:
        candidato = os.path.join(curr, 'redwine', 'winequality-red.csv')
        candidato2 = os.path.join(curr, 'winequality-red.csv')
        if os.path.exists(candidato):
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
        messagebox.showerror("Error", f"No se encontró el archivo winequality-red.csv en:\n{ruta_origen}")
        return None

    df = pd.read_csv(ruta_origen)
    df = df.drop_duplicates()
    return df

datos_vino = cargar_datos()
if datos_vino is None:
    exit()


root = tk.Tk()
root.title("Proyecto de Regresión y Clasificación - Calidad de Vinos")
root.geometry("1180x920")

style = ttk.Style()
style.theme_use('clam')

notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)

notebook.add(tab1, text=" 🎨 Pestaña 1: Gráfica de Zonas y Multiplicador ")
notebook.add(tab2, text=" 📋 Pestaña 2: Gráficas de Validación, Matriz de Confusión y Métricas ")
notebook.pack(expand=1, fill="both")


frame_controles = ttk.LabelFrame(tab1, text=" Controles de Ajuste ")
frame_controles.pack(fill="x", padx=10, pady=5)

lbl_mult = ttk.Label(frame_controles, text="Multiplicador de Alcohol:")
lbl_mult.pack(side="left", padx=10, pady=5)

val_mult = tk.DoubleVar(value=1.5)
lbl_val_mult = ttk.Label(frame_controles, text="1.5x")

def actualizar_label_mult(val):
    lbl_val_mult.config(text=f"{float(val):.1f}x")
    actualizar_grafico1()

slider_mult = ttk.Scale(frame_controles, from_=0.5, to=3.0, value=1.5, variable=val_mult, command=actualizar_label_mult)
slider_mult.pack(side="left", padx=5, pady=5)
lbl_val_mult.pack(side="left", padx=5)

lbl_umbral = ttk.Label(frame_controles, text="Umbral de Calidad:")
lbl_umbral.pack(side="left", padx=(20, 5), pady=5)

entry_umbral = ttk.Entry(frame_controles, width=10)
entry_umbral.insert(0, "6.5")
entry_umbral.pack(side="left", padx=5, pady=5)

frame_plot1 = ttk.Frame(tab1)
frame_plot1.pack(expand=True, fill="both", padx=10, pady=5)

fig1, ax1 = plt.subplots(figsize=(9, 4.5))
canvas1 = FigureCanvasTkAgg(fig1, master=frame_plot1)
canvas1.get_tk_widget().pack(expand=True, fill="both")

def actualizar_grafico1():
    ax1.clear()
    mult = val_mult.get()

    try:
        umbral = float(entry_umbral.get())
    except ValueError:
        umbral = 6.5

    X_unaria = datos_vino[['alcohol']].values * mult
    y_real = datos_vino['quality'].values

    mod = LinearRegression().fit(X_unaria, y_real)
    x_rango = np.linspace(X_unaria.min() - 0.5, X_unaria.max() + 0.5, 200).reshape(-1, 1)
    y_pred_rango = mod.predict(x_rango)

    mask_verde = y_real >= umbral
    mask_rojo = y_real < umbral

    x_min, x_max = x_rango.min(), x_rango.max()
    y_max = y_real.max() + 0.5
    y_min = y_real.min() - 0.5

    ax1.fill_between([x_min, x_max], umbral, y_max, color='green', alpha=0.12, label='Zona Verde (Vino Premium >= Umbral)')
    ax1.fill_between([x_min, x_max], y_min, umbral, color='red', alpha=0.12, label='Zona Roja (Vino Regular < Umbral)')

    ax1.scatter(X_unaria[mask_verde], y_real[mask_verde], color='green', edgecolors='black', s=45, alpha=0.8, label='Vinos Premium')
    ax1.scatter(X_unaria[mask_rojo], y_real[mask_rojo], color='red', edgecolors='black', s=45, alpha=0.8, label='Vinos Regulares')

    ax1.plot(x_rango, y_pred_rango, color='black', linestyle='--', linewidth=2, label='Ajuste de Regresión')
    ax1.axhline(umbral, color='purple', linestyle=':', linewidth=2, label=f'Umbral (Calidad {umbral})')

    ax1.set_title(f'Gráfica 1: Separación Lógica por Zonas (Escala Multiplicador de Alcohol = x{mult:.1f})', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Nivel de Alcohol (Con Multiplicador)', fontsize=9)
    ax1.set_ylabel('Calidad del Vino', fontsize=9)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    canvas1.draw()

btn_aplicar = ttk.Button(frame_controles, text="Actualizar Umbral", command=actualizar_grafico1)
btn_aplicar.pack(side="left", padx=10)

actualizar_grafico1()


cols_todas = [c for c in datos_vino.columns if c != 'quality']
X1 = datos_vino[cols_todas].values
y = datos_vino['quality'].values
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y, test_size=0.2, random_state=42)

scaler1 = StandardScaler()
X1_tr_s = scaler1.fit_transform(X1_tr)
X1_te_s = scaler1.transform(X1_te)

m1 = LinearRegression().fit(X1_tr_s, y1_tr)
p1_tr = m1.predict(X1_tr_s)
p1_te = m1.predict(X1_te_s)


cols_modelo2 = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
X2 = datos_vino[cols_modelo2].values

X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X2_tr_s = scaler2.fit_transform(X2_tr)
X2_te_s = scaler2.transform(X2_te)

m2 = Ridge(alpha=10.0).fit(X2_tr_s, y2_tr)
p2_tr = m2.predict(X2_tr_s)
p2_te = m2.predict(X2_te_s)


umbral_eval = 6.5

y1_te_bin = (y1_te >= umbral_eval).astype(int)
p1_te_bin = (p1_te >= umbral_eval).astype(int)

y2_te_bin = (y2_te >= umbral_eval).astype(int)
p2_te_bin = (p2_te >= umbral_eval).astype(int)

cm2 = confusion_matrix(y2_te_bin, p2_te_bin, labels=[1, 0])
tn2, fp2, fn2, tp2 = confusion_matrix(y2_te_bin, p2_te_bin, labels=[0, 1]).ravel()

acc1 = accuracy_score(y1_te_bin, p1_te_bin)
acc2 = accuracy_score(y2_te_bin, p2_te_bin)

rec1 = recall_score(y1_te_bin, p1_te_bin, zero_division=0)
rec2 = recall_score(y2_te_bin, p2_te_bin, zero_division=0)

prec1 = precision_score(y1_te_bin, p1_te_bin, zero_division=0)
prec2 = precision_score(y2_te_bin, p2_te_bin, zero_division=0)


frame_plots_tab2 = ttk.LabelFrame(tab2, text=" Evaluación Gráfica: Diagnóstico de Alineación Lineal y Matriz de Confusión ")
frame_plots_tab2.pack(fill="both", expand=True, padx=10, pady=5)

fig2, (ax2_scatter, ax2_cm) = plt.subplots(1, 2, figsize=(10, 3.8))
canvas2 = FigureCanvasTkAgg(fig2, master=frame_plots_tab2)
canvas2.get_tk_widget().pack(expand=True, fill="both")


ax2_scatter.scatter(y2_tr, p2_tr, color='blue', alpha=0.6, label='Entrenamiento (Train)')
ax2_scatter.scatter(y2_te, p2_te, color='red', alpha=0.8, marker='^', label='Prueba (Test)')

min_val = min(y.min(), min(p2_tr.min(), p2_te.min()))
max_val = max(y.max(), max(p2_tr.max(), p2_te.max()))
ax2_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Alineación Perfecta (1:1)')

ax2_scatter.set_title("Gráfica 2: Calidad Real vs. Predicha (Alineado)", fontsize=10, fontweight='bold')
ax2_scatter.set_xlabel("Calidad Real", fontsize=8)
ax2_scatter.set_ylabel("Calidad Predicha", fontsize=8)
ax2_scatter.legend(loc='upper left', fontsize=7)
ax2_scatter.grid(True, linestyle='--', alpha=0.5)


disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Premium (>=6.5)', 'Regular (<6.5)'])
disp.plot(ax=ax2_cm, cmap='Greens', colorbar=False)
ax2_cm.set_title("Matriz de Confusión (Test - Modelo Mejorado)", fontsize=10, fontweight='bold')
ax2_cm.set_xlabel("Clase Predicha", fontsize=8)
ax2_cm.set_ylabel("Clase Real", fontsize=8)

fig2.tight_layout()
canvas2.draw()


frame_t2_middle = ttk.LabelFrame(tab2, text=" Comparación de Métricas de Validación ")
frame_t2_middle.pack(fill="x", padx=10, pady=5)

columnas = ("Métrica", "Modelo con Todas las Variables", "Modelo Mejorado (Variables Filtradas + Ridge)")
tree = ttk.Treeview(frame_t2_middle, columns=columnas, show="headings", height=8)

tree.heading("Métrica", text="Métrica de Evaluación")
tree.heading("Modelo con Todas las Variables", text="Modelo Base")
tree.heading("Modelo Mejorado (Variables Filtradas + Ridge)", text="Modelo Alineado")

tree.column("Métrica", width=340)
tree.column("Modelo con Todas las Variables", width=230, anchor="center")
tree.column("Modelo Mejorado (Variables Filtradas + Ridge)", width=230, anchor="center")

filas_metricas = [
    ("MAE Generalización (Test Error)", f"{mean_absolute_error(y1_te, p1_te):.4f}", f"{mean_absolute_error(y2_te, p2_te):.4f}"),
    ("RMSE Generalización (Test RMSE)", f"{np.sqrt(mean_squared_error(y1_te, p1_te)):.4f}", f"{np.sqrt(mean_squared_error(y2_te, p2_te)):.4f}"),
    ("R² Generalización (Test R²)", f"{r2_score(y1_te, p1_te):.4f}", f"{r2_score(y2_te, p2_te):.4f}"),
    ("Exactitud / Accuracy ((TP+TN)/(P+N))", f"{acc1 * 100:.1f}%", f"{acc2 * 100:.1f}%"),
    ("Sensibilidad / Recall (TP/P)", f"{rec1 * 100:.1f}%", f"{rec2 * 100:.1f}%"),
    ("Precisión (TP/(TP+FP))", f"{prec1 * 100:.1f}%", f"{prec2 * 100:.1f}%"),
    ("Matriz de Confusión (TP / FP / FN / TN)", "-", f"TP:{tp2} | FP:{fp2} | FN:{fn2} | TN:{tn2}"),
    ("Diagnóstico de Linealidad", "Ligero desvío por ruido químico", "Puntos ajustados y mayor estabilidad")
]

for fila in filas_metricas:
    tree.insert("", "end", values=fila)

tree.pack(fill="x", padx=5, pady=5)


frame_info = ttk.LabelFrame(tab2, text=" Explicación del Ajuste y Selección de Variables ")
frame_info.pack(fill="x", padx=10, pady=5)

txt_info = tk.Text(frame_info, wrap="word", font=("Arial", 9), height=5)
txt_info.pack(fill="both", expand=True, padx=5, pady=5)

explicacion = """• Variables Conservadas: 'alcohol', 'volatile acidity', 'sulphates' y 'citric acid'. Estas variables mostraron la mayor correlación matemática, por lo que se mantuvieron para capturar la base química que define la calidad del vino.
• Variables Descartadas: Se omitieron componentes de baja correlación (como 'residual sugar', 'pH' o 'free sulfur dioxide'), ya que introducen ruido al modelo lineal sin aportar precisión a la predicción final.
• Alineación 1:1 y Ridge: Al filtrar las variables ruidosas y aplicar regularización Ridge, el modelo estabiliza los coeficientes, mejorando la generalización (Test Error) y alineando mejor las predicciones a la diagonal real."""

txt_info.insert("1.0", explicacion)
txt_info.config(state="disabled")


root.mainloop()
