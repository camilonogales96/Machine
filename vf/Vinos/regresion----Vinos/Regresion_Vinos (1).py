import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def cargar_datos_vinos_regresion():
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
        return None

    df = pd.read_csv(ruta_origen).drop_duplicates()
    return df

def crear_interfaz_vinos_regresion(parent_widget):
    datos_vino = cargar_datos_vinos_regresion()
    if datos_vino is None:
        lbl = ttk.Label(parent_widget, text="No se encontró winequality-red.csv", font=('Segoe UI', 12, 'bold'), foreground='red')
        lbl.pack(pady=20)
        return

    notebook = ttk.Notebook(parent_widget)
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)

    notebook.add(tab1, text=" 🎨 Pestaña 1: Gráfica de Zonas y Multiplicador ")
    notebook.add(tab2, text=" 📋 Pestaña 2: Gráfica de Validación y Métricas (Diapositiva 1.5) ")
    notebook.pack(expand=1, fill="both")

    # ==========================================
    # PESTAÑA 1: GRÁFICA 1 (DISPERSIÓN Y ZONAS)
    # ==========================================
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

    fig1 = mfigure.Figure(figsize=(9, 4.5))
    ax1 = fig1.add_subplot(111)
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

        mask_azul = y_real >= umbral
        mask_rojo = y_real < umbral

        x_min, x_max = x_rango.min(), x_rango.max()
        y_min, y_max = 2.5, 8.5

        ax1.fill_between([x_min, x_max], umbral, y_max, color='blue', alpha=0.12, label='Zona Azul (Vino Excelente >= Umbral)')
        ax1.fill_between([x_min, x_max], y_min, umbral, color='red', alpha=0.12, label='Zona Roja (Vino Estándar/Bajo < Umbral)')

        ax1.scatter(X_unaria[mask_azul], y_real[mask_azul], color='blue', edgecolors='black', s=45, alpha=0.8, label='Puntos Azules')
        ax1.scatter(X_unaria[mask_rojo], y_real[mask_rojo], color='red', edgecolors='black', s=45, alpha=0.8, label='Puntos Rojos')

        ax1.plot(x_rango, y_pred_rango, color='black', linestyle='--', linewidth=2, label='Ajuste de Regresión')
        ax1.axhline(umbral, color='purple', linestyle=':', linewidth=2, label=f'Umbral ({umbral:.1f})')

        ax1.set_title(f'Gráfica 1: Separación por Zonas - Calidad de Vino (Escala Multiplicador = x{mult:.1f})', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Alcohol (Con Multiplicador)', fontsize=9)
        ax1.set_ylabel('Calidad del Vino', fontsize=9)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, linestyle='--', alpha=0.5)

        canvas1.draw()

    btn_aplicar = ttk.Button(frame_controles, text="Actualizar Umbral", command=actualizar_grafico1)
    btn_aplicar.pack(side="left", padx=10)

    actualizar_grafico1()

    # ==========================================
    # PESTAÑA 2: GRÁFICA 2 Y VALIDACIÓN (DIAPOSITIVA 1.5)
    # ==========================================
    cols_todas = [c for c in datos_vino.columns if c != 'quality']
    cols_filtradas = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']

    # Modelo 1: Todas las variables
    X1 = datos_vino[cols_todas].values
    y = datos_vino['quality'].values
    X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y, test_size=0.2, random_state=42)

    scaler1 = StandardScaler()
    X1_tr_s = scaler1.fit_transform(X1_tr)
    X1_te_s = scaler1.transform(X1_te)

    m1 = LinearRegression().fit(X1_tr_s, y1_tr)
    p1_tr = m1.predict(X1_tr_s)
    p1_te = m1.predict(X1_te_s)

    # Modelo 2: Variables seleccionadas (Modelo Mejorado)
    X2 = datos_vino[cols_filtradas].values
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y, test_size=0.2, random_state=42)

    scaler2 = StandardScaler()
    X2_tr_s = scaler2.fit_transform(X2_tr)
    X2_te_s = scaler2.transform(X2_te)

    m2 = Ridge(alpha=1.0).fit(X2_tr_s, y2_tr)
    p2_tr = m2.predict(X2_tr_s)
    p2_te = m2.predict(X2_te_s)

    # Contenedor Superior: Gráfica 2
    frame_plot2 = ttk.LabelFrame(tab2, text=" Gráfica 2: Evaluador de Ajuste (Valores Reales vs. Predicciones del Modelo) ")
    frame_plot2.pack(fill="both", expand=True, padx=10, pady=5)

    fig2 = mfigure.Figure(figsize=(9, 3.8))
    ax2 = fig2.add_subplot(111)
    canvas2 = FigureCanvasTkAgg(fig2, master=frame_plot2)
    canvas2.get_tk_widget().pack(expand=True, fill="both")

    ax2.scatter(y2_tr, p2_tr, color='blue', alpha=0.6, label='Entrenamiento (Train)')
    ax2.scatter(y2_te, p2_te, color='red', alpha=0.8, marker='^', label='Prueba / Generalización (Test)')

    min_val = min(y.min(), min(p2_tr.min(), p2_te.min()))
    max_val = max(y.max(), max(p2_tr.max(), p2_te.max()))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Predicción Perfecta (1:1)')

    ax2.set_title("Gráfica 2: Diagnóstico de Ajuste - Calidad Real vs. Calidad Predicha", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Calidad Real del Vino", fontsize=9)
    ax2.set_ylabel("Calidad Predicha por el Modelo", fontsize=9)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    canvas2.draw()

    # Contenedor Inferior: Tabla de Métricas
    frame_t2_middle = ttk.LabelFrame(tab2, text=" Comparación de Métricas de Validación ")
    frame_t2_middle.pack(fill="x", padx=10, pady=5)

    columnas = ("Métrica", "Modelo Original (Todas las vars)", "Modelo Mejorado (Vars Seleccionadas)")
    tree = ttk.Treeview(frame_t2_middle, columns=columnas, show="headings", height=5)

    tree.heading("Métrica", text="Métrica de Validación (Diapositiva 1.5)")
    tree.heading("Modelo Original (Todas las vars)", text="Modelo Original")
    tree.heading("Modelo Mejorado (Vars Seleccionadas)", text="Modelo Mejorado")

    tree.column("Métrica", width=340)
    tree.column("Modelo Original (Todas las vars)", width=230, anchor="center")
    tree.column("Modelo Mejorado (Vars Seleccionadas)", width=230, anchor="center")

    filas_metricas = [
        ("MAE Entrenamiento (Train Error)", f"{mean_absolute_error(y1_tr, p1_tr):.4f}", f"{mean_absolute_error(y2_tr, p2_tr):.4f}"),
        ("MAE Generalización (Test Error)", f"{mean_absolute_error(y1_te, p1_te):.4f}", f"{mean_absolute_error(y2_te, p2_te):.4f}"),
        ("RMSE Generalización (Test RMSE)", f"{np.sqrt(mean_squared_error(y1_te, p1_te)):.4f}", f"{np.sqrt(mean_squared_error(y2_te, p2_te)):.4f}"),
        ("R² Generalización (Test R²)", f"{r2_score(y1_te, p1_te):.4f}", f"{r2_score(y2_te, p2_te):.4f}"),
        ("Diagnóstico de Desempeño", "Ruido por variables poco útiles", "Mayor estabilidad frente a generalización")
    ]

    for fila in filas_metricas:
        tree.insert("", "end", values=fila)

    tree.pack(fill="x", padx=5, pady=5)

    # Cuadro explicativo
    frame_info = ttk.LabelFrame(tab2, text=" Análisis Teórico según Diapositiva 1.5 ")
    frame_info.pack(fill="x", padx=10, pady=5)

    txt_info = tk.Text(frame_info, wrap="word", font=("Arial", 9), height=4)
    txt_info.pack(fill="both", expand=True, padx=5, pady=5)

    explicacion = """• Generalización y Robustez: El 'Test Error' (Error de Generalización) nos indica qué tan bien rinde el modelo ante muestras nuevas de vinos.
• Subajuste vs Sobreajuste: Al remover las variables químicas con baja correlación y conservar las determinantes ('alcohol', 'volatile acidity', 'sulphates'), reducimos el ruido. Esto previene el sobreajuste (donde el modelo memoriza el ruido del entrenamiento) y mejora la capacidad de ajuste a datos invisibles.
• Error Irreducible: El modelo no es 100% perfecto porque existe variación (Bias) natural en la degustación de vinos."""

    txt_info.insert("1.0", explicacion)
    txt_info.config(state="disabled")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Regresión Vinos")
    root.geometry("1150x850")
    crear_interfaz_vinos_regresion(root)
    root.mainloop()