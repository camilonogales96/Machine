import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def calcular_metricas(modelo, X, y):
    """Devuelve un diccionario con todas las métricas de un modelo sobre (X, y)."""
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        'Exactitud':            accuracy_score(y, y_pred),
        'Exactitud balanceada': balanced_accuracy_score(y, y_pred),
        'Precisión':            precision_score(y, y_pred, zero_division=0),
        'Recall (sensib.)':     recall_score(y, y_pred, zero_division=0),
        'Especificidad':        tn / (tn + fp) if (tn + fp) else 0.0,
        'F1':                   f1_score(y, y_pred, zero_division=0),
        'ROC-AUC':              roc_auc_score(y, y_prob),
        'PR-AUC':               average_precision_score(y, y_prob),
        'MCC':                  matthews_corrcoef(y, y_pred),
        'Kappa':                cohen_kappa_score(y, y_pred),
    }


def cargar_datos_hr_arbol():
    """Busca y carga de manera robusta el dataset HRDataset_v14.csv."""
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    curr = directorio_actual
    ruta_origen = None
    
    while curr:
        candidato = os.path.join(curr, 'HRDataset_v14.csv')
        if os.path.exists(candidato):
            ruta_origen = candidato
            break
        cand_sub = os.path.join(curr, 'Dataset', 'human_resources', 'HRDataset_v14.csv')
        if os.path.exists(cand_sub):
            ruta_origen = cand_sub
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    if not ruta_origen:
        cand_cwd = os.path.join(os.getcwd(), 'HRDataset_v14.csv')
        if os.path.exists(cand_cwd):
            ruta_origen = cand_cwd
        else:
            ruta_origen = os.path.join(directorio_actual, 'HRDataset_v14.csv')

    if not os.path.exists(ruta_origen):
        return None, None

    datos_hr = pd.read_csv(ruta_origen).drop_duplicates().dropna(subset=['Termd'])
    return datos_hr, ruta_origen


def decorar_ramas_arbol(ax, modelo, feature_names, class_names, fontsize=9):
    """
    Dibuja el árbol y etiqueta individualmente cada bifurcación con 'Sí (cumple)'
    en verde a la izquierda y 'No (no cumple)' en rojo a la derecha.
    """
    anotaciones = plot_tree(
        modelo,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=fontsize,
        proportion=False,
        impurity=True,
        precision=2,
        ax=ax
    )

    arbol_interno = modelo.tree_
    if arbol_interno.node_count <= 1:
        return anotaciones

    COLOR_SI = '#2e7d32'   # verde: la condición del nodo se cumple (rama izquierda)
    COLOR_NO = '#c62828'   # rojo: la condición del nodo NO se cumple (rama derecha)

    anotaciones_nodos = [a for a in anotaciones if a.get_text().strip() not in ('True', 'False')]
    if len(anotaciones_nodos) != arbol_interno.node_count:
        return anotaciones

    for anotacion in anotaciones:
        if anotacion.get_text().strip() in ('True', 'False'):
            anotacion.set_visible(False)

    for id_padre in range(arbol_interno.node_count):
        id_izquierdo = arbol_interno.children_left[id_padre]
        id_derecho = arbol_interno.children_right[id_padre]

        if id_izquierdo == -1:
            continue  # nodo hoja: sin bifurcaciones

        flecha_izquierda = anotaciones_nodos[id_izquierdo].arrow_patch
        if flecha_izquierda is not None:
            flecha_izquierda.set_color(COLOR_SI)
            flecha_izquierda.set_linewidth(2.2)

        flecha_derecha = anotaciones_nodos[id_derecho].arrow_patch
        if flecha_derecha is not None:
            flecha_derecha.set_color(COLOR_NO)
            flecha_derecha.set_linewidth(2.2)

        x_padre, y_padre = anotaciones_nodos[id_padre].get_position()
        x_izq, y_izq = anotaciones_nodos[id_izquierdo].get_position()
        x_der, y_der = anotaciones_nodos[id_derecho].get_position()

        ax.text((x_padre + x_izq) / 2, (y_padre + y_izq) / 2, 'Sí\n(cumple)',
                color=COLOR_SI, fontsize=8.5, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=COLOR_SI, alpha=0.9),
                zorder=10)

        ax.text((x_padre + x_der) / 2, (y_padre + y_der) / 2, 'No\n(no cumple)',
                color=COLOR_NO, fontsize=8.5, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=COLOR_NO, alpha=0.9),
                zorder=10)

    return anotaciones


def crear_interfaz_hr_arbol(parent_widget):
    datos_hr, ruta_origen = cargar_datos_hr_arbol()
    if datos_hr is None:
        lbl = ttk.Label(parent_widget, text="No se encontró HRDataset_v14.csv", font=('Segoe UI', 12, 'bold'), foreground='red')
        lbl.pack(pady=20)
        return

    # 1. Preprocesamiento y selección de variables
    variables_numericas = datos_hr.select_dtypes(include=['int64', 'float64']).columns
    datos_hr_numerico = datos_hr[variables_numericas].fillna(0)

    objetivo = datos_hr_numerico['Termd']

    # Configuración de exclusión para evitar sobreajuste y fuga de datos
    EXCLUIR_POSIBLE_FUGA = True
    EXCLUIR_ZIP = True
    PESO_CLASES = 'balanced'

    columnas_a_descartar = ['Termd', 'EmpID']
    if EXCLUIR_POSIBLE_FUGA:
        columnas_a_descartar.append('EmpStatusID')
    if EXCLUIR_ZIP:
        columnas_a_descartar.append('Zip')
    columnas_a_descartar = [c for c in columnas_a_descartar if c in datos_hr_numerico.columns]

    variables_entrada = datos_hr_numerico.drop(columns=columnas_a_descartar)

    # División estratificada entrenamiento / prueba
    X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
        variables_entrada, objetivo, test_size=0.2, random_state=77, stratify=objetivo
    )

    print("=== ENTRENAMIENTO DEL MODELO ÁRBOL DE DECISIÓN (RRHH) ===")
    print(f"Ruta del archivo cargado: {ruta_origen}")
    print(f"Volumen de datos de entrenamiento: {len(X_entrenar)}")
    print(f"Volumen de datos de prueba: {len(X_probar)}")
    print(f"Columnas descartadas como predictoras: {columnas_a_descartar}")

    # 2. Creación y entrenamiento del modelo Árbol de Decisión Base
    modelo_arbol = DecisionTreeClassifier(class_weight=PESO_CLASES, criterion='gini', max_depth=4, random_state=77)
    modelo_arbol.fit(X_entrenar, y_entrenar)

    predicciones = modelo_arbol.predict(X_probar)
    exactitud = accuracy_score(y_probar, predicciones)
    matriz_confusion = confusion_matrix(y_probar, predicciones)
    reporte_clasificacion = classification_report(y_probar, predicciones, zero_division=0)

    print("\n=== REPORTE DE EVALUACIÓN - ÁRBOL DE DECISIÓN ===")
    print(f"Exactitud (Accuracy): {exactitud * 100:.2f}%")
    print("\nMatriz de Confusión:")
    print(matriz_confusion)
    print("\nReporte de Clasificación:")
    print(reporte_clasificacion)

    # Importancia de las variables
    importancias = pd.DataFrame({
        'Variable': variables_entrada.columns,
        'Importancia': modelo_arbol.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    print("\n=== IMPORTANCIA DE LAS VARIABLES ===")
    print(importancias[importancias['Importancia'] > 0].to_string(index=False))

    # 3. Poda del Árbol (Cost-Complexity Pruning) con Validación Cruzada
    modelo_completo = DecisionTreeClassifier(class_weight=PESO_CLASES, criterion='gini', random_state=77)
    modelo_completo.fit(X_entrenar, y_entrenar)
    ruta_poda = modelo_completo.cost_complexity_pruning_path(X_entrenar, y_entrenar)
    ccp_alphas = ruta_poda.ccp_alphas

    cv_poda = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    exactitudes_train, exactitudes_test, num_nodos, puntajes_cv = [], [], [], []
    for alpha in ccp_alphas:
        arbol_alpha = DecisionTreeClassifier(class_weight=PESO_CLASES, criterion='gini', random_state=77, ccp_alpha=alpha)
        arbol_alpha.fit(X_entrenar, y_entrenar)
        exactitudes_train.append(arbol_alpha.score(X_entrenar, y_entrenar))
        exactitudes_test.append(arbol_alpha.score(X_probar, y_probar))
        num_nodos.append(arbol_alpha.tree_.node_count)
        puntajes_cv.append(cross_val_score(arbol_alpha, X_entrenar, y_entrenar,
                                           cv=cv_poda, scoring='balanced_accuracy').mean())

    mejor_cv = max(puntajes_cv)
    candidatos_alpha = [(a, n, p) for a, n, p in zip(ccp_alphas, num_nodos, puntajes_cv)
                        if p >= mejor_cv - 0.005 and n > 1]
    if not candidatos_alpha:
        candidatos_alpha = [(a, n, p) for a, n, p in zip(ccp_alphas, num_nodos, puntajes_cv) if p == mejor_cv]
    alpha_recomendado, nodos_recomendado, exactitud_recomendada = max(candidatos_alpha, key=lambda t: t[0])

    modelo_arbol_podado = DecisionTreeClassifier(class_weight=PESO_CLASES, criterion='gini', random_state=77, ccp_alpha=alpha_recomendado)
    modelo_arbol_podado.fit(X_entrenar, y_entrenar)
    exactitud_podado = modelo_arbol_podado.score(X_probar, y_probar)
    nodos_original = modelo_arbol.tree_.node_count
    nodos_podado = modelo_arbol_podado.tree_.node_count

    print("\n=== REPORTE COMPARATIVO DE PODA DEL ÁRBOL (CCP) ===")
    print(f"Árbol Base (max_depth=4)      : Nodos = {nodos_original}  | Exactitud = {exactitud * 100:.2f}%")
    print(f"Árbol Podado (ccp_alpha={alpha_recomendado:.4f}): Nodos = {nodos_podado}  | Exactitud = {exactitud_podado * 100:.2f}%")

    # Tabla de Métricas Completas (Base vs Podado)
    tabla_metricas = pd.DataFrame({
        'Base - Train':   calcular_metricas(modelo_arbol, X_entrenar, y_entrenar),
        'Base - Test':    calcular_metricas(modelo_arbol, X_probar, y_probar),
        'Podado - Train': calcular_metricas(modelo_arbol_podado, X_entrenar, y_entrenar),
        'Podado - Test':  calcular_metricas(modelo_arbol_podado, X_probar, y_probar),
    })

    print("\n=== MÉTRICAS COMPLETAS (BASE vs PODADO) ===")
    print(tabla_metricas.round(4).to_string())

    print("\n=== REPORTE DE CLASIFICACIÓN - ÁRBOL PODADO (TEST) ===")
    print(classification_report(y_probar, modelo_arbol_podado.predict(X_probar),
                                target_names=['Activo', 'Terminado'], zero_division=0))

    # Validación cruzada estratificada (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    scoring = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    print("\n=== VALIDACIÓN CRUZADA (5-fold estratificada, media ± desv.) ===")
    for nombre, modelo_cv in [
        ('Base (max_depth=4)', DecisionTreeClassifier(class_weight=PESO_CLASES, criterion='gini', max_depth=4, random_state=77)),
        (f'Podado (ccp_alpha={alpha_recomendado:.4f})',
         DecisionTreeClassifier(class_weight=PESO_CLASES, criterion='gini', random_state=77, ccp_alpha=alpha_recomendado)),
    ]:
        res = cross_validate(modelo_cv, variables_entrada, objetivo, cv=cv, scoring=scoring)
        print(f"\n{nombre}")
        for m in scoring:
            vals = res[f'test_{m}']
            print(f"  {m:<18}: {vals.mean():.4f} ± {vals.std():.4f}")

    # 4. Exportación de archivos CSV
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    datos_ajustados = datos_hr.copy()
    datos_ajustados['Prediccion_Termd_Arbol'] = modelo_arbol_podado.predict(variables_entrada)
    datos_ajustados['Estado_Predicho_Arbol'] = datos_ajustados['Prediccion_Termd_Arbol'].map({0: 'Activo', 1: 'Terminado'})

    ruta_exportacion_arbol = os.path.join(directorio_actual, 'HRDataset_Ajustado_Arbol.csv')
    datos_ajustados.to_csv(ruta_exportacion_arbol, index=False)

    ruta_metricas = os.path.join(directorio_actual, 'Metricas_Arbol.csv')
    tabla_metricas.round(4).to_csv(ruta_metricas)

    # Guardar copia en el directorio raíz del proyecto si existe
    directorio_raiz = os.path.abspath(os.path.join(directorio_actual, '..', '..', '..'))
    if os.path.exists(os.path.join(directorio_raiz, 'main.py')):
        try:
            datos_ajustados.to_csv(os.path.join(directorio_raiz, 'HRDataset_Ajustado_Arbol.csv'), index=False)
            tabla_metricas.round(4).to_csv(os.path.join(directorio_raiz, 'Metricas_Arbol.csv'))
        except Exception:
            pass

    print("\n=== GENERACIÓN DE NUEVO ARCHIVO CSV ===")
    print(f"[ÉXITO] Se ha generado el nuevo archivo CSV ajustado con las predicciones del modelo Árbol:")
    print(f"📁 Ruta: {ruta_exportacion_arbol}")
    print(f"📁 Tabla de métricas: {ruta_metricas}")

    # =========================================================================
    # 5. Generación de Figuras de Visualización
    # =========================================================================

    # FIGURA 1: Matriz de Confusión + Frontera de Decisión (PCA 2D)
    fig1 = mfigure.Figure(figsize=(13, 6))
    FigureCanvasTkAgg(fig1)
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)

    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Activo (0)', 'Terminado (1)'],
                yticklabels=['Activo (0)', 'Terminado (1)'])
    ax1.set_title('Matriz de Confusión - Árbol de Decisión', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Valor Real')

    scaler = StandardScaler()
    X_entrenar_escalado = scaler.fit_transform(X_entrenar)
    pca = PCA(n_components=2)
    X_entrenar_pca = pca.fit_transform(X_entrenar_escalado)

    modelo_arbol_2d = DecisionTreeClassifier(class_weight=PESO_CLASES, criterion='gini', max_depth=4, random_state=77)
    modelo_arbol_2d.fit(X_entrenar_pca, y_entrenar)

    x_min, x_max = X_entrenar_pca[:, 0].min() - 1, X_entrenar_pca[:, 0].max() + 1
    y_min, y_max = X_entrenar_pca[:, 1].min() - 1, X_entrenar_pca[:, 1].max() + 1
    step_x = (x_max - x_min) / 200
    step_y = (y_max - y_min) / 200

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x), np.arange(y_min, y_max, step_y))
    Z = modelo_arbol_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax2.scatter(X_entrenar_pca[:, 0], X_entrenar_pca[:, 1], c=y_entrenar, cmap=plt.cm.coolwarm, edgecolors='k')
    ax2.set_title('Frontera de Decisión (Árbol 2D vía PCA)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Componente Principal 1')
    ax2.set_ylabel('Componente Principal 2')
    legend1 = ax2.legend(*scatter.legend_elements(), title="Estado (Termd)")
    ax2.add_artist(legend1)
    fig1.tight_layout()

    # FIGURA 2: Estructura del Árbol de Decisión Completo
    fig2 = mfigure.Figure(figsize=(18, 9))
    FigureCanvasTkAgg(fig2)
    ax3 = fig2.add_subplot(111)
    decorar_ramas_arbol(ax3, modelo_arbol, list(variables_entrada.columns), ['Activo', 'Terminado'], fontsize=9)
    ax3.set_title('Estructura Visual del Árbol de Decisión (Reglas de División)', fontsize=13, fontweight='bold')
    fig2.text(0.5, 0.01,
              'Cómo leer cada nodo: si la condición mostrada (ej. "Variable <= valor") se CUMPLE, se avanza '
              'por la rama verde "Sí"; si NO se cumple, se avanza por la rama roja "No".',
              ha='center', va='bottom', fontsize=10, style='italic', color='#333333')
    fig2.tight_layout(pad=2.0, rect=[0, 0.04, 1, 1])

    # FIGURA 3: Gráfico de Dispersión
    fig3 = mfigure.Figure(figsize=(9, 6))
    FigureCanvasTkAgg(fig3)
    ax4 = fig3.add_subplot(111)
    sns.regplot(data=datos_hr_numerico, x='PerfScoreID', y='EngagementSurvey', x_jitter=0.15,
                scatter_kws=dict(alpha=0.6, s=45, color='#2a6ebb', edgecolor='white'),
                line_kws=dict(color='#c0392b', linewidth=2.5), ax=ax4)
    ax4.set_xlabel('PerfScoreID (1=PIP, 2=Necesita mejorar, 3=Cumple, 4=Excede)', fontsize=10)
    ax4.set_ylabel('EngagementSurvey (encuesta de compromiso, 1 a 5)', fontsize=10)
    ax4.set_title('Relación entre Desempeño y Compromiso del Empleado', fontsize=12, fontweight='bold')
    correlacion_perf_engagement = datos_hr_numerico['PerfScoreID'].corr(datos_hr_numerico['EngagementSurvey'])
    fig3.text(0.5, 0.01,
              f'Cada punto es un empleado. La línea muestra la tendencia general: a mejor evaluación de desempeño, '
              f'mayor compromiso reportado (correlación = {correlacion_perf_engagement:.2f}).',
              ha='center', va='bottom', fontsize=9.5, style='italic', color='#333333')
    fig3.tight_layout(rect=[0, 0.05, 1, 1])

    # FIGURA 4: Poda del Árbol (CCP)
    fig4 = mfigure.Figure(figsize=(13, 6))
    FigureCanvasTkAgg(fig4)
    axA = fig4.add_subplot(1, 2, 1)
    axB = fig4.add_subplot(1, 2, 2)

    axA.plot(ccp_alphas, exactitudes_train, marker='o', markersize=6, label='Entrenamiento', color='#2a6ebb')
    axA.plot(ccp_alphas, exactitudes_test, marker='o', markersize=6, label='Prueba', color='#c0392b')
    axA.plot(ccp_alphas, puntajes_cv, marker='s', markersize=5, label='Exact. balanceada CV (train)', color='#e08e0b')
    axA.axvline(alpha_recomendado, color='#2e7d32', linestyle='--', linewidth=1.8, label=f'α sugerido ({alpha_recomendado:.3f})')
    axA.set_xlabel('ccp_alpha (nivel de poda)')
    axA.set_ylabel('Exactitud')
    axA.set_title('Exactitud vs. Nivel de Poda')
    axA.legend()

    axB.plot(ccp_alphas, num_nodos, marker='o', markersize=6, color='#6a3d9a')
    axB.axvline(alpha_recomendado, color='#2e7d32', linestyle='--', linewidth=1.8, label=f'Alpha sugerido ({nodos_recomendado} nodos)')
    axB.set_xlabel('ccp_alpha (nivel de poda)')
    axB.set_ylabel('Cantidad de nodos del árbol')
    axB.set_title('Complejidad del Árbol vs. Nivel de Poda')
    axB.legend()

    fig4.suptitle(f'Poda del Árbol (Cost-Complexity Pruning)\nÁrbol Base: {nodos_original} nodos ({exactitud*100:.1f}%) | Árbol Podado: {nodos_podado} nodos ({exactitud_podado*100:.1f}%)', fontsize=11, fontweight='bold')
    fig4.text(0.5, 0.01,
              'La poda elimina ramas redundantes. La línea verde marca el árbol más '
              'simple con la mejor exactitud balanceada en validación cruzada (sobre entrenamiento).',
              ha='center', va='bottom', fontsize=9.5, style='italic', color='#333333')
    fig4.tight_layout(rect=[0, 0.05, 1, 0.93])

    # FIGURA 5: Visualización de la Estructura del Árbol Podado
    fig5 = mfigure.Figure(figsize=(18, 9))
    FigureCanvasTkAgg(fig5)
    ax5 = fig5.add_subplot(111)
    decorar_ramas_arbol(ax5, modelo_arbol_podado, list(variables_entrada.columns), ['Activo', 'Terminado'], fontsize=9)
    ax5.set_title(f'Estructura del Árbol Podado Optimizado (ccp_alpha = {alpha_recomendado:.4f}, Nodos = {nodos_podado})', fontsize=12, fontweight='bold')
    fig5.tight_layout()

    # FIGURA 6: Curvas ROC / Precision-Recall y Métricas
    fig6 = mfigure.Figure(figsize=(15, 5.5))
    FigureCanvasTkAgg(fig6)
    axR = fig6.add_subplot(1, 3, 1)
    axP = fig6.add_subplot(1, 3, 2)
    axM = fig6.add_subplot(1, 3, 3)

    for modelo, nombre, color in [(modelo_arbol, 'Base', '#2a6ebb'),
                                  (modelo_arbol_podado, 'Podado', '#c0392b')]:
        prob = modelo.predict_proba(X_probar)[:, 1]
        fpr, tpr, _ = roc_curve(y_probar, prob)
        prec, rec, _ = precision_recall_curve(y_probar, prob)
        axR.plot(fpr, tpr, color=color, lw=2,
                 label=f'{nombre} (AUC = {roc_auc_score(y_probar, prob):.2f})')
        axP.plot(rec, prec, color=color, lw=2,
                 label=f'{nombre} (AP = {average_precision_score(y_probar, prob):.2f})')

    axR.plot([0, 1], [0, 1], 'k--', label='Azar')
    axR.set_xlabel('Tasa de falsos positivos')
    axR.set_ylabel('Tasa de verdaderos positivos')
    axR.set_title('Curva ROC (Test)', fontweight='bold')
    axR.legend(loc='lower right')

    axP.axhline(y_probar.mean(), color='k', linestyle='--', label='Azar (prevalencia)')
    axP.set_xlabel('Recall')
    axP.set_ylabel('Precisión')
    axP.set_title('Curva Precision-Recall (Test)', fontweight='bold')
    axP.legend(loc='lower left')

    metricas_barras = ['Exactitud', 'Precisión', 'Recall (sensib.)', 'Especificidad', 'F1', 'ROC-AUC']
    tabla_metricas.loc[metricas_barras, ['Base - Test', 'Podado - Test']].plot(
        kind='bar', ax=axM, color=['#2a6ebb', '#c0392b'], edgecolor='k')
    axM.set_ylim(0, 1.05)
    axM.set_title('Métricas en Test: Base vs Podado', fontweight='bold')
    axM.set_xticklabels(metricas_barras, rotation=35, ha='right')
    axM.grid(axis='y', alpha=0.3)

    fig6.tight_layout()

    # =========================================================================
    # 6. Montaje en Pestañas (Notebook)
    # =========================================================================
    notebook = ttk.Notebook(parent_widget)
    notebook.pack(fill='both', expand=True)

    figuras_con_titulos = [
        (fig1, "Resumen (Matriz + Frontera)"),
        (fig2, "Árbol completo"),
        (fig3, "Gráfico de Dispersión"),
        (fig4, "Poda del Árbol"),
        (fig5, "Árbol Podado Optimizado"),
        (fig6, "Métricas (ROC / PR)"),
    ]

    for figura, titulo in figuras_con_titulos:
        pestana = ttk.Frame(notebook)
        notebook.add(pestana, text=titulo)

        canvas = FigureCanvasTkAgg(figura, master=pestana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, pestana)
        toolbar.update()


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Resultados - Árbol de Decisión RRHH")
    root.geometry("1300x850")
    crear_interfaz_hr_arbol(root)
    root.mainloop()