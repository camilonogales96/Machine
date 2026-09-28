import os

def process_file(filename, is_lineal):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    gui_start = content.find("root = tk.Tk()")
    if gui_start == -1:
        print("GUI start not found in", filename)
        return

    before_gui = content[:gui_start]
    after_gui = content[gui_start:]

    if is_lineal:
        calc_logic = """
# ==========================================
# CALCULO MODELO MEJORADO
# ==========================================
num_eliminar = 3
variables_a_eliminar = coefs.sort_values(ascending=True).head(num_eliminar).index.tolist()
variables_conservadas = [c for c in variables_entrada.columns if c not in variables_a_eliminar]

X_entrenar_mejorado = X_entrenar[variables_conservadas]
X_probar_mejorado = X_probar[variables_conservadas]

modelo_svm_mejorado = SVC(kernel='linear', C=1.0, random_state=77)
modelo_svm_mejorado.fit(X_entrenar_mejorado, y_entrenar)

pred_mej_test = modelo_svm_mejorado.predict(X_probar_mejorado)
pred_mej_train = modelo_svm_mejorado.predict(X_entrenar_mejorado)
pred_orig_train = modelo_svm_lineal.predict(X_entrenar)

matriz_confusion_mejorado = confusion_matrix(y_probar, pred_mej_test)

def calcular_metricas(y_true, y_pred, cm):
    acc = accuracy_score(y_true, y_pred)
    err = 1.0 - acc
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    prec_macro = report['macro avg']['precision']
    rec_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']
    
    spec_list = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_list.append(spec)
    spec_macro = np.mean(spec_list)
    return acc, err, prec_macro, rec_macro, spec_macro, f1_macro

acc_orig, err_orig, prec_orig, rec_orig, spec_orig, f1_orig = calcular_metricas(y_probar, predicciones, matriz_confusion)
acc_mej, err_mej, prec_mej, rec_mej, spec_mej, f1_mej = calcular_metricas(y_probar, pred_mej_test, matriz_confusion_mejorado)

acc_train_orig = accuracy_score(y_entrenar, pred_orig_train)
acc_train_mej = accuracy_score(y_entrenar, pred_mej_train)

pca_mej = PCA(n_components=2)
X_probar_pca_mej = pca_mej.fit_transform(X_probar_mejorado)
varianza_mej = pca_mej.explained_variance_ratio_
pca2_mej = PCA(n_components=2)
X_entrenar_pca_mej = pca2_mej.fit_transform(X_entrenar_mejorado)
modelo_svm_2d_mej = SVC(kernel='linear', C=1.0, random_state=77)
modelo_svm_2d_mej.fit(X_entrenar_pca_mej, y_entrenar)

"""
    else:
        calc_logic = """
# ==========================================
# CALCULO MODELO MEJORADO
# ==========================================
num_eliminar = 3
variables_a_eliminar = importancias.sort_values(ascending=True).head(num_eliminar).index.tolist()
variables_conservadas = [c for c in variables_entrada.columns if c not in variables_a_eliminar]

X_entrenar_mejorado = X_entrenar[variables_conservadas]
X_probar_mejorado = X_probar[variables_conservadas]

modelo_svm_mejorado = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_svm_mejorado.fit(X_entrenar_mejorado, y_entrenar)

pred_mej_test = modelo_svm_mejorado.predict(X_probar_mejorado)
pred_mej_train = modelo_svm_mejorado.predict(X_entrenar_mejorado)
pred_orig_train = modelo_svm_no_lineal.predict(X_entrenar)

matriz_confusion_mejorado = confusion_matrix(y_probar, pred_mej_test)

def calcular_metricas(y_true, y_pred, cm):
    acc = accuracy_score(y_true, y_pred)
    err = 1.0 - acc
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    prec_macro = report['macro avg']['precision']
    rec_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']
    
    spec_list = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_list.append(spec)
    spec_macro = np.mean(spec_list)
    return acc, err, prec_macro, rec_macro, spec_macro, f1_macro

acc_orig, err_orig, prec_orig, rec_orig, spec_orig, f1_orig = calcular_metricas(y_probar, predicciones, matriz_confusion)
acc_mej, err_mej, prec_mej, rec_mej, spec_mej, f1_mej = calcular_metricas(y_probar, pred_mej_test, matriz_confusion_mejorado)

acc_train_orig = accuracy_score(y_entrenar, pred_orig_train)
acc_train_mej = accuracy_score(y_entrenar, pred_mej_train)

pca_mej = PCA(n_components=2)
X_probar_pca_mej = pca_mej.fit_transform(X_probar_mejorado)
varianza_mej = pca_mej.explained_variance_ratio_
pca2_mej = PCA(n_components=2)
X_entrenar_pca_mej = pca2_mej.fit_transform(X_entrenar_mejorado)
modelo_svm_2d_mej = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=77)
modelo_svm_2d_mej.fit(X_entrenar_pca_mej, y_entrenar)

x_min2d_mej = X_entrenar_pca_mej[:, 0].min() - margen
x_max2d_mej = X_entrenar_pca_mej[:, 0].max() + margen
y_min2d_mej = X_entrenar_pca_mej[:, 1].min() - margen
y_max2d_mej = X_entrenar_pca_mej[:, 1].max() + margen
xx_mej, yy_mej = np.meshgrid(np.arange(x_min2d_mej, x_max2d_mej, 0.12),
                             np.arange(y_min2d_mej, y_max2d_mej, 0.12))
Z_mej = modelo_svm_2d_mej.predict(np.c_[xx_mej.ravel(), yy_mej.ravel()]).reshape(xx_mej.shape)

"""
    new_content = before_gui + calc_logic + after_gui

    mainloop_idx = new_content.rfind("root.mainloop()")
    if mainloop_idx == -1:
        print("mainloop not found")
        return

    before_mainloop = new_content[:mainloop_idx]
    after_mainloop = new_content[mainloop_idx:]

    if is_lineal:
        plot_mej_logic = """
    for i, clase in enumerate(clases):
        idx = y_probar.values == clase
        ax_m.scatter(X_probar_pca_mej[idx, 0] * mult, X_probar_pca_mej[idx, 1] * mult,
                    label=f'Calidad {clase}', alpha=0.75,
                    color=cmap_vino(i / max(1, len(clases)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)
    
    xp_m = np.linspace((X_probar_pca_mej[:, 0].min()-1)*mult, (X_probar_pca_mej[:, 0].max()+1)*mult, 400)
    normas_m = np.linalg.norm(modelo_svm_2d_mej.coef_, axis=1)
    idx_principal_m = int(np.argmax(normas_m))
    for k, (w2, b2) in enumerate(zip(modelo_svm_2d_mej.coef_, modelo_svm_2d_mej.intercept_)):
        if abs(w2[1]) < 1e-10: continue
        yp_m = (-w2[0]*(xp_m/mult) - b2) / w2[1]
        yp_m = yp_m * mult
        if k == idx_principal_m:
            ax_m.plot(xp_m, yp_m, 'w-', linewidth=2.2, alpha=0.95, zorder=5)
        else:
            ax_m.plot(xp_m, yp_m, color='white', linewidth=0.8, alpha=0.22, zorder=4)
"""
    else:
        plot_mej_logic = """
    clases_sorted = sorted(objetivo.unique())
    ax_m.contourf(xx_mej * mult, yy_mej * mult, Z_mej, alpha=0.25, cmap='RdYlGn',
                 levels=np.arange(min(clases_sorted)-0.5, max(clases_sorted)+1, 1))

    for i, clase in enumerate(clases_sorted):
        idx = y_probar.values == clase
        ax_m.scatter(X_probar_pca_mej[idx, 0] * mult, X_probar_pca_mej[idx, 1] * mult,
                    label=f'Calidad {clase}', alpha=0.8,
                    color=cmap_vino(i / max(1, len(clases_sorted)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)
"""

    tabs_code = f"""
# ==========================================
# PESTAÑA 4: MODELO MEJORADO
# ==========================================
tab4 = ttk.Frame(notebook)
notebook.add(tab4, text='  ✨  Modelo Mejorado  ')

frame_vars = tk.Frame(tab4, bg='#1e1e2e')
frame_vars.pack(side='top', fill='x', pady=5)

lbl_vars_orig = tk.Label(frame_vars, text=f"VARIABLES ORIGINALES\\nNúmero total: {{len(variables_entrada.columns)}}", bg='#1e1e2e', fg='#cdd6f4', font=('Segoe UI', 10, 'bold'))
lbl_vars_orig.pack(side='left', padx=20)

txt_conservadas = "\\n".join([f"- {{v}}" for v in variables_conservadas])
lbl_vars_cons = tk.Label(frame_vars, text=f"VARIABLES CONSERVADAS\\n{{txt_conservadas}}", bg='#1e1e2e', fg='#a6e3a1', font=('Segoe UI', 9), justify='left')
lbl_vars_cons.pack(side='left', padx=20)

txt_eliminadas = "\\n".join([f"- {{v}}" for v in variables_a_eliminar])
lbl_vars_elim = tk.Label(frame_vars, text=f"VARIABLES ELIMINADAS\\n{{txt_eliminadas}}", bg='#1e1e2e', fg='#f38ba8', font=('Segoe UI', 9), justify='left')
lbl_vars_elim.pack(side='left', padx=20)

frame_graf = tk.Frame(tab4, bg='#1e1e2e')
frame_graf.pack(side='top', fill='both', expand=True)

frame_ctrl = tk.Frame(frame_graf, bg='#1e1e2e')
frame_ctrl.pack(side='top', fill='x', pady=5)
tk.Label(frame_ctrl, text="Multiplicador de visualización:", bg='#1e1e2e', fg='#cdd6f4').pack(side='left', padx=5)

mult_var = tk.DoubleVar(value=1.0)

fig4 = mfigure.Figure(figsize=(7, 5), facecolor='#1e1e2e')
ax4 = fig4.add_subplot(111)
ax4.set_facecolor='#1e1e2e'
canvas4 = FigureCanvasTkAgg(fig4, master=frame_graf)
canvas4.get_tk_widget().pack(fill='both', expand=True)

def actualizar_pca_mejorado(*args):
    ax4.clear()
    mult = mult_var.get()
    ax_m = ax4
    {plot_mej_logic}
    ax_m.set_title(f'PCA Modelo Mejorado (Mult: {{mult}})\\nPC1: {{varianza_mej[0]*100:.1f}}% | PC2: {{varianza_mej[1]*100:.1f}}%', color='#cdd6f4')
    ax_m.tick_params(colors='#cdd6f4')
    canvas4.draw()

mult_combo = ttk.Combobox(frame_ctrl, textvariable=mult_var, values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], width=5, state='readonly')
mult_combo.pack(side='left', padx=5)
mult_combo.bind('<<ComboboxSelected>>', actualizar_pca_mejorado)

actualizar_pca_mejorado()

# ==========================================
# PESTAÑA 5: VALIDACIÓN Y COMPARACIÓN
# ==========================================
tab5 = ttk.Frame(notebook)
notebook.add(tab5, text='  ⚖️  Validación y Comparación  ')

canvas_tab5 = tk.Canvas(tab5, bg='#1e1e2e', highlightthickness=0)
scrollbar5 = ttk.Scrollbar(tab5, orient="vertical", command=canvas_tab5.yview)
scrollable_frame5 = ttk.Frame(canvas_tab5)

scrollable_frame5.bind(
    "<Configure>",
    lambda e: canvas_tab5.configure(
        scrollregion=canvas_tab5.bbox("all")
    )
)

canvas_tab5.create_window((0, 0), window=scrollable_frame5, anchor="nw")
canvas_tab5.configure(yscrollcommand=scrollbar5.set)
canvas_tab5.pack(side="left", fill="both", expand=True)
scrollbar5.pack(side="right", fill="y")

frame_matrices = tk.Frame(scrollable_frame5, bg='#1e1e2e')
frame_matrices.pack(fill='x', pady=10)

fig_cm = mfigure.Figure(figsize=(10, 4), facecolor='#1e1e2e')
ax_cm1 = fig_cm.add_subplot(1, 2, 1)
ax_cm2 = fig_cm.add_subplot(1, 2, 2)
for ax_cm in (ax_cm1, ax_cm2): ax_cm.set_facecolor('#1e1e2e')

clases_labels = [str(c) for c in sorted(objetivo.unique())]
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax_cm1, xticklabels=clases_labels, yticklabels=clases_labels)
ax_cm1.set_title('Matriz Original', color='#cdd6f4')
ax_cm1.tick_params(colors='#cdd6f4')

sns.heatmap(matriz_confusion_mejorado, annot=True, fmt='d', cmap='Greens', ax=ax_cm2, xticklabels=clases_labels, yticklabels=clases_labels)
ax_cm2.set_title('Matriz Mejorado', color='#cdd6f4')
ax_cm2.tick_params(colors='#cdd6f4')
fig_cm.tight_layout()
embed_figure(frame_matrices, fig_cm)

frame_metricas = tk.Frame(scrollable_frame5, bg='#1e1e2e')
frame_metricas.pack(fill='x', pady=10)

tabla_texto = f\"\"\"MÉTRICA                 ORIGINAL       MEJORADO
------------------------------------------------------
Accuracy                {{acc_orig*100:6.2f}}%       {{acc_mej*100:6.2f}}%
Error                   {{err_orig*100:6.2f}}%       {{err_mej*100:6.2f}}%
Precision               {{prec_orig*100:6.2f}}%       {{prec_mej*100:6.2f}}%
Recall                  {{rec_orig*100:6.2f}}%       {{rec_mej*100:6.2f}}%
Specificity             {{spec_orig*100:6.2f}}%       {{spec_mej*100:6.2f}}%
F1                      {{f1_orig*100:6.2f}}%       {{f1_mej*100:6.2f}}%

ERROR DE ENTRENAMIENTO Y GENERALIZACIÓN
------------------------------------------------------
Accuracy entrenamiento  {{acc_train_orig*100:6.2f}}%       {{acc_train_mej*100:6.2f}}%
Error entrenamiento     {{(1-acc_train_orig)*100:6.2f}}%       {{(1-acc_train_mej)*100:6.2f}}%
Accuracy prueba         {{acc_orig*100:6.2f}}%       {{acc_mej*100:6.2f}}%
Error generalización    {{err_orig*100:6.2f}}%       {{err_mej*100:6.2f}}%

BRECHA DE GENERALIZACIÓN
------------------------------------------------------
Brecha Original: {{(acc_train_orig - acc_orig)*100:6.2f}}%
Brecha Mejorado: {{(acc_train_mej - acc_mej)*100:6.2f}}%
Interpretación: Valores altos indican sobreajuste.

RESUMEN FINAL
------------------------------------------------------
Variables               {{len(variables_entrada.columns):6d}}         {{len(variables_conservadas):6d}}
Variables eliminadas: {{', '.join(variables_a_eliminar)}}
Variables conservadas: {{', '.join(variables_conservadas)}}
Dif. Accuracy: {{(acc_mej - acc_orig)*100:+.2f}}%
Dif. Error:    {{(err_mej - err_orig)*100:+.2f}}%
Dif. F1:       {{(f1_mej - f1_orig)*100:+.2f}}%
Dif. Recall:   {{(rec_mej - rec_orig)*100:+.2f}}%
Dif. Precision:{{(prec_mej - prec_orig)*100:+.2f}}%
\"\"\"
lbl_metricas = tk.Label(frame_metricas, text=tabla_texto, bg='#1e1e2e', fg='#cdd6f4', font=('Consolas', 10), justify='left')
lbl_metricas.pack(padx=20, pady=10)

"""

    final_content = before_mainloop + tabs_code + after_mainloop
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"Processed {filename}")

process_file(r"c:\Users\alex\Downloads\modelacion v2\modelacion\Vinos\SVM_VINO\SVM_VINO.py", True)
process_file(r"c:\Users\alex\Downloads\modelacion v2\modelacion\Vinos\SVM_VINO\SVM_VINO_NO_LINEAL.py", False)

