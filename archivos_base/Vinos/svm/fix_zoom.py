import os
import numpy as np

def update_file(filepath, is_lineal):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    start_idx = content.find("def actualizar_pca_mejorado(*args):")
    if start_idx == -1: return

    end_idx = content.find("mult_combo = ttk.Combobox", start_idx)
    if end_idx == -1: return

    before = content[:start_idx]
    after = content[end_idx:]

    if is_lineal:
        func = """def actualizar_pca_mejorado(*args):
    ax4.clear()
    mult = mult_var.get()
    ax_m = ax4
    ax_m.set_facecolor('#1e1e2e')

    # Límites fijos para evitar que Matplotlib haga auto-zoom
    mult_max = 3.0
    graf_max_x = np.max(np.abs(X_probar_pca_mej[:, 0])) * mult_max + 2.0
    graf_max_y = np.max(np.abs(X_probar_pca_mej[:, 1])) * mult_max + 2.0

    # Malla visual fija que abarca todo el espacio posible
    xx_visual, yy_visual = np.meshgrid(np.arange(-graf_max_x, graf_max_x, 0.1),
                                       np.arange(-graf_max_y, graf_max_y, 0.1))

    # Predecir las regiones escalando inversamente las coordenadas visuales
    Z = modelo_svm_2d_mej.predict(np.c_[xx_visual.ravel() / mult, yy_visual.ravel() / mult]).reshape(xx_visual.shape)

    clases_sorted = sorted(y_probar.unique())
    cmap_vino = plt.cm.RdYlGn

    # Zonas de decisión con alpha 0.12 para que el fondo oscuro sobresalga
    ax_m.contourf(xx_visual, yy_visual, Z, alpha=0.12, cmap='RdYlGn',
                  levels=np.arange(min(clases_sorted)-0.5, max(clases_sorted)+1, 1))

    # Puntos con coordenadas multiplicadas
    for i, clase in enumerate(clases_sorted):
        idx = y_probar.values == clase
        ax_m.scatter(X_probar_pca_mej[idx, 0] * mult, X_probar_pca_mej[idx, 1] * mult,
                    label=f'Calidad {clase}', alpha=0.75,
                    color=cmap_vino(i / max(1, len(clases_sorted)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)

    # Hiperplanos
    xp_visual = np.linspace(-graf_max_x, graf_max_x, 400)
    normas_m = np.linalg.norm(modelo_svm_2d_mej.coef_, axis=1)
    idx_principal_m = int(np.argmax(normas_m))
    for k, (w2, b2) in enumerate(zip(modelo_svm_2d_mej.coef_, modelo_svm_2d_mej.intercept_)):
        if abs(w2[1]) < 1e-10: continue
        yp_visual = (-w2[0]*(xp_visual/mult) - b2) / w2[1]
        yp_visual = yp_visual * mult
        if k == idx_principal_m:
            ax_m.plot(xp_visual, yp_visual, 'w-', linewidth=2.5, alpha=1.0, zorder=5, label='Hiperplano principal')
        else:
            ax_m.plot(xp_visual, yp_visual, color='white', linewidth=1.2, alpha=0.6, linestyle='--', zorder=4)

    # Fijar estrictamente los límites de los ejes para que se note la separación
    ax_m.set_xlim(-graf_max_x, graf_max_x)
    ax_m.set_ylim(-graf_max_y, graf_max_y)

    ax_m.set_title(f'SVM Lineal — Visualización PCA Mejorado (Mult: {mult})\\nPC1: {varianza_mej[0]*100:.1f}%  |  PC2: {varianza_mej[1]*100:.1f}%',
                  fontsize=12, fontweight='bold', color='#cdd6f4', pad=12)
    ax_m.set_xlabel(f'Componente Principal 1 ({varianza_mej[0]*100:.2f}% varianza)', color='#cdd6f4')
    ax_m.set_ylabel(f'Componente Principal 2 ({varianza_mej[1]*100:.2f}% varianza)', color='#cdd6f4')
    ax_m.tick_params(colors='#cdd6f4')
    for spine in ax_m.spines.values():
        spine.set_edgecolor('#313244')
    ax_m.legend(loc='best', fontsize=9, facecolor='#313244', labelcolor='#cdd6f4')
    ax_m.grid(True, color='#313244', linewidth=0.5, alpha=0.6)
    fig4.tight_layout(pad=2)

    canvas4.draw()

"""
    else:
        func = """def actualizar_pca_mejorado(*args):
    ax4.clear()
    mult = mult_var.get()
    ax_m = ax4
    ax_m.set_facecolor('#1e1e2e')

    # Límites fijos para evitar auto-zoom
    mult_max = 3.0
    graf_max_x = np.max(np.abs(X_probar_pca_mej[:, 0])) * mult_max + 2.0
    graf_max_y = np.max(np.abs(X_probar_pca_mej[:, 1])) * mult_max + 2.0

    xx_visual, yy_visual = np.meshgrid(np.arange(-graf_max_x, graf_max_x, 0.1),
                                       np.arange(-graf_max_y, graf_max_y, 0.1))

    # Inversa para predecir regiones
    Z = modelo_svm_2d_mej.predict(np.c_[xx_visual.ravel() / mult, yy_visual.ravel() / mult]).reshape(xx_visual.shape)

    clases_sorted = sorted(objetivo.unique())
    cmap_vino = plt.cm.RdYlGn

    ax_m.contourf(xx_visual, yy_visual, Z, alpha=0.12, cmap='RdYlGn',
                 levels=np.arange(min(clases_sorted)-0.5, max(clases_sorted)+1, 1))

    for i, clase in enumerate(clases_sorted):
        idx = y_probar.values == clase
        ax_m.scatter(X_probar_pca_mej[idx, 0] * mult, X_probar_pca_mej[idx, 1] * mult,
                    label=f'Calidad {clase}', alpha=0.8,
                    color=cmap_vino(i / max(1, len(clases_sorted)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)

    ax_m.set_xlim(-graf_max_x, graf_max_x)
    ax_m.set_ylim(-graf_max_y, graf_max_y)

    ax_m.set_title(f'SVM No Lineal (RBF) — Visualización PCA Mejorado (Mult: {mult})\\nPC1: {varianza_mej[0]*100:.1f}%  |  PC2: {varianza_mej[1]*100:.1f}%',
                  fontsize=12, fontweight='bold', color='#cdd6f4', pad=12)
    ax_m.set_xlabel(f'Componente Principal 1 ({varianza_mej[0]*100:.2f}% varianza)', color='#cdd6f4')
    ax_m.set_ylabel(f'Componente Principal 2 ({varianza_mej[1]*100:.2f}% varianza)', color='#cdd6f4')
    ax_m.tick_params(colors='#cdd6f4')
    for spine in ax_m.spines.values():
        spine.set_edgecolor('#313244')
    ax_m.legend(loc='best', fontsize=9, facecolor='#313244', labelcolor='#cdd6f4')
    ax_m.grid(True, color='#313244', linewidth=0.5, alpha=0.5)
    fig4.tight_layout(pad=2)

    canvas4.draw()

"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(before + func + after)
    print(f"Updated {filepath}")

update_file('SVM_VINO.py', True)
update_file('SVM_VINO_NO_LINEAL.py', False)
