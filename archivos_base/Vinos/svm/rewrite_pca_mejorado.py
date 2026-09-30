import os
import re

def rewrite_file(filepath, is_lineal):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    start_idx = content.find("def actualizar_pca_mejorado(*args):")
    if start_idx == -1:
        print(f"Error: actualizar_pca_mejorado not found in {filepath}")
        return

    end_idx = content.find("mult_combo = ttk.Combobox", start_idx)
    if end_idx == -1:
        print(f"Error: mult_combo not found in {filepath}")
        return

    before_func = content[:start_idx]
    after_func = content[end_idx:]

    if is_lineal:
        func_content = """def actualizar_pca_mejorado(*args):
    ax4.clear()
    mult = mult_var.get()
    ax_m = ax4

    margen = 2.0
    x_min_mesh = X_probar_pca_mej[:, 0].min() - margen
    x_max_mesh = X_probar_pca_mej[:, 0].max() + margen
    y_min_mesh = X_probar_pca_mej[:, 1].min() - margen
    y_max_mesh = X_probar_pca_mej[:, 1].max() + margen

    xx, yy = np.meshgrid(np.arange(x_min_mesh, x_max_mesh, 0.05),
                         np.arange(y_min_mesh, y_max_mesh, 0.05))
    Z = modelo_svm_2d_mej.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    clases_sorted = sorted(y_probar.unique())
    cmap_vino = plt.cm.RdYlGn

    ax_m.contourf(xx * mult, yy * mult, Z, alpha=0.25, cmap='RdYlGn',
                  levels=np.arange(min(clases_sorted)-0.5, max(clases_sorted)+1, 1))

    for i, clase in enumerate(clases_sorted):
        idx = y_probar.values == clase
        ax_m.scatter(X_probar_pca_mej[idx, 0] * mult, X_probar_pca_mej[idx, 1] * mult,
                    label=f'Calidad {clase}', alpha=0.75,
                    color=cmap_vino(i / max(1, len(clases_sorted)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)

    xp_m = np.linspace(x_min_mesh*mult, x_max_mesh*mult, 400)
    normas_m = np.linalg.norm(modelo_svm_2d_mej.coef_, axis=1)
    idx_principal_m = int(np.argmax(normas_m))
    for k, (w2, b2) in enumerate(zip(modelo_svm_2d_mej.coef_, modelo_svm_2d_mej.intercept_)):
        if abs(w2[1]) < 1e-10: continue
        yp_m = (-w2[0]*(xp_m/mult) - b2) / w2[1]
        yp_m = yp_m * mult
        if k == idx_principal_m:
            ax_m.plot(xp_m, yp_m, 'w-', linewidth=2.2, alpha=0.95, zorder=5, label='Hiperplano principal')
        else:
            ax_m.plot(xp_m, yp_m, color='white', linewidth=0.8, alpha=0.22, zorder=4)

    ax_m.set_xlim(x_min_mesh * mult, x_max_mesh * mult)
    ax_m.set_ylim(y_min_mesh * mult, y_max_mesh * mult)

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
        func_content = """def actualizar_pca_mejorado(*args):
    ax4.clear()
    mult = mult_var.get()
    ax_m = ax4

    clases_sorted = sorted(objetivo.unique())
    cmap_vino = plt.cm.RdYlGn

    ax_m.contourf(xx_mej * mult, yy_mej * mult, Z_mej, alpha=0.25, cmap='RdYlGn',
                 levels=np.arange(min(clases_sorted)-0.5, max(clases_sorted)+1, 1))

    for i, clase in enumerate(clases_sorted):
        idx = y_probar.values == clase
        ax_m.scatter(X_probar_pca_mej[idx, 0] * mult, X_probar_pca_mej[idx, 1] * mult,
                    label=f'Calidad {clase}', alpha=0.8,
                    color=cmap_vino(i / max(1, len(clases_sorted)-1)), edgecolors='#cdd6f4', linewidths=0.3, zorder=3)

    ax_m.set_xlim((X_probar_pca_mej[:, 0].min()-12)*mult, (X_probar_pca_mej[:, 0].max()+12)*mult)
    ax_m.set_ylim((X_probar_pca_mej[:, 1].min()-12)*mult, (X_probar_pca_mej[:, 1].max()+12)*mult)

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

    new_content = before_func + func_content + after_func

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Updated {filepath}")

rewrite_file("c:\\\\Users\\\\alex\\\\Downloads\\\\modelacion v2\\\\modelacion\\\\Vinos\\\\SVM_VINO\\\\SVM_VINO.py", True)
rewrite_file("c:\\\\Users\\\\alex\\\\Downloads\\\\modelacion v2\\\\modelacion\\\\Vinos\\\\SVM_VINO\\\\SVM_VINO_NO_LINEAL.py", False)
