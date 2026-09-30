import os

def update_visuals(filepath, is_lineal):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()


    content = content.replace(
        "ax_m.contourf(xx * mult, yy * mult, Z, alpha=0.25, cmap='RdYlGn',",
        "ax_m.contourf(xx * mult, yy * mult, Z, alpha=0.12, cmap='RdYlGn',"
    )
    content = content.replace(
        "ax_m.contourf(xx_mej * mult, yy_mej * mult, Z_mej, alpha=0.25, cmap='RdYlGn',",
        "ax_m.contourf(xx_mej * mult, yy_mej * mult, Z_mej, alpha=0.12, cmap='RdYlGn',"
    )

    if is_lineal:

        content = content.replace(
            "ax_m.plot(xp_m, yp_m, 'w-', linewidth=2.2, alpha=0.95, zorder=5, label='Hiperplano principal')",
            "ax_m.plot(xp_m, yp_m, 'w-', linewidth=2.5, alpha=1.0, zorder=5, label='Hiperplano principal')"
        )
        content = content.replace(
            "ax_m.plot(xp_m, yp_m, color='white', linewidth=0.8, alpha=0.22, zorder=4)",
            "ax_m.plot(xp_m, yp_m, color='white', linewidth=1.2, alpha=0.6, linestyle='--', zorder=4)"
        )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {filepath}")

update_visuals('SVM_VINO.py', True)
update_visuals('SVM_VINO_NO_LINEAL.py', False)
