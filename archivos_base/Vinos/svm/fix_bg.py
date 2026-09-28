import os

def insert_facecolor(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    target = "    ax_m = ax4\n"
    if target not in content:
        print(f"Target not found in {filepath}")
        return

    replacement = target + "    ax_m.set_facecolor('#1e1e2e')\n"
    new_content = content.replace(target, replacement)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Updated {filepath}")

insert_facecolor(r"c:\Users\alex\Downloads\modelacion v2\modelacion\Vinos\SVM_VINO\SVM_VINO.py")
insert_facecolor(r"c:\Users\alex\Downloads\modelacion v2\modelacion\Vinos\SVM_VINO\SVM_VINO_NO_LINEAL.py")
