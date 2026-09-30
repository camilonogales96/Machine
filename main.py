import os
import sys


if 'TCL_LIBRARY' not in os.environ or 'TK_LIBRARY' not in os.environ:
    base_prefix = getattr(sys, 'base_prefix', sys.prefix)
    tcl_cand = os.path.join(base_prefix, 'tcl', 'tcl8.6')
    tk_cand = os.path.join(base_prefix, 'tcl', 'tk8.6')
    if os.path.exists(tcl_cand):
        os.environ['TCL_LIBRARY'] = tcl_cand
    if os.path.exists(tk_cand):
        os.environ['TK_LIBRARY'] = tk_cand

import importlib.util
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')


try:
    import data_cleaner
    data_cleaner.ejecutar_limpieza_completa()
except Exception as e:
    print(f"[ADVERTENCIA] No se pudo ejecutar data_cleaner directamente: {e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cargar_modulo_funcion(subpath, funcion_nombre):
    """Carga dinámicamente una función de un archivo python en la carpeta vf."""
    abs_path = os.path.join(BASE_DIR, 'vf', *subpath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"No se encontró el archivo: {abs_path}")

    mod_name = os.path.splitext(os.path.basename(abs_path))[0].replace(' ', '_').replace('-', '_')
    spec = importlib.util.spec_from_file_location(mod_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, funcion_nombre):
        raise AttributeError(f"El módulo {mod_name} no posee la función {funcion_nombre}")
    return getattr(mod, funcion_nombre)

class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dashboard de Machine Learning - Vinos & Recursos Humanos")
        self.root.geometry("1300x850")
        self.root.configure(bg='#11111b')

        self.current_button = None


        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
        self.style.configure('TNotebook.Tab', background='#313244', foreground='#cdd6f4', padding=[12, 6], font=('Segoe UI', 9, 'bold'))
        self.style.map('TNotebook.Tab', background=[('selected', '#89b4fa')], foreground=[('selected', '#11111b')])


        self.main_container = tk.Frame(self.root, bg='#11111b')
        self.main_container.pack(fill='both', expand=True)


        self.sidebar = tk.Frame(self.main_container, bg='#181825', width=260)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)


        header_frame = tk.Frame(self.sidebar, bg='#1e1e2e', height=70)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        lbl_title = tk.Label(header_frame, text="🤖 ML DASHBOARD", bg='#1e1e2e', fg='#89b4fa', font=('Segoe UI', 13, 'bold'))
        lbl_title.pack(pady=12)
        lbl_sub = tk.Label(header_frame, text="Modelos de Clasificación y Regresión", bg='#1e1e2e', fg='#a6adc8', font=('Segoe UI', 8))
        lbl_sub.pack()


        self.content_area = tk.Frame(self.main_container, bg='#1e1e2e')
        self.content_area.pack(side='right', fill='both', expand=True)


        self.modulos = {
            'vinos_regresion': {
                'cat': 'VINOS',
                'label': '📈 Regresión Lineal',
                'path': ['Vinos', 'regresion----Vinos', 'Regresion_Vinos (1).py'],
                'func': 'crear_interfaz_vinos_regresion'
            },
            'vinos_arbol': {
                'cat': 'VINOS',
                'label': '🌳 Árbol de Decisión',
                'path': ['Vinos', 'tree-----Vinos', 'ArbolDecisionVinos.py'],
                'func': 'crear_interfaz_vinos_arbol'
            },
            'vinos_svm_lineal': {
                'cat': 'VINOS',
                'label': '⚡ SVM Lineal',
                'path': ['Vinos', 'svm---Vinos', 'SVM_VINO.py'],
                'func': 'crear_interfaz_vinos_svm_lineal'
            },
            'vinos_svm_nolineal': {
                'cat': 'VINOS',
                'label': '🌀 SVM No Lineal (RBF)',
                'path': ['Vinos', 'svm---Vinos', 'SVM_VINO_NO_LINEAL.py'],
                'func': 'crear_interfaz_vinos_svm_nolineal'
            },
            'vinos_rna': {
                'cat': 'VINOS',
                'label': '🧠 Red Neuronal (RNA)',
                'path': ['Vinos', 'RNA--Vinos', 'RNA-Vinos.py'],
                'func': 'crear_interfaz_vinos_rna'
            },
            'vinos_resultados': {
                'cat': 'VINOS',
                'label': '🏆 Resultados y Comparativa',
                'path': ['Vinos', 'resultados_vinos.py'],
                'func': 'crear_interfaz_vinos_resultados'
            },
            'hr_regresion': {
                'cat': 'HR',
                'label': '📈 Regresión Lineal',
                'path': ['HR', 'regresion----HR', 'RegresionLineal1HumanResources (1).py'],
                'func': 'crear_interfaz_hr_regresion'
            },
            'hr_arbol': {
                'cat': 'HR',
                'label': '🌳 Árbol de Decisión',
                'path': ['HR', 'tree-----HR', 'ArbolDecision.py'],
                'func': 'crear_interfaz_hr_arbol'
            },
            'hr_svm_lineal': {
                'cat': 'HR',
                'label': '⚡ SVM Lineal',
                'path': ['HR', 'svm---HR', 'SVMRRHH.py'],
                'func': 'crear_interfaz_svm_lineal'
            },
            'hr_svm_nolineal': {
                'cat': 'HR',
                'label': '🌀 SVM No Lineal (RBF)',
                'path': ['HR', 'svm---HR', 'NoLineal.py'],
                'func': 'crear_interfaz_svm_nolineal'
            },
            'hr_rna': {
                'cat': 'HR',
                'label': '🧠 Red Neuronal (RNA)',
                'path': ['HR', 'RNA--HR', 'RNA-RH.py'],
                'func': 'crear_interfaz_hr_rna'
            },
            'hr_resultados': {
                'cat': 'HR',
                'label': '🏆 Resultados y Comparativa',
                'path': ['HR', 'resultados_hr.py'],
                'func': 'crear_interfaz_hr_resultados'
            },
        }

        self.buttons = {}
        self.construir_menu()
        self.cargar_modulo('vinos_regresion')

    def construir_menu(self):

        tk.Label(self.sidebar, text="🍷 DATASET VINOS", bg='#181825', fg='#f9e2af', font=('Segoe UI', 9, 'bold'), anchor='w').pack(fill='x', padx=15, pady=(15, 5))
        for key in ['vinos_regresion', 'vinos_arbol', 'vinos_svm_lineal', 'vinos_svm_nolineal', 'vinos_rna', 'vinos_resultados']:
            self.crear_boton_menu(key)


        tk.Label(self.sidebar, text="👥 RECURSOS HUMANOS (HR)", bg='#181825', fg='#a6e3a1', font=('Segoe UI', 9, 'bold'), anchor='w').pack(fill='x', padx=15, pady=(20, 5))
        for key in ['hr_regresion', 'hr_arbol', 'hr_svm_lineal', 'hr_svm_nolineal', 'hr_rna', 'hr_resultados']:
            self.crear_boton_menu(key)

    def crear_boton_menu(self, key):
        info = self.modulos[key]
        btn = tk.Button(
            self.sidebar,
            text=f"  {info['label']}",
            bg='#181825',
            fg='#cdd6f4',
            activebackground='#313244',
            activeforeground='#ffffff',
            bd=0,
            font=('Segoe UI', 9),
            anchor='w',
            cursor='hand2',
            command=lambda k=key: self.cargar_modulo(k)
        )
        btn.pack(fill='x', padx=10, pady=2, ipady=5)
        self.buttons[key] = btn

    def cargar_modulo(self, key):

        if self.current_button:
            self.current_button.config(bg='#181825', fg='#cdd6f4', font=('Segoe UI', 9))

        btn = self.buttons[key]
        btn.config(bg='#89b4fa', fg='#11111b', font=('Segoe UI', 9, 'bold'))
        self.current_button = btn


        for child in self.content_area.winfo_children():
            child.destroy()

        info = self.modulos[key]
        try:
            func = cargar_modulo_funcion(info['path'], info['func'])
            container = tk.Frame(self.content_area, bg='#1e1e2e')
            container.pack(fill='both', expand=True)
            func(container)
        except Exception as e:
            err_lbl = tk.Label(self.content_area, text=f"Error cargando módulo '{info['label']}':\n{e}", bg='#1e1e2e', fg='#f38ba8', font=('Segoe UI', 11))
            err_lbl.pack(pady=40)

if __name__ == '__main__':
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()
