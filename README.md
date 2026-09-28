# Proyecto Machine Learning: Limpieza y Evaluación de Modelos (HR & Vinos)

Este proyecto es un entorno de Machine Learning interactivo que realiza la limpieza, normalización y análisis predictivo mediante modelos de Regresión, Árboles de Decisión y Máquinas de Vectores de Soporte (SVM) sobre dos datasets principales:
1. **HR Dataset (Recursos Humanos)**: Clasificación de rotación / terminación laboral (`Termd`).
2. **Wine Quality Dataset (Calidad del Vino)**: Análisis y clasificación de calidad del vino tinto.

---

## 📋 Requisitos Previos

* **Python 3.8+** (Probado y compatible con Python 3.10+)
* **Tkinter** (Incluido por defecto en la mayoría de instalaciones de Python para Windows y macOS). En Linux/Ubuntu se instala con `sudo apt-get install python3-tk`.

---

## 🚀 Guía de Instalación y Configuración

Sigue estos comandos paso a paso en tu terminal (PowerShell, CMD o Bash) para preparar el entorno e iniciar el proyecto:

### 1. Clonar o Abrir el Proyecto
Abre la terminal en la carpeta raíz del proyecto:

### 2. Crear un Entorno Virtual (Recomendado)
Crear un entorno virtual ayuda a aislar las dependencias del proyecto.

**En Windows (PowerShell / CMD):**
```powershell
python -m venv venv
```

**En Linux / macOS:**
```bash
python3 -m venv venv
```

### 3. Activar el Entorno Virtual

**En Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```
*(Si encuentras un error de políticas de ejecución en PowerShell, ejecuta primero `Set-ExecutionPolicy Unrestricted -Scope Process`)*

**En Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**En Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Instalar las Dependencias
Instala las librerías necesarias ejecutando:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Ejecución del Proyecto

### Opción A: Ejecutar la Interfaz Gráfica (Dashboard Completo)
El script principal `main.py` ejecuta automáticamente la limpieza de datos (si no se ha ejecutado aún) y despliega una interfaz de usuario interactiva (GUI) para entrenar modelos, visualizar métricas y analizar características:

```bash
python main.py
```

### Opción B: Limpieza de Datos Independiente
Si deseas únicamente limpiar y exportar los datasets procesados (`dataset_hr_limpio.csv` y `dataset_vino_limpio.csv`), ejecuta:

```bash
python data_cleaner.py
```

---

## 📁 Estructura del Proyecto

```text
Machine Proyecto/
├── HRDataset_v14.csv        # Dataset original de Recursos Humanos
├── winequality-red.csv      # Dataset original de Calidad del Vino
├── data_cleaner.py          # Script de preprocesamiento, limpieza y normalización
├── main.py                  # Dashboard interactivo Tkinter y evaluador de modelos ML
├── requirements.txt         # Lista de dependencias del proyecto Python
└── README.md                # Documentación del proyecto
```

---

## 🛠️ Tecnologías Utilizadas

* **Pandas & NumPy**: Procesamiento y manipulación de datos.
* **Scikit-Learn**: Modelado de Machine Learning (SVM, Árboles de Decisión, Regresión Lineal/Ridge, Matriz de Confusión).
* **Matplotlib & Seaborn**: Generación de gráficos y mapas de calor.
* **Tkinter**: Interfaz gráfica de usuario interactiva.
