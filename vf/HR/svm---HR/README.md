# SVM Lineal y No Lineal - Predicción de Desvinculación Laboral (`HRDataset_v14.csv`)

Este módulo implementa y compara dos modelos de **Máquinas de Vectores de Soporte (SVM)** — uno con kernel lineal y otro con kernel RBF (no lineal) — para clasificar si un empleado está activo (`Termd = 0`) o ha causado baja (`Termd = 1`).

---

## Modelo empleado: SVM Lineal (`SVMRRHH.py`)

El algoritmo busca el **hiperplano de margen máximo** que separa las dos clases. En un espacio de $n$ características, el hiperplano se define como:

$$w \cdot x + b = 0$$

Los vectores de soporte son los puntos más cercanos al hiperplano, y el modelo maximiza el margen entre ellos. El parámetro `C=1.0` controla la penalización por clasificaciones incorrectas.

### Flujo del algoritmo (SVM Lineal)

1. Carga del dataset, eliminación de duplicados y nulos en `Termd`.
2. Selección de variables numéricas (`int64`, `float64`); relleno de nulos con `0`.
3. Estandarización con `StandardScaler`.
4. División 80/20 (`random_state=77`).
5. Entrenamiento: `SVC(kernel='linear', C=1.0)`.
6. Evaluación: exactitud, matriz de confusión y reporte de clasificación.
7. Proyección 2D vía PCA para visualizar el hiperplano y sus márgenes.
8. Visualización de los 15 coeficientes más y menos representativos del hiperplano.

---

## Modelo empleado: SVM No Lineal - Kernel RBF (`NoLineal.py`)

Cuando los datos no son linealmente separables, el algoritmo aplica el **truco del kernel** (*Kernel Trick*) para mapearlos implícitamente a un espacio de mayor dimensión donde sí lo son. La función Kernel Gaussiana (RBF) es:

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

donde $\gamma$ controla la curvatura de la frontera de decisión.

### Flujo del algoritmo (SVM No Lineal)

1. Mismo preprocesamiento que el modelo lineal (pasos 1–4).
2. Entrenamiento: `SVC(kernel='rbf', C=1.0, gamma='scale')`.
3. Evaluación: exactitud, matriz de confusión y reporte de clasificación.
4. Proyección 2D vía PCA para visualizar la frontera curva.
5. Importancia de variables calculada por **permutación** (`permutation_importance`, 15 repeticiones).

---

## Comparativa de modelos

| Característica | SVM Lineal | SVM No Lineal (RBF) |
|---|---|---|
| Frontera de decisión | Hiperplano plano | Curva adaptativa |
| Exactitud general | ~96.83% | ~96.83% |
| Recall clase 1 (Terminados) | ~94.4% | ~100% |
| Falsos negativos | 1 | 0 |

---

## Estructura del módulo

```text
📁 HR/
├── 📄 HRDataset_v14.csv          ← dataset en la raíz (buscado dinámicamente)
└── 📁 svm---RH/
    ├── 📄 SVMRRHH.py
    ├── 📄 NoLineal.py
    ├── 📄 NoLineal.md
    └── 📄 README.md
```

> Ambos scripts detectan automáticamente `HRDataset_v14.csv` subiendo en la jerarquía de carpetas.

---

## Dependencias

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## Ejecutar

```bash
python SVMRRHH.py
python NoLineal.py
```