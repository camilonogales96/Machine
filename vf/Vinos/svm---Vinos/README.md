# SVM Lineal y No Lineal - Clasificación de Calidad del Vino (`winequality-red.csv`)

Este módulo implementa y compara dos modelos de **Máquinas de Vectores de Soporte (SVM)** para clasificar la calidad del vino tinto (escala 3–8) a partir de sus características fisicoquímicas.

---

## Modelo empleado: SVM Lineal (`SVM_VINO.py`)

El algoritmo construye un conjunto de **hiperplanos lineales** que separan las clases de calidad con el mayor margen posible. Para clasificación multiclase se usa la estrategia *One-vs-One* implícita de `SVC`. El hiperplano separador se define como:

$$w \cdot x + b = 0$$

El parámetro `C=1.0` penaliza los puntos que caen dentro del margen o son mal clasificados.

### Flujo del algoritmo (SVM Lineal)

1. Búsqueda dinámica de `dataset_vino_limpio.csv` o `winequality-red.csv` subiendo por la jerarquía de carpetas.
2. Separación de variable objetivo (`quality`) y características.
3. Estandarización con `StandardScaler`.
4. División 80/20 con estratificación por clase (`random_state=77`).
5. Entrenamiento: `SVC(kernel='linear', C=1.0)`.
6. Evaluación: exactitud, matriz de confusión y reporte de clasificación.
7. Proyección 2D vía PCA del conjunto de prueba para visualizar los hiperplanos.
8. Visualización de los 15 coeficientes de mayor y menor magnitud por clase.

---

## Modelo empleado: SVM No Lineal - Kernel RBF (`SVM_VINO_NO_LINEAL.py`)

Para capturar relaciones no lineales entre los compuestos químicos y la calidad percibida, se aplica el **truco del kernel** con función Gaussiana (RBF):

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

El parámetro `gamma='scale'` ajusta automáticamente $\gamma = \frac{1}{n\_features \cdot \text{Var}(X)}$.

### Flujo del algoritmo (SVM No Lineal)

1. Mismo preprocesamiento que el modelo lineal (pasos 1–4).
2. Entrenamiento: `SVC(kernel='rbf', C=1.0, gamma='scale')`.
3. Evaluación: exactitud, matriz de confusión y reporte de clasificación.
4. Proyección 2D vía PCA del conjunto de entrenamiento; generación de la malla para la frontera curva.
5. Importancia de variables calculada por **permutación** (`permutation_importance`, 10 repeticiones).

---

## Estructura del módulo

```text
📁 Machine/
├── 📄 winequality-red.csv        ← dataset en la raíz (buscado dinámicamente)
└── 📁 modelacion/Vinos/SVM_VINO/
    ├── 📄 SVM_VINO.py
    ├── 📄 SVM_VINO_NO_LINEAL.py
    └── 📄 README.md
```

---

## Dependencias

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## Ejecutar

### SVM Lineal

```powershell
py .\SVM_VINO.py
```

### SVM No Lineal

```powershell
py .\SVM_VINO_NO_LINEAL.py
```