# Explicación del Modelo SVM No Lineal Aplicado en Recursos Humanos (`NoLineal.py`)

## 1. Introducción

El archivo `NoLineal.py` implementa un modelo de **Máquinas de Vectores de Soporte No Lineales (Non-Linear Support Vector Machines)** utilizando el **Kernel RBF (Radial Basis Function / Gaussiano)** (`kernel='rbf'`).

A diferencia de los modelos lineales que intentan trazar una línea recta o hiperplano plano para separar las clases, el SVM No Lineal es capaz de construir **fronteras de decisión curvas y complejas** para clasificar el estado de terminación laboral de los empleados (`Termd`: `0` para Activo, `1` para Terminado).

---

## 2. Fundamento Matemático del SVM No Lineal: El "Truco del Kernel" (*Kernel Trick*)

Cuando un conjunto de datos no puede separarse adecuadamente mediante una línea recta en su espacio original, el algoritmo aplica el denominado **Truco del Kernel** (*Kernel Trick*).

### Proyecto a una Dimensión Superior ($\Phi$)
El algoritmo mapa implícitamente las características de entrada $x$ a un espacio de características de mayor dimensión $\Phi(x)$, donde las dos clases sí se vuelven linealmente separables.

### Ecuación del Kernel RBF (Gaussiano)
La función Kernel Gaussiana o de Base Radial calcula la similitud entre dos puntos $x$ y $x'$ mediante la fórmula:

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

Donde:
- $\|x - x'\|^2$ es la distancia euclidiana al cuadrado entre dos observaciones.
- $\gamma$ (gamma) es el parámetro que controla el alcance de la influencia de cada ejemplo individual.

### Parámetros Clave del Modelo
1. **`kernel='rbf'`**: Permite crear contornos o regiones cerradas alrededor de las observaciones.
2. **`C = 1.0` (Penalización):** Controla el equilibrio entre lograr un margen amplio y minimizar los errores de clasificación.
3. **`gamma='scale'`:** Define la curvatura de la frontera. Un $\gamma$ más alto genera bordes más ajustados y complejos.

---

## 3. Preprocesamiento de Datos Aplicado

El script mantiene la misma metodología de preprocesamiento que `Limpieza_HR.py` y `SVMRRHH.py` para asegurar comparabilidad:

1. **Carga y Limpieza:** Eliminación de duplicados y filtro de nulos en `Termd`.
2. **Selección Numérica:** Filtrado de variables de tipo entero y flotante, rellenando vacíos con `0`.
3. **Estandarización (`StandardScaler`):** Normalización de características ($z = \frac{x - \mu}{\sigma}$), fundamental para que la distancia euclidiana en el kernel RBF no sea distorsionada por variables de distinta escala.
4. **División 80/20:** 248 registros para entrenamiento y 63 registros para evaluación en prueba (`random_state=77`).

---

## 4. Representación Gráfica de la Frontera No Lineal

Al ejecutar `NoLineal.py`, el script genera dos subgráficas:

1. **Matriz de Confusión Térmica (Naranja):** Muestra el desempeño visual en las clasificaciones reales vs predichas.
2. **Frontera de Decisión No Lineal (PCA 2D):**
   - Utiliza **PCA (Principal Component Analysis)** para proyectar las 17 características a 2 componentes principales.
   - Muestra cómo la frontera azul/roja adopta una **forma curva adaptativa** en lugar de una línea recta rígida, delimitando con mayor flexibilidad las regiones pertenecientes a los empleados que causaron baja.

---

## 5. Evaluación del Modelo y Métricas Resultantes

Al ejecutar `python NoLineal.py`, se registran los siguientes resultados:

* **Exactitud (Accuracy):** **`96.83%`** (61 de 63 predicciones correctas en prueba).
* **Matriz de Confusión:**
  ```
  [[43  2]   -> 43 Activos correctos, 2 Falsos Positivos
   [ 0 18]]  -> 18 Terminados correctos, 0 Falsos Negativos
  ```
* **Destacado Principal - Recall del 100% en la Clase 1:**
  - El modelo No Lineal logró detectar **el 100% de los empleados que renunciaron o fueron terminados (`Termd = 1`)** (Recall = 1.00), sin cometer ningún Falso Negativo.
  - Esto es crucial en Recursos Humanos, ya que omitir la detección de un empleado en riesgo de rotación (Falso Negativo) suele ser más costoso que una falsa alarma.

---

## 6. Cuadro Comparativo: SVM Lineal vs. SVM No Lineal (RBF)

| Característica | SVM Lineal (`SVMRRHH.py`) | SVM No Lineal RBF (`NoLineal.py`) |
| :--- | :--- | :--- |
| **Frontera de Decisión** | Línea recta / Hiperplano plano | Curva / Adaptativa |
| **Complejidad Computacional** | Menor (más rápido) | Mayor (cálculo de distancias RBF) |
| **Exactitud General** | 96.83% | 96.83% |
| **Recall Clase 1 (Terminados)**| 94.4% (1 error) | **100.0% (0 errores)** |
| **Falsos Negativos (Termd=1)** | 1 empleado no detectado | **0 empleados no detectados** |

---

## 7. Conclusión

El modelo SVM No Lineal con Kernel RBF demuestra ser sumamente efectivo para capturar patrones no lineales en los datos de Recursos Humanos. Su capacidad para delimitar fronteras curvas le permitió alcanzar una sensibilidad óptima (Recall 100%) en la identificación de bajas de personal.
