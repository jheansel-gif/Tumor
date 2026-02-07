# 🧠 Clasificación de Tumores con TensorFlow

Este proyecto implementa una red neuronal para la **clasificación de tumores de mama (benignos vs malignos)** utilizando el dataset clásico de *Breast Cancer Wisconsin* disponible en `scikit-learn`.

El objetivo es mostrar un flujo completo de **Machine Learning con Deep Learning**: carga de datos, preprocesamiento, entrenamiento, validación, evaluación y predicción.

---

## 📊 Dataset

Se utiliza el dataset:

- `load_breast_cancer()` de `sklearn.datasets`
- Contiene características numéricas de tumores (radio, textura, perímetro, área, etc.)
- Etiquetas:
  - `0` → Benigno  
  - `1` → Maligno  

---

## ⚙️ Flujo del Algoritmo

1. **Carga de datos**
   ```python
   data = load_breast_cancer()
   X = data.data
   y = data.target

1. Normalización
Se estandarizan las variables con StandardScaler para mejorar la convergencia del modelo.

2. División Train / Test
train_test_split(X, y, test_size=0.2, random_state=42)


3. Definición del modelo
Red neuronal completamente conectada (MLP):

- Capa densa de 32 neuronas (ReLU)

- Capa densa de 16 neuronas (ReLU)

- Capa de salida de 1 neurona (Sigmoid)

4. Entrenamiento

- Optimizador: Adam

- Función de pérdida: Binary Crossentropy

- Métrica: Accuracy

5. Evaluación
Se evalúa el desempeño del modelo sobre el conjunto de prueba (test).

6. Predicción
Se realiza una predicción para un ejemplo individual y se clasifica como Benigno o Maligno.


Resultados

El modelo alcanza una precisión aproximada entre 95% y 97% en el conjunto de test, lo cual indica una buena capacidad de generalización.

Ejemplo de salida en consola:

Precisión en test: 0.9649
Predicción: Tumor MALIGNO

Visualización del Entrenamiento

Durante el entrenamiento se grafica la evolución de:
Accuracy
Validation Accuracy
La gráfica se guarda automáticamente en el archivo:
entrenamiento.png