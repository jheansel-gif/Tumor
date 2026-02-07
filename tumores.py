import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Cargar dataset
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = benigno, 1 = maligno

# 2. Normalizar datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Definir modelo (red neuronal)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binaria
])

# 5. Compilar modelo (backprop + optimización)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. Entrenar modelo
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

# 7. Evaluar en test
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión en test: {acc:.4f}")

# 8. Graficar entrenamiento
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Entrenamiento del modelo')
plt.show()

plt.savefig("entrenamiento.png", dpi=150, bbox_inches="tight")
print("Gráfica guardada como entrenamiento.png")

# 9. Probar predicción con un ejemplo
sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)[0][0]

if pred > 0.5:
    print("Predicción: Tumor MALIGNO")
else:
    print("Predicción: Tumor BENIGNO")
