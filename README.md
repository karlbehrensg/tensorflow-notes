<div align="center">
  <h1>Notas de TensorFlow 2</h1>
</div>

# Contenido
- [Construir y entrenar un modelo de red neuronal usando TensorFlow 2](#Construir-y-entrenar-un-modelo-de-red-neuronal-usando-TensorFlow-2)
    - [Instalar TensorFlow 2 y Matplotlib](#Instalar-TensorFlow-2-y-Matplotlib)
    - [Construir, compilar y entrenar modelos de ML usando TensorFlow](#Construir,-compilar-y-entrenar-modelos-de-ML-usando-TensorFlow)

# Construir y entrenar un modelo de red neuronal usando TensorFlow 2

## Instalar TensorFlow 2 y Matplotlib

Para instalar la libreria de TensorFlow y Matplotlib en nuestra consola (dentro del entorno virtual) ejecutamos:

```
pip install tensorflow matplotlib
```

## Construir, compilar y entrenar modelos de ML usando TensorFlow

Para construir un modelo de ML para una regresión primero crearemos nuestros datos.

```python
X = -1, 0, 1, 2, 3, 4
Y = -3, -1, 1, 3, 5, 7
```

La solucion algebraica para los datos anteriores seria `y=2x-1`.

Ahora para crear un modelo que resuelva estos datos creamos una red simple y compilamos.

```python
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
```

La red debe recibir un numpy array para entrerar.

```python
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
```

Ahora entrenamos

```python
model.fit(xs, ys, epochs=500)
```