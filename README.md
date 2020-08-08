<div align="center">
  <h1>Notas de TensorFlow 2</h1>
</div>

# Contenido
- [Construir y entrenar un modelo de red neuronal usando TensorFlow 2](#Construir-y-entrenar-un-modelo-de-red-neuronal-usando-TensorFlow-2)
    - [Instalar TensorFlow 2 y Matplotlib](#Instalar-TensorFlow-2-y-Matplotlib)
    - [Construir, compilar y entrenar modelos de ML usando TensorFlow](#Construir-compilar-y-entrenar-modelos-de-ML-usando-TensorFlow)
    - [Predecir resultados](#Predecir-resultados)
    - [Construir modelo secuencial con multiples capas](#Construir-modelo-secuencial-con-multiples-capas)
    - [Construir modelo de clasificacion binaria](#Construir-modelo-de-clasificacion-binaria)
    - [Construir modelo de clasificacion multi clase](#Construir-modelo-de-clasificacion-multi-clase)
    - [Graficar perdida y certeza del modelo durante entrenamiento](#Graficar-perdida-y-certeza-del-modelo-durante-entrenamiento)
    - [Evitando el overfitting](#Evitando-el-overfitting)
    - [Modelos pre entrenados (transfer learning)](#Modelos-pre-entrenados-transfer-learning)

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

## Predecir resultados

Para predecir resultados tomamos nuestro modelo y ejecutamos el metodo `predict`.

```python
model.predict(X)
```

Esto nos devolvera el resultado.

## Construir modelo secuencial con multiples capas

El modelo secuencial se define con `Sequential` y la primera capa debe siempre llevar el `input_shape`.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=[1]),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(1),
])
```

## Construir modelo de clasificacion binaria

El hecho de generar un modelo para clasificacion binaria implica que solo tendremos 2 valores de salida, por lo que el metodo de activacion por excelencia es el sigmoide.

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
```

## Construir modelo de clasificacion multi clase

Para la clasificacion de multiclase cambiamos la activacion de la ultima capa por `softmax`.

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

## Graficar perdida y certeza del modelo durante entrenamiento

Durante el entramiento podemos conservar la perdida y certeza de nuetro modelo, para ello lo asociamos a una variable.

```python
history = model.fit(xs, ys, epochs=500)
```

Separamos los valores de `accuracy` y `loss` en variables, y para tener la cantidad de `epochs` existente simplemente vemos el largo de las variables.

```python
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )
```

<div> 
  <img src="accuracy.png" width="350">
  <br>
  <img src="loss.png" width="350">
</div>

## Evitando el overfitting

### Augumentation

Esta tecnica tiene principal uso en las imagenes, y consiste en que una misma imagen tome varias formas al pasar por el entrenamiento. Por lo general se utilisa `ImageDataGenerator` para esta técnica.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
```

Tambien se puede usar en una misma capa.

```python
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])
```

Y simplemente la referenciamos en el modelo al construirlo.

```python
model = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # Rest of your model
])
```

### Dropout

Esta es una capa en el modelo que tiene como objetivo no activar ciertos nodos al momento de la optimización.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5), # Capa de Dropout
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Modelos pre entrenados (transfer learning)

Existe la posibilidad de tranferir el conocimiento de un modelo pre-entrenado a nuestro modelo. Primero debemos cargar nuestro modelo, en este caso InceptionV3 desde TensorFlow, y los pesos los cargaremos desde un archivo.

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights.h5'
```

Definimos sus parametros

```python
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, # Si deseamos conectar otro modelo lo dejamos en False
                                weights = None)
```

Cargamos los pesos

```python
pre_trained_model.load_weights(local_weights_file)
```

Y definiremos que todas las capas no son entrenable

```python
for layer in pre_trained_model.layers:
    layer.trainable = False
```

Por ultimo veremos la forma de la salida de los datos y referenciamos a la ultima capa.

```python
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
```

Ahora creamos nuestro modelo y lo acoplamos con el modelo pre-entrenado.

```python
from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(1, activation='sigmoid')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
```

## Verificar la forma de los inputs

Los input de datos siempre seran de la dimension de nuestro X_train sin contar la coordenada batch.

```python
>>>train_set
<PrefetchDataset shapes: ((None, None, 1), (None, None, 1)), types: (tf.float64, tf.float64)>
>>>X_train.shape
(3000,)
```

En este caso el input necesario seria `[None, 1]`.

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                        strides=1, padding="causal",
                        activation="relu",
                        input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])
```