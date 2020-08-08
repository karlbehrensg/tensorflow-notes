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
    - [Verificar la forma de los inputs](#Verificar-la-forma-de-los-inputs)
    - [Batching y Prefetching](#Batching-y-Prefetching)
    - [Callbacks](#Callbacks)
    - [Leer datos de multiples archivos](#Leer-datos-de-multiples-archivos)
    - [Leer distintos formatos](#Leer-distintos-formatos)
- [Clasificacion de imagenes](#Clasificacion-de-imagenes)
    - [Red neuronal Convulusional con Conv2D y capas pooling](#Red-neuronal-Convulusional-con-Conv2D-y-capas-pooling)
    - [ImageDataGenerator](#ImageDataGenerator)

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

Los input de datos siempre seran de la dimension de nuestro X_train sin contar la coordenada batch, mas la cantidad de canales de la capa, es por esto que para distintas capas cambia su `input_shape`.

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

## Batching y Prefetching

Podemos hacer que nuestros datos vayan ingresando a nuestro modelo con el metodo `batch`, pero para eso debemos transformar nuestros datos en tipo `Dataset` de TensorFlow.

El `prefetch` indica cuantos batch estara preparando cuando se estan ejecutando el entrenamiento.

```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
```

## Callbacks

Los callbacks se ejecutan durante cada epoca de entrenamiento, estos pueden detener el entrenamiento cuando un parametro de evaluacion alcanza un valor deseado, cuando no hay mejoras u otros objetivos.

Un ejemplo para detener el entrenamiento cuando __no hay mejoras__.

```python
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                    epochs=100, validation_data=(X_valid, y_valid), 
                    callbacks=[early_stopping_cb])
```

Guardar __checkpoint__ del modelo durante el entrenamiento.

```python
checkpoint_cb = keras.callbacks.ModelCheckpoint('my_keras_model.h5')

history = model.fit(X_train, y_train, 
                    epochs=10, validation_data=(X_valid, y_valid), 
                    callbacks=[checkpoint_cb])
```

O un callback __personalizado__.

```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.6):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```

## Leer datos de multiples archivos

Para esto primero debemos tener en una variable las direccion de estos archivos para luego insertarlo como un `Dataset.list_files`.

```python
train_filepaths = ['datasets/my_data_00.csv', 'datasets/my_data_01.csv', ...]

filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
```

Por defecto esto retornara un dataset barajado, en caso de querer lo contrario definimos `shuffle=False`.

## Leer distintos formatos

### JSON

Para leer datos con formato JSON.

```python
import json

with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
```

### CSV

```python
import csv

with open('bbc-text.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = ' ' + word + ' '
            sentence = sentence.replace(token, ' ')
            sentence = sentence.replace('  ', ' ')
        sentences.append(sentence)
```

# Clasificacion de imagenes

## Red neuronal Convulusional con Conv2D y capas pooling

```python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)
```

## ImageDataGenerator

Los directorios para las ImageDataGenerator deben separase en __Train__ y __Test__, y en cada uno deben estar separados por las __clases__.

```python
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/tmp/rps/"
training_datagen = ImageDataGenerator(
    rescale = 1./255,
	rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = "/tmp/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=126
)
```