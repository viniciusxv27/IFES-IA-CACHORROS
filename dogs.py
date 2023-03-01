import tensorflow as tf

tf.config.run_functions_eagerly(True)

# Diretório de treinamento e teste
train_dir = "cachorros/treino/"
test_dir = "cachorros/teste/"

# Cria geradores de imagem para dados de treinamento e teste
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

print(train_datagen)
print(test_datagen)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=['poodle', 'pug']
        )

validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=['poodle', 'pug']
        )

tf.keras.backend.clear_session()

# Define o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Treina o modelo
history = model.fit(validation_generator, epochs=4)

# Avalia a acurácia do modelo com dados de teste
test_loss, test_acc = model.evaluate(validation_generator, verbose=1)

assetividade = "{:.0f}%".format(test_acc*100)

print('Assertividade:', assetividade)

# SALVAR O ARQUIVO (OPCIONAL)
# model.save("modelo_cachorros.h5")


'''

2º PASSO 

CARREGAR E TESTAR


import numpy as np

# carrega a imagem
img = tf.keras.preprocessing.image.load_img('cachorro.jpg', target_size=(224, 224))

# converte a imagem em um array numpy
img_array = tf.keras.preprocessing.image.img_to_array(img)

# adiciona uma dimensão extra para representar o lote de imagens (batch)
img_array = np.expand_dims(img_array, axis=0)

# normaliza os valores dos pixels para estarem entre 0 e 1
img_array /= 255.0

# faz a previsão com o modelo
model = tf.keras.models.load_model('modelo_cachorro.h5')
prediction = model.predict(img_array)

# exibe a previsão
print('Este é um(a): ',prediction)'''
