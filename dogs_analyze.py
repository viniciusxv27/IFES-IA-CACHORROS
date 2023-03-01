import numpy as np
import tensorflow as tf

# salve a imagem que deseja verificar como "cachorro.jpg"

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
print('Este é um(a): ', prediction)