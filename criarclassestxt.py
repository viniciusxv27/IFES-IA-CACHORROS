import requests
import os

url = 'https://dog.ceo/api/breeds/list/all'
racas_json = requests.get(url).json()
racas_json = racas_json['message']
lista2 = []
#separa as imagens por nome
for key, value in racas_json.items():
    
    if (value):
        for name_value in value:
            #pega as imagens de sub raças da API de imagem
            lista2.append(f"{key}-{name_value}")
            
    else:
        # pega imagem de raça
        lista2.append(key)
print(len(lista2))

with open('classes.txt', 'w') as arquivo:
    for linha in lista2:
        arquivo.write(linha + '\n')