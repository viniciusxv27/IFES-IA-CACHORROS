import requests
import os

def main():
    
    #pega os nomes da API de lista
    url = 'https://dog.ceo/api/breeds/list/all'
    racas_json = requests.get(url).json()
    racas_json = racas_json['message']
    print(racas_json)
    
    #separa as imagens por nome
    for key, value in racas_json.items():
        
        if (value):
            for name_value in value:
                #pega as imagens de sub raças da API de imagem
                url = 'https://dog.ceo/api/breed/' + key + '/' + name_value + '/images/random'
                img = requests.get(url).json()
                
                #baixa a imagem
                download(key, name_value, img)
        else:
            # pega imagem de raça
            url = 'https://dog.ceo/api/breed/' + key + '/images/random'
            img = requests.get(url).json()

            #baixa imgagem
            download(key, '', img)
            
#download das raças e sub raças                    
def download(name, subname, img):
        url = img['message']
        if (img['status'] == 'success'):
            img = requests.get(url)
            
            if (subname != ''):
                subname = '-' + subname
            
            if not os.path.exists(name + subname):
                os.makedirs(name + subname)
                print('Diretório criado com sucesso.')
                with open(name + subname  + '/' + name + '.jpg', 'wb') as file:
                     file.write(img.content)
                     print('Imagem salva com sucesso.')


main()