import requests
import os


def main():
    qtd = 10

    url = 'https://dog.ceo/api/breeds/list/all'
    racas_json = requests.get(url).json()
    racas_json = racas_json['message']

    for key, value in racas_json.items():

        if (value):
            for name_value in value:
                url = 'https://dog.ceo/api/breed/' + key + '/' + \
                    name_value + '/images/random/' + str(qtd)
                img = requests.get(url).json()

                download(key, name_value, img)
        else:
            url = 'https://dog.ceo/api/breed/' + \
                key + '/images/random/' + str(qtd)
            img = requests.get(url).json()

            download(key, '', img)

def download(name, subname, req):
    url = req['message']
    if (req['status'] == 'success'):

        if (subname != ''):
            subname = '-' + subname

        if (not os.path.exists(name + subname)):
            os.makedirs(name + subname)

        for link in url:
            img = requests.get(link)

            name_file = 'https://images.dog.ceo/breeds/' + name + subname + '/'
            name_file = link[len(name_file)::]

            with open(name + subname + '/' + name_file, 'wb') as file:
                file.write(img.content)

main()
