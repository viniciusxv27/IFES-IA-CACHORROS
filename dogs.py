#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 02:32:37 2023

@author: viniciuscostamarcelos
"""

import requests
import json
import pandas as pd

request = requests.get('https://dog.ceo/api/breeds/list/all')
#Faz uma requisiçao a API de cachorros

json_response = request.json()
#Retorna o código JSON requisitado

lista_json = json.dumps(json_response).replace("message", "Raças")
#Formata o código JSON retornado e muda o message para Raças

table = pd.read_json(lista_json)
#Le o código JSON e transforma em tabela
    
print(table)
#Imprime a tabela na tela
