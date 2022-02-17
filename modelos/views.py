from django.shortcuts import render
from django.http import HttpResponse
from pathlib import Path

import json

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt


# ==============================================================================
# importamos los modelos
# ==============================================================================


BASE_DIR = Path(__file__).resolve().parent.parent

modelos = {
    "e_0_1k": pickle.load(open(str(BASE_DIR) + '/modelos/models/e_0_1k.sav', 'rb')),
    "e_1k_9k": pickle.load(open(str(BASE_DIR) + '/modelos/models/e_1k_9k.sav', 'rb')),
    "e_9k+": pickle.load(open(str(BASE_DIR) + '/modelos/models/e_9k+.sav', 'rb')),

    "m_0_1k": tf.keras.models.load_model(str(BASE_DIR) + '/modelos/models/m_0_1k.h5'),
    "m_1k_9k": tf.keras.models.load_model(str(BASE_DIR) + '/modelos/models/m_1k_9k.h5'),

    "lr_1k_9k": pickle.load(open(str(BASE_DIR) + '/modelos/models/lr_1k_9k.sav', 'rb')),
    "lr_9k+": pickle.load(open(str(BASE_DIR) + '/modelos/models/lr_9k+.sav', 'rb')),
}


# ==============================================================================
# Escaladores de delitos en uriel
# ==============================================================================

def obtenerEscalador(pib):
    if pib < 1000:
        return modelos['e_0_1k']
    elif pib > 9000:
        return modelos['e_9k+']
    else:
        return modelos['e_1k_9k']


def escalar_uriel_inv(json_data, x):
    min_max_scaler = obtenerEscalador(json_data['pib'])
    uriel_maximo = min_max_scaler.data_max_[6]
    uriel_minimo = min_max_scaler.data_min_[6]
    return x * (uriel_maximo - uriel_minimo) + uriel_minimo


# ==============================================================================
# procesador de entradas
# ==============================================================================


def procesar_entrada(json_data):
    x = [[
        json_data['pib'],
        json_data['participaciones'],
        json_data['regalias'],
        json_data['censo_electoral'],
        json_data['censo_habitacional'],
        json_data['homicidios'],
        0
    ]]
    x = obtenerEscalador(json_data['pib']).transform(x)
    x = np.array([x[0][0:6]])

    return np.reshape(x, (x.shape[0], 1, x.shape[1]))


# ==============================================================================
# Obtener modelos
# ==============================================================================

def obtenerRegresionLineal(json_data):
    if json_data['pib'] < 9000:
        return modelos['lr_1k_9k']
    else:
        print('lr_9k+')
        return modelos['lr_9k+']


def obtenerRed(json_data):
    if json_data['pib'] < 1000:
        return modelos['m_0_1k']
    else:
        return modelos['m_1k_9k']

# ==============================================================================
# Predictor de delitos en uriel red
# ==============================================================================


def red(request):
    if request.method == 'POST':
        json_data = json.loads(request.body)
        x = procesar_entrada(json_data)
        prediccion = obtenerRed(json_data).predict(x)[0][0]

        return HttpResponse(escalar_uriel_inv(json_data, prediccion))

    response = HttpResponse()
    response.status_code = 405
    return response


# ==============================================================================
# Predictor de delitos en uriel regresion
# ==============================================================================

def regresion(request):
    if request.method == 'POST':
        json_data = json.loads(request.body)
        x = procesar_entrada(json_data)
        prediccion = obtenerRegresionLineal(json_data).predict(x[0])

        return HttpResponse(escalar_uriel_inv(json_data, prediccion))

    response = HttpResponse()
    response.status_code = 405
    return response
