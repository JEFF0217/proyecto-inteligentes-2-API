import os
from Prediccion import Prediccion
import base64
import pathlib
from PIL import Image
import cv2
from io import BytesIO
import numpy as np


clases = ["beachballs", "billiardball", "bowlingball", "football", "golfball",
          "paintballs", "pokemonballs", "soccerball", "tennisball", "volleyball"]


def readb64(base64_string):
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2GRAY)


def readb642(image):

    decoded_data = base64.b64decode(image)
    np_data = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    cv2.imshow("test", img)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def predecir(imagen):

    ancho = 128
    alto = 128
    # print(files)

    imgageDecoded = readb642(imagen)

    pred = Prediccion('modelos/modelo_1v5.h5', ancho, alto)

    clase = pred.predecir(imgageDecoded)
    print(clase)

    probabilidades = formatPorcentajes(clase["prediccion"])
    resultados = {"clase": clases[clase["clase"]],
                  "probabilidades": probabilidades}

    return resultados


def formatPorcentajes(predicciones):
    resultado = "Probabilidades: "
    index = 0

    for prediccion in predicciones:
        resultado += "\n" + clases[index] + ": " + "{:.4%}".format(prediccion)
        index += 1
    return (resultado)
