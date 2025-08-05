# -*- coding: utf-8 -*-
"""
Created on Wed May 15 23:46:42 2024

@author: joel_
"""

# Importamos las bibliotecas
import tkinter
import pickle
import numpy as np
 
# Creamos la ventana raíz
root = tkinter.Tk()
root.title("Sistema para predicción de precios")
# Configura el tamaño (widthxheight)
root.geometry('500x200')
 
# Cargamos el modelo
modelo = pickle.load(open('modelo.pkl', 'rb'))


# Añadimos una etiqueta para colocar el resultado
lblResultado = tkinter.Label(root, text = "El costo del auto es: ")
lblResultado.grid(column = 1, row = 7)
lblPrediccion = tkinter.Label(root, text = "0")
lblPrediccion.grid(column = 2, row = 7)

# Colocamos campos de entrada para cada característica
lblYear = tkinter.Label(root, text = "Año: ")
lblYear.grid(column = 1, row = 0)
txtYear = tkinter.Entry(root, width = 10)
txtYear.grid(column = 2, row = 0)

lblMileage = tkinter.Label(root, text = "Kilometraje: ")
lblMileage.grid(column = 1, row = 1)
txtMileage = tkinter.Entry(root, width = 10)
txtMileage.grid(column = 2, row = 1)

lblFuelType = tkinter.Label(root, text = "Combustible: ")
lblFuelType.grid(column = 1, row = 2)
txtFuelType = tkinter.Entry(root, width = 10)
txtFuelType.grid(column = 2, row = 2)

lblTax = tkinter.Label(root, text = "Impuesto: ")
lblTax.grid(column = 1, row = 3)
txtTax = tkinter.Entry(root, width = 10)
txtTax.grid(column = 2, row = 3)

lblMpg = tkinter.Label(root, text = "Mpg: ")
lblMpg.grid(column = 1, row = 4)
txtMpg = tkinter.Entry(root, width = 10)
txtMpg.grid(column = 2, row = 4)

lblEngineSize = tkinter.Label(root, text = "Máquina: ")
lblEngineSize.grid(column = 1, row = 5)
txtEngineSize = tkinter.Entry(root, width = 10)
txtEngineSize.grid(column = 2, row = 5)

#Funciones para preprocesar y postprocesar

def estandarizar(x):
  x_mean = np.array([2.01662658e+03, 2.44982880e+04, 1.61098738e-01, 7.71714922e+01,
       5.86742390e+01, 1.36391982e+00])
  x_std = np.array([1.90009739e+00, 1.79573016e+04, 3.67622000e-01, 6.72932142e+01,
       1.09916475e+01, 2.81298431e-01])
  x_pre = (x - x_mean)/x_std
  return x_pre

def estandarInverso(y_estandarizado):
  y_mean = 1.43277506e+04
  y_std = 4.64418736e+03
  y = (y_estandarizado * y_std) + y_mean
  return y

# Función para realizar predicción cuando se presione el boton
def clicked():

    year = float(txtYear.get())
    mileage = float(txtMileage.get())
    fuelType = float(txtFuelType.get())
    tax = float(txtTax.get())
    mpg = float(txtMpg.get())
    engineSize = float(txtEngineSize.get())
    
    features = np.array([year, mileage, fuelType, tax, mpg, engineSize])
    featureStd = estandarizar(features)
    
    predictionStd = modelo.predict(featureStd.reshape(1, -1))
    
    prediction = estandarInverso(predictionStd)
    #modelo.predict(features.reshape(1, -1))

    lblPrediccion.configure(text = str(round(prediction[0], 2)))
 
# Definimos el boton
btnCalcular = tkinter.Button(root, text = "Calcular" , fg = "red", command=clicked)
btnCalcular.grid(column = 4, row = 0)

# Ejecuta Tkinter
root.mainloop()