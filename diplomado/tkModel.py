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


# Función para realizar predicción cuando se presione el boton
def clicked():

    year = float(txtYear.get())
    mileage = float(txtMileage.get())
    fuelType = float(txtFuelType.get())
    tax = float(txtTax.get())
    mpg = float(txtMpg.get())
    engineSize = float(txtEngineSize.get())
    
    features = np.array([year, mileage, fuelType, tax, mpg, engineSize])
    prediction = modelo.predict(features.reshape(1, -1))

    lblPrediccion.configure(text = str(round(prediction[0], 2)))
 
# Definimos el boton
btnCalcular = tkinter.Button(root, text = "Calcular" , fg = "red", command=clicked)
btnCalcular.grid(column = 4, row = 0)

# Ejecuta Tkinter
root.mainloop()