import tkinter as tk

gui = tk.Tk()

gui.title('Mi primera interfaz')
gui.geometry('350x200')

lbl = tk.Label(gui, text='Escribe tu mensaje')

lbl.grid(column=0, row=0)

txt = tk.Entry(gui, width= 15)
txt.grid(column=1, row=0)

lblRes = tk.Label(gui)

lblRes.grid(column=0, row=1)

def mostrar():
    mensaje = "Escribiste: " + txt.get()
    lblRes.config(text= mensaje) 
 
btn = tk.Button(gui, text = 'Ejecutar', command= mostrar)
btn.grid(column=2, row=0)

    
gui.mainloop()