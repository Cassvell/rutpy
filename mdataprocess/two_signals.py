import matplotlib.pyplot as plt 
import numpy as np

x = np.linspace(1,21,3000)
y1 = np.cos(x)
y2 = np.cos(x + np.pi)  # Opposite phase (180° phase shift)

plt.plot(x,y1, color='red')
plt.plot(x,y2, color='blue')
plt.show()