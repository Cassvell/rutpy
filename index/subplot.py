import matplotlib.pyplot as plt
from PIL import Image
import os

# Configuración
imagen1_path = '/home/isaac/rutidl/output/dispersion_general_dst.eps'  # Reemplaza con tu ruta
imagen2_path = '/home/isaac/rutidl/output/dispersion_general_dst_lm.eps'  # Reemplaza con tu ruta
output_path = '/home/isaac/rutidl/output/panel_combinado.eps'  # Archivo de salida

# 1. Leer las imágenes EPS y convertirlas a formato manejable (PNG temporal)
def eps_to_temp_png(eps_path):
    img = Image.open(eps_path)
    temp_path = eps_path.replace('.eps', '_temp.png')
    img.save(temp_path, 'PNG', resolution=300)  # Alta resolución
    return temp_path

# Crear archivos temporales
temp1 = eps_to_temp_png(imagen1_path)
temp2 = eps_to_temp_png(imagen2_path)

# 2. Crear la figura con dos paneles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

# Cargar imágenes temporales
img1 = plt.imread(temp1)
img2 = plt.imread(temp2)

# Mostrar imágenes en los paneles
ax1.imshow(img1)
ax1.axis('off')  # Oculta ejes
ax1.set_title('Panel A', pad=10)

ax2.imshow(img2)
ax2.axis('off')
ax2.set_title('Panel B', pad=10)

# Ajustes de diseño
plt.tight_layout(pad=2.0)

# 3. Guardar como EPS (vectorial) o PNG
fig.savefig(output_path, format='eps', bbox_inches='tight', dpi=300)
# Alternativa para PNG: fig.savefig(output_path.replace('.eps','.png'), ...)

# Limpiar archivos temporales
os.remove(temp1)
os.remove(temp2)

print(f"¡Paneles combinados guardados en {output_path}!")