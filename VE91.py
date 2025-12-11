import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, m_p

# Constantes físicas
k_B = k * 1e7  # constante de Boltzmann en erg/K
m_p_g = m_p * 1e3  # masa del protón en g

class VientoSolar:
    def __init__(self, tipo, n_p, v, T, B_1AU=8e-5):
        """
        Inicializa parámetros del viento solar
        """
        self.tipo = tipo
        self.n_1AU = n_p  # cm⁻³ (cc)
        self.v_1AU = v * 1e5  # cm/s
        self.T_1AU = T  # K
        self.B_1AU = B_1AU  # G
        self.R_sol = 6.96e10  # radio solar en cm
        self.AU = 1.496e13  # 1 AU en cm
        
        if tipo == 'lento':
            self.color = 'blue'
            self.gamma_profile = self._gamma_lento
        else:  # rápido
            self.color = 'red' 
            self.gamma_profile = self._gamma_rapido
    
    def _gamma_lento(self, r):
        r_AU = r / self.AU
        if r_AU <= 0.3:
            return 1.1
        elif r_AU <= 0.8:
            return 1.2
        elif r_AU <= 2.0:
            return 1.3
        else:
            return 1.4
    
    def _gamma_rapido(self, r):
        r_AU = r / self.AU
        if r_AU <= 0.2:
            return 1.05
        elif r_AU <= 0.6:
            return 1.15
        elif r_AU <= 1.5:
            return 1.25
        else:
            return 1.35
    
    def densidad(self, r):
        r_AU = r / self.AU
        return self.n_1AU * (1.0 / r_AU)**2 * (self.v_1AU / self.velocidad(r))
    
    def velocidad(self, r):
        r_AU = r / self.AU
        if self.tipo == 'lento':
            if r_AU < 0.3:
                return 200 * 1e5
            else:
                v_inf = 350 * 1e5
                r_acc = 0.5  # AU
                return v_inf * (1 - np.exp(-r_AU/r_acc))
        else:
            if r_AU < 0.2:
                return 300 * 1e5
            else:
                v_inf = 600 * 1e5
                r_acc = 0.4  # AU
                return v_inf * (1 - np.exp(-r_AU/r_acc))
    
    def temperatura(self, r):
        r_AU = r / self.AU
        gamma = self.gamma_profile(r)
        return self.T_1AU * (self.densidad(r) / self.n_1AU)**(gamma - 1)
    
    def campo_magnetico_radial(self, r):
        r_AU = r / self.AU
        return self.B_1AU * (1.0 / r_AU)**2
    
    def campo_magnetico_tangencial(self, r):
        r_AU = r / self.AU
        omega_sol = 2.87e-6
        v = self.velocidad(r)
        # B_θ ≈ - (Ω r sinθ / v) B_r
        return - (omega_sol * r * np.sin(np.pi/4) / v) * self.campo_magnetico_radial(r)
    
    def campo_magnetico_total(self, r):
        B_r = self.campo_magnetico_radial(r)
        B_theta = self.campo_magnetico_tangencial(r)
        return np.sqrt(B_r**2 + B_theta**2)
    
    def velocidad_sonido(self, r):
        T = self.temperatura(r)
        gamma = self.gamma_profile(r)
        return np.sqrt(gamma * k_B * T / m_p_g)
    
    def velocidad_alfven(self, r):
        B = self.campo_magnetico_total(r)
        n = self.densidad(r)
        return B / np.sqrt(4 * np.pi * n * m_p_g)
    
    def velocidad_magnetosonica(self, r):
        C_s = self.velocidad_sonido(r)
        C_A = self.velocidad_alfven(r)
        return np.sqrt(C_s**2 + C_A**2)
    
    def numero_mach_sonico(self, r):
        v = self.velocidad(r)
        C_s = self.velocidad_sonido(r)
        return v / C_s
    
    def numero_mach_alfven(self, r):
        v = self.velocidad(r)
        C_A = self.velocidad_alfven(r)
        return v / C_A
    
    def numero_mach_magnetosonico(self, r):
        v = self.velocidad(r)
        C_MS = self.velocidad_magnetosonica(r)
        return v / C_MS

def crear_graficas_individuales_5UA():
    """Crea cada gráfica en una ventana separada con eje radial de 0.1 a 5 UA"""
    
    # Crear objetos para ambos tipos de viento
    viento_lento = VientoSolar('lento', n_p=7, v=350, T=40000)      # 7 cm⁻³ (cc)
    viento_rapido = VientoSolar('rapido', n_p=2, v=600, T=120000)  # 2 cm⁻³ (cc)
    
    vientos = [viento_lento, viento_rapido]
    
    # Rango radial [0.1 UA, 5 UA] en cm
    AU = 1.496e13  # cm
    r_min = 0.1 * AU
    r_max = 5.0 * AU
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), 250)
    r_UA = r_values / AU  # En unidades astronómicas

    # Gráfica 1: Densidad
    plt.figure(1, figsize=(11, 6))
    for viento in vientos:
        n_p = [viento.densidad(r) for r in r_values]
        plt.plot(r_UA, n_p, color=viento.color, linewidth=2, label=f'Viento {viento.tipo}')
    
    # Línea vertical para 1 UA
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='1 UA')
    plt.text(1.05, plt.ylim()[1]*0.9, '1 UA', color='red', fontsize=11, fontweight='bold')
    
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(0.1, 5)  # Inicia en 0.1 UA
    plt.xlabel('r [UA]')
    plt.ylabel('Densidad de Protones [cm⁻³]')
    plt.title('Densidad')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfica 2: Velocidad
    plt.figure(2, figsize=(11, 6))
    for viento in vientos:
        v = [viento.velocidad(r) / 1e5 for r in r_values]
        plt.plot(r_UA, v, color=viento.color, linewidth=2, label=f'Viento {viento.tipo}')
    
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='1 UA')
    plt.text(1.05, plt.ylim()[1]*0.9, '1 UA', color='red', fontsize=11, fontweight='bold')
    
    plt.xscale('log')
    plt.xlim(0.1, 5)  # Inicia en 0.1 UA
    plt.xlabel('r [UA]')
    plt.ylabel('Velocidad [km/s]')
    plt.title('Velocidad')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfica 3: Temperatura
    plt.figure(3, figsize=(11, 6))
    for viento in vientos:
        T = [viento.temperatura(r) for r in r_values]
        plt.plot(r_UA, T, color=viento.color, linewidth=2, label=f'Viento {viento.tipo}')
    
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='1 UA')
    plt.text(1.05, plt.ylim()[1]*0.9, '1 UA', color='red', fontsize=11, fontweight='bold')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.1, 5)  # Inicia en 0.1 UA
    plt.xlabel('r [UA]')
    plt.ylabel('Temperatura [K]')
    plt.title('Temperatura')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfica 7: Números de Mach - Viento Lento
    plt.figure(3, figsize=(11, 6))
    M_s = [viento_lento.numero_mach_sonico(r) for r in r_values]
    M_A = [viento_lento.numero_mach_alfven(r) for r in r_values]
    M_MS = [viento_lento.numero_mach_magnetosonico(r) for r in r_values]
    
    plt.plot(r_UA, M_s, 'b-', linewidth=2, label='M_s (Lento)')
    plt.plot(r_UA, M_A, 'b--', linewidth=2, label='M_A (Lento)')
    plt.plot(r_UA, M_MS, 'b:', linewidth=2, label='M_MS (Lento)')
    
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='1 UA')
    plt.text(1.05, plt.ylim()[1]*0.9, '1 UA', color='red', fontsize=11, fontweight='bold')
    
    plt.xscale('log')
    plt.xlim(0.1, 5)  # Inicia en 0.1 UA
    plt.xlabel('r [UA]')
    plt.ylabel('Número de Mach')
    plt.title('Números de Mach - Viento Lento')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfica 8: Números de Mach - Viento Rápido
    plt.figure(4, figsize=(11, 6))
    M_s = [viento_rapido.numero_mach_sonico(r) for r in r_values]
    M_A = [viento_rapido.numero_mach_alfven(r) for r in r_values]
    M_MS = [viento_rapido.numero_mach_magnetosonico(r) for r in r_values]
    
    plt.plot(r_UA, M_s, 'r-', linewidth=2, label='M_s (Rápido)')
    plt.plot(r_UA, M_A, 'r--', linewidth=2, label='M_A (Rápido)')
    plt.plot(r_UA, M_MS, 'r:', linewidth=2, label='M_MS (Rápido)')
    
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='1 UA')
    plt.text(1.05, plt.ylim()[1]*0.9, '1 UA', color='red', fontsize=11, fontweight='bold')
    
    plt.xscale('log')
    plt.xlim(0.1, 5)  # Inicia en 0.1 UA
    plt.xlabel('r [UA]')
    plt.ylabel('Número de Mach')
    plt.title('Números de Mach - Viento Rápido')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfica 9: Componente Radial del Campo Magnético
    plt.figure(5, figsize=(11, 6))
    for viento in vientos:
        B_r = [viento.campo_magnetico_radial(r) * 1e5 for r in r_values]  # nT
        plt.plot(r_UA, B_r, color=viento.color, linewidth=2, label=f'B_r - {viento.tipo}')
    
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='1 UA')
    plt.text(1.05, plt.ylim()[1]*0.9, '1 UA', color='red', fontsize=11, fontweight='bold')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.1, 5)  # Inicia en 0.1 UA
    plt.xlabel('r [UA]')
    plt.ylabel('B_r [nT]')
    plt.title('Componente Radial del Campo Magnético')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfica 10: Componente Transversal del Campo Magnético
    plt.figure(6, figsize=(11, 6))
    for viento in vientos:
        B_theta = [viento.campo_magnetico_tangencial(r) * 1e5 for r in r_values]  # nT
        plt.plot(r_UA, np.abs(B_theta), color=viento.color, linewidth=2, label=f'|B_θ| - {viento.tipo}')
    
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='1 UA')
    plt.text(1.05, plt.ylim()[1]*0.9, '1 UA', color='red', fontsize=11, fontweight='bold')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.1, 5)  # Inicia en 0.1 UA
    plt.xlabel('r [UA]')
    plt.ylabel('|B_θ| [nT]')
    plt.title('Componente Transversal del Campo Magnético')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
   
# Ejecutar análisis
if __name__ == "__main__":
    print("Generando gráficas individuales...")
    print("Eje radial en Unidades Astronómicas (UA)")
    print("Rango: 0.1 UA a 5 UA (inicia en 0.1 UA)")
    print("Densidades en cm⁻³ (cc): Viento Lento=7 cc, Viento Rápido=2 cc")
    print("Línea vertical en 1 UA")

    
    crear_graficas_individuales_5UA()
    

