# -*- coding: utf-8 -*-
"""
Plots
Created on Sun Oct 24 21:16:55 2021

@author: Paulo De Los Santos
"""
import funciones as fun
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('radar_pulsado.mat')

# Inciso 1

## Variables Extraidas

Np = mat['Np'][0]
Ntargets = mat['Ntargets'][0]
PRF = mat['PRF'][0]
PRI = mat['PRI'][0]
SNR = mat['SNR'][0]
T = mat['T'][0]
T_out = mat['T_out'][0]
W = mat['W'][0]
fc = mat['fc'][0]
fs = mat['fs'][0]
ranges = mat['ranges'][0]
rmax = mat['rmax'][0]
rmin = mat['rmin'][0]
rua = mat['rua'][0]
s = mat['s']
vels = mat['vels'][0]
vua = mat['vua'][0]
y = mat['y']


plt.figure()
plt.plot(np.real(s), label = 'Parte real de la señal Chirp')
plt.grid(True)
plt.legend(loc = 'best')
plt.savefig('Images/SenalChirpReal' +".png")
plt.show()

plt.figure()
plt.plot(np.imag(s), label = 'Parte imaginaria de la señal Chirp')
plt.grid(True)
plt.legend(loc = 'best')
plt.savefig('Images/SenalChirpImaginaria' +".png")
plt.show()


# Inciso 2 - Senal recibida en un canal.

## Senal recibida
y_canal = y[:,0]
rango = fun.rango((1.8e-4),fs, T[0])
y_canal = y_canal[:np.size(rango)]
## Senal recibida
y_canal = np.abs(y_canal)*np.abs(y_canal)*0.5

plt.figure()
plt.plot(rango,y_canal, label = 'Potencia Instantánea Recibida')
plt.grid(True)
plt.legend(loc = 'best')
plt.ylabel('Potencia [W]')
plt.xlabel('Rango [km]')
plt.savefig('Images/PotenciaRecibida' +".png")
plt.show()

## Senal recibida dB
y_canal = 10*np.log10(y_canal)

plt.figure()
plt.plot(rango,y_canal, label = 'Potencia Instantánea Recibida')
plt.grid(True)
plt.legend(loc = 'best')
plt.ylabel('Potencia [dB]')
plt.xlabel('Rango [km]')
plt.savefig('Images/PotenciaRecibidadB' +".png")
plt.show()

# Inciso 3 - Filtro Adaptado
Chirp_referencia = fun.ref_chirp(1, W, T[0], fs[0])

s = np.array(Chirp_referencia)
h = np.conjugate(Chirp_referencia)
h = h[::-1]

# Inciso 4 - Comparacion

y_canal = y[:,0]

Y_adapt = fun.filtroAdaptado(y,h)
Y_ad = Y_adapt[:,0]

rango = fun.rango((1.8e-4),fs, T[0])

y_canal = y_canal[:np.size(rango)]
Y_ad = Y_ad[:np.size(rango)]


plt.figure()
plt.plot(rango,0.5*np.abs(y_canal)**2, label = 'Sin Filtro Adaptado')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia')
plt.savefig('Images/SinFiltro' +".png")
plt.show()


plt.figure()
plt.plot(rango,0.5*np.abs(Y_ad)**2, label = 'Con Filtro Adaptado')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia')
plt.savefig('Images/ConFiltro' +".png")
plt.show()


# ---------------------------------------------------------------------

Chirp_referencia = fun.ref_chirp(1, W, T[0], fs[0])
rango = fun.rango((1.8e-4),fs, T[0])
s = np.array(Chirp_referencia)
h = np.conjugate(Chirp_referencia)
h = h[::-1]

Ym = fun.cancelador2pulsosMAX(y,h)
Ym = Ym[:np.size(rango),:]

Ym3 = fun.cancelador3pulsosMAX(y,h)
Ym3 = Ym3[:np.size(rango),:]

Yf = fun.STI2(y,h)
Yf = Yf[:np.size(rango),:]

YCFARm = fun.CFAR(1e-6, 20, SNR[0], Ym, Guarda = 3)
YCFARm3 = fun.CFAR(1e-6,30, SNR[0], Ym3, Guarda = 3)
YCFARf = fun.CFAR(1e-6, 20, SNR[0], Yf, Guarda = 3)

ym_det = fun.Detector(Ym, YCFARm, integBinaria = True, cant = 1)
ym3_det = fun.Detector(Ym3, YCFARm3, integBinaria = True, cant = 1)
yf_det = fun.Detector(Yf, YCFARf, integBinaria = True, cant = 5)

# ---------------------------------------------------------------------

# Inciso 5 - Filtro MTI

plt.figure()
plt.plot(rango,0.5*np.abs(Ym[:,0])**2, label = 'Filtro Simple Cancelador')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia')
plt.savefig('Images/SimpleCancelador' +".png")
plt.show()


plt.figure()
plt.plot(rango,0.5*np.abs(Ym3[:,0])**2, label = 'Filtro doble Cancelador')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia')
plt.savefig('Images/DobleCancelador' +".png")
plt.show()

plt.figure()
plt.plot(rango,0.5*np.abs(Yf[:,0])**2, label = 'Filtro STI')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia')
plt.savefig('Images/STI' +".png")
plt.show()

# Inciso 6 - Umbral CFAR MTI

plt.figure()
plt.plot(rango, 0.5*np.abs(Ym[:,1])**2, label = 'Salida Cancelador de dos pulsos')
plt.plot(rango, 0.5*np.abs(YCFARm[:,1]), label = 'Umbral de detección CFAR')
plt.grid(True)
plt.ylim(0,1.5e4)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia')
plt.savefig('Images/SalidaUmbral2pulsos' +".png")
plt.show()


plt.figure()
plt.plot(rango, 0.5*np.abs(Ym[:,1])**2, label = 'Salida Cancelador de dos pulsos')
plt.plot(rango, 0.5*np.abs(YCFARm[:,1]), label = 'Umbral de detección CFAR')
plt.grid(True)
plt.ylim(0,2e4)
plt.xlim(4,6)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia [W]')
plt.savefig('Images/SalidaUmbral2pulsos5km' +".png")
plt.show()

# Inciso 7 - Deteccion MTI

plt.figure()
plt.plot(rango, ym_det[:,0], label = 'Integrador Binario - Cancelador de dos pulsos')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.savefig('Images/IntegradorBinario2Pulsos' +".png")
plt.show()


plt.figure()
plt.plot(rango, ym3_det[:,0], label = 'Integrador Binario - Cancelador de tres pulsos')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.savefig('Images/IntegradorBinario3Pulsos' +".png")
plt.show()


plt.figure()
plt.plot(rango, 0.5*np.abs(Yf[:,1])**2, label = 'Salida Filtro STI')
plt.plot(rango, 0.5*np.abs(YCFARf[:,1]), label = 'Umbral de detección CFAR')
plt.grid(True)
#plt.ylim(0,2.5e4)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia [W]')

# Inciso 8 - Deteccion MTI


plt.figure()
plt.plot(rango, 0.5*np.abs(Yf[:,0])**2, label = 'Salida Filtro STI')
plt.plot(rango, 0.5*np.abs(YCFARf[:,0]), label = 'Umbral de detección CFAR')
plt.grid(True)
plt.ylim(0,0.8e5)
plt.xlim(24,26)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.ylabel('Potencia [W]')

plt.figure()
plt.plot(rango, yf_det[:,0], label = 'Integrador Binario - STI')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Rango [km]')
plt.savefig('Images/IntegradorBinarioSTI' +".png")
plt.show()


fs = 14
#plt.contourf(np.arange(1,9), rango, output, cmap="gray", levels=1)
plt.contourf(np.arange(1,9), rango, np.abs(Ym3), cmap="gray", levels=50)
cbar = plt.colorbar()
# cbar.set_label(r"$|h_0|^2$")
cbar.ax.tick_params(labelsize = fs)
# plt.contour(px, py, h, colors="black")
plt.ylabel("Rango [km]")
plt.xlabel("Número de Salida del Filtro")
plt.tight_layout()
plt.savefig('Images/DeteccionesConFiltro3Pulsos' + ".png")
plt.show()