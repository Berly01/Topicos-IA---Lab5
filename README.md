# Optimizacion en MLP - Adam y RMS-Prop

Laboratorio 5

## Ejecucion

Compilacion
```bash
  nvcc kernel.cu -o kernel
```
Ejecucion
```bash
  ./kernel
```
## Optimizadores
La cabezera MLP.hpp, al final del metodo update_mini_batch, se realiza la actualizacion de pesos y biases. Dependiendo del metodo de optimizacion seleccionado se aplica en la actualizacion. 
