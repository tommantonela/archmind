# Selección-De-La-Ruta-Óptima

* Status: accepted
* Date: 2023-10-26

## Context and Problem Statement

El sistema cuenta con dos algoritmos de optimización para realizar el reparto según la demora y necesitamos que ese tiempo sea el mínimo.

## Decision Drivers

* RF04: Módulo reparto y rutas.

## Considered Options

* 0003-1-Patrón Strategy

## Decision Outcome

Chosen option: "0003-1-Patrón Strategy", because Permite seleccionar el algoritmo más óptimo de una forma sencilla y eficiente mediante una interfaz de la cual heredan los distintos algoritmos.

## Pros and Cons of the Options

### 0003-1-Patrón Strategy

Patrón de diseño que permite definir algoritmos colocando cada uno en una clase diferente pudiendo alternar de esta forma el algoritmo que se desee usar.

* Good, because Te permite implementar varios algoritmos.
* Good, because Se utiliza el algoritmo que mejor convenga al sistema.
* Good, because Fácil de escalar en case de que se quieran añadir más algoritmos.
* Bad, because Cada algoritmo es completamente independiente por lo que en caso de que uno necesitara información del otro no podrían conseguirla.
