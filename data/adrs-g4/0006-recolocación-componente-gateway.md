# Recolocación-Componente-Gateway

* Status: accepted
* Date: 2023-10-31

## Context and Problem Statement

El componente gateway no debería situarse en la capa de presentación puesto que no es algo que el usuario pueda controlar.

## Decision Drivers

* RF08: Componente Gateway.

## Considered Options

* 0006-1-Introducirlo en la capa de lógica de negocio
* 0006-2-Introducirlo entre la capa de presentación y la capa de lógica de negocio

## Decision Outcome

Chosen option: "0006-2-Introducirlo entre la capa de presentación y la capa de lógica de negocio", because Creemos que de esta forma el UML queda más limpio y claro.

### Positive Consequences

* UML más sencillo.

### Negative Consequences

* No se puede saber viendo el UML que clases utilizan el componente.

## Pros and Cons of the Options

### 0006-1-Introducirlo en la capa de lógica de negocio

Puesto que interactúa con las clases de la capa de lógica de negocio podría situarse dentro de ella.

* Good, because Estaría más claro las clases y paquetes que utilizarían este componente.
* Bad, because Añade más relaciones al UML.

### 0006-2-Introducirlo entre la capa de presentación y la capa de lógica de negocio

Consiste en situar el componente gateway entre las dos primeras capas del diseño.

* Good, because Es más claro visualmente.
* Bad, because No se sabe con exactitud las clases que utilizan el componente.
