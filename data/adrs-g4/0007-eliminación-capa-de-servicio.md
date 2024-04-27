# Eliminación-Capa-De-Servicio

* Status: accepted
* Date: 2023-10-31

## Context and Problem Statement

No es necesario tener separadas las clases Incidencias y Estadísticas en una capa a parte puesto que forman parte de la lógica de negocio.

## Decision Drivers

* RF05: Módulo de estadísticas.
* RF06:Módulo de incidencias.

## Considered Options

* 0007-1-Introducir las clases en la capa de lógica de negocio

## Decision Outcome

Chosen option: "0007-1-Introducir las clases en la capa de lógica de negocio", because Con esta decisión corregimos el error inicial y obtenemos un diseño más cercano a la descripción del problema.

### Positive Consequences

* Eliminación capa inecesaria.
* Corregimos un error de diseño.

### Negative Consequences

* Más clases en la capa de lógica de negocio.

## Pros and Cons of the Options

### 0007-1-Introducir las clases en la capa de lógica de negocio

Como las funcionalidades de las dos clases tienen que ver con la lógica del negocio, decidimos introducir las clases en la capa correspondiente.

* Good, because Eliminamos capa inecesaria.
* Good, because Corregimos una decisión de diseño errónea.
* Bad, because Añade más clases a la capa de lógica de negocio.
