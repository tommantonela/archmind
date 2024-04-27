# Componente-Estadística

* Status: accepted
* Date: 2023-11-07

## Context and Problem Statement

Necesitamos crear un módulo estadísticas que nos permita proporcionar información sobre el estado de los pedidos y la situación en tiempo real de los camiones.

## Decision Drivers

* RF05: Módulo de estadísticas.

## Considered Options

* 0011-1-Componente Apache Common Maths
* 0011-2-Implementar una clase

## Decision Outcome

Chosen option: "0011-1-Componente Apache Common Maths", because Es más sencillo de diseñar y con ello resolvemos el problema.

### Positive Consequences

* Fácil de diseñar.
* Solución más simple.

### Negative Consequences

* Es necesario intalarse la librería y Apache Maven.

## Pros and Cons of the Options

### 0011-1-Componente Apache Common Maths

Apache Common Maths es una librería que permite realizar estadísticas.

* Good, because Compatible con Java.
* Good, because Autocontenida.
* Good, because Ligera.
* Bad, because Requiere instalarse tambien Apache Maven.

### 0011-2-Implementar una clase

Crear una clase que implemente todas las operaciones de las estadisticas.

* Good, because Podemos seleccionar que consultas se pueden realizar.
* Good, because No sería necesario instalar ninguna librería externa.
* Bad, because Más complejo de diseñar e implementar.
