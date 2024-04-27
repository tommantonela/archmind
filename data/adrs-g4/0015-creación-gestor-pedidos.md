# Creación-Gestor-Pedidos

* Status: accepted
* Date: 2023-11-12

## Context and Problem Statement

Necesitamos que solo se cree una instancia de GestorPedidos.

## Decision Drivers

* RF03.2: Gestor pedidos.

## Considered Options

* 0015-1-Patrón Singleton

## Decision Outcome

Chosen option: "0015-1-Patrón Singleton", because Utilizando este patrón resolvemos el problema de tener solo una instancia de una forma sencilla.

## Pros and Cons of the Options

### 0015-1-Patrón Singleton

Es un patrón de diseño creacional que se utiliza para crear una sola instancia de una clase.

* Good, because Conseguimos tener una sola instancia de GestorPedidos.
* Good, because Solución sencilla.
* Bad, because Mayor complejidan en el diseño.
