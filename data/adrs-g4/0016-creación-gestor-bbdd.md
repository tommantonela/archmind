# Creación-Gestor-BBDD

* Status: accepted
* Date: 2023-11-12

## Context and Problem Statement

Necesitamos que solo se cree una instancia del gestor de bases de datos.

## Decision Drivers

* RF02.1: Acceso a datos clientes.
* RF03.1: Acceso a datos pedidos.

## Considered Options

* 0016-1-Patrón Singleton

## Decision Outcome

Chosen option: "0016-1-Patrón Singleton", because Conseguimos solucionar el problema de una forma sencilla.

## Pros and Cons of the Options

### 0016-1-Patrón Singleton

Patrón de diseño creacional que se utiliza para crear una sola instancia de una clase.

* Good, because Soluciona el problema de una forma fácil.
* Bad, because Mayor complejidad en el diseño.
