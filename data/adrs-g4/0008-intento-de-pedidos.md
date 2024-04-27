# Intento-De-Pedidos

* Status: accepted
* Date: 2023-10-31

## Context and Problem Statement

Necesitamos que tras un número limitado de intentos al realizar el pedido este se bloquee.

## Decision Drivers

* RF3.3-Contabilizar número de intentos de realizar pedidos.

## Considered Options

* 0008-1-Patrón Retry

## Decision Outcome

Chosen option: "0008-1-Patrón Retry", because Es la forma más óptima de resolver el problema puesto que es un patrón de diseño el cual permite controlar todo el proceso de intentos al realizar un pedido.

### Positive Consequences

* Solución eficiente.
* Es escalable.
* Permite controlar el número de intentos a la hora de realizar un pedido.

### Negative Consequences

* UML más complejo.

## Pros and Cons of the Options

### 0008-1-Patrón Retry

Es un patrón de estabilidad el cual consiste en reintentar una operación que ha fallado añadiendo además el número de intentos que se pueden realizar.

* Good, because Soluciona de manera eficiente el problema de diseño.
* Bad, because Añade más complejidad al diagrama UML.
