# Modularizar-Pedidos

* Status: accepted
* Date: 2023-10-26

## Context and Problem Statement

Queremos englobar todas las clases relacionadas con los pedidos para agrupar su funcionalidad.

## Decision Drivers

* RF03: Módulo pedidos.
* RF03.3: Contabilizar número de intentos de realizar pedidos.

## Considered Options

* 0002-1-Módulo de Pedidos

## Decision Outcome

Chosen option: "0002-1-Módulo de Pedidos", because Facilitará la implementación de las futuras decisiones y nos permitirá reunir en el mismo módulo todo lo relacionado con los pedidos.

### Positive Consequences

* Este módulo actuará como intermediario entre el cliente y el gestor de pedidos.

## Pros and Cons of the Options

### 0002-1-Módulo de Pedidos

Hacer un módulo que contenga la clase Pedidos.

* Good, because Es escalable.
* Good, because Consigue más modularidad.
* Bad, because Si se hacen muchos paquetes puede empeorar la visibilidad del diseño.
