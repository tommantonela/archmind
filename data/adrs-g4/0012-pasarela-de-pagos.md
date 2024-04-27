# Pasarela-De-Pagos

* Status: accepted
* Date: 2023-11-07

## Context and Problem Statement

Necesitamos escoger el componente externo que realizará los pagos y se conectará con la pasarela de pagos.

## Decision Drivers

* RF07.1: Pasarela externa.

## Considered Options

* 0012-1-Redsys Api Rest

## Decision Outcome

Chosen option: "0012-1-Redsys Api Rest", because Con este componente podemos diseñar de una forma sencilla la realización de los pagos puesto que es compatible con Java.

### Positive Consequences

* Compatible con Java.
* Fácil de diseñar.

## Pros and Cons of the Options

### 0012-1-Redsys Api Rest

Redsys es una de las pasarelas de pagos web más utilizadas permitiendo pagar mediante tarjeta de crédito y débito.

* Good, because Es compatible con Java.
* Good, because Fiable y probada.
* Bad, because Es necesario inscribirse para poder usarlo.
