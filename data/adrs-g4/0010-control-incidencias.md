# Control-Incidencias

* Status: accepted
* Date: 2023-11-05

## Context and Problem Statement

Necesitamos notificar el tipo de incidencia de una manera eficiente.

## Decision Drivers

* RF06: Modulo de incidencias.

## Considered Options

* 0010-1-Patrón publish and subscribe
* 0010-2-Patrón observer

## Decision Outcome

Chosen option: "0010-1-Patrón publish and subscribe", because No necesita ver al que manda la notificación, la clase GestorPedidos lo organiza.

### Positive Consequences

* No se llama a gestor pedidos, permitiendole hacer otras acciones antes de atender la cola.

### Negative Consequences

* GestorPedidos tiene que estar escuchando siempre que haya una incidencia.

## Pros and Cons of the Options

### 0010-1-Patrón publish and subscribe

Usar el patron publish and subscribe para notificar las incidencias.

* Good, because No depende de quien reciba las notificaciones.
* Bad, because No manda notificaciones especificas para cada clase.

### 0010-2-Patrón observer

Usar el patron observer para notificar las incidencias.

* Good, because Llama a las clases necesarias cuando hay un evento.
* Bad, because Necesita observadores a los que llamar.
* Bad, because LLama cuando hay un evento mandandole a la clase a hacer metodos de inmediato.
