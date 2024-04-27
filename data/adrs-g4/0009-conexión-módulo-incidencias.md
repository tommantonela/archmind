# Conexión-Módulo-Incidencias

* Status: accepted
* Date: 2023-11-02

## Context and Problem Statement

El módulo incidencias debe reportar cualquier tipo de incidencia a los gestores de las rutas.

## Decision Drivers

* RF06: Módulo de incidencias.

## Considered Options

* 0009-1-Conectar el módulo incidencias con la gestión de reparto y rutas.
* 0009-2-Conectar el módulo incidencias con el gestor pedidos.

## Decision Outcome

Chosen option: "0009-2-Conectar el módulo incidencias con el gestor pedidos.", because En el enunciado se pide de forma explícita.

### Positive Consequences

* Relegamos todas las comunicaciones de logica de servicios a gestor pedidos.
* Más escalable.

### Negative Consequences

* GestorPedidos se vuelve más complejo.
* Involucramos más clases de forma innecesaria.

## Pros and Cons of the Options

### 0009-1-Conectar el módulo incidencias con la gestión de reparto y rutas.

Puesto que las incidencias ocurren a la hora del reparto, las notificaciones se realizan en el módulo de reparto y rutas.

* Good, because Implementación más lógica del problema.
* Good, because Liberar de carga a la clase GestorPedidos.
* Bad, because Mayor complejidad para comunicar las incidencias con el resto del sistema.

### 0009-2-Conectar el módulo incidencias con el gestor pedidos.

El enunciado del problema describe que la clase GestorPedidos hace de intermediario entre los clientes, pedidos, el reparto y las incidencias comunicando dichas funcionalidades.

* Good, because Solución más fiel al enunciado.
* Good, because Evita el problema de comunicación debido a que esta clase actúa como intermediario en todo el sistema.
* Bad, because Mayor complejidad a la hora de implementar la clase GestorPedidos.
