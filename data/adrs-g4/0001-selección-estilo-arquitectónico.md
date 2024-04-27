# Selección-Estilo-Arquitectónico

* Status: accepted
* Date: 2023-10-20

## Context and Problem Statement

En primer lugar debemos decidir cual es el estilo arquitectónico más optimo para realizar el diseño del problema dado.

## Decision Drivers

* RF01: Cambio de arquitectura.

## Considered Options

* 0001-1-Estilo-Por-Capas
* 0001-2-Estilo-REST
* 0001-3-Estilo-Por-Eventos

## Decision Outcome

Chosen option: "0001-1-Estilo-Por-Capas", because Permite separar los distintos componentes en capas de una forma óptima y nos permite implementar el resto de los problemas aplicando patrones de diseño.

### Negative Consequences

* Al cambiar una capa se afecta a otras

## Pros and Cons of the Options

### 0001-1-Estilo-Por-Capas

Arquitectura la cual separa los componentes de la aplicación en 4 capas, siendo estas la capa de presentación, la capa de lógica de negocio, capa de datos y capa de servicio.

* Good, because Es escalable.
* Good, because Permite la utilización de patrones de diseño dentro de cada capa.
* Good, because Permite tener una estructura organizada.
* Bad, because Una estructura compleja con muchas capas puede dificultar la comprensión y el mantenimiento.
* Bad, because La comunicación entre capas puede ser compleja.

### 0001-2-Estilo-REST

Arquitectura basada en que el cliente envíe peticiones al servidor y este le responda con el resultado de la consulta.

* Good, because El acceso en el sistema pasará a ser por protocolos HTTP/REST por lo que sería conveniente utilizar este estilo.
* Bad, because El estilo por capas es mejor ya que permite desacoplar unos módulos de otros.

### 0001-3-Estilo-Por-Eventos

Arquitectura que permite detectar los eventos, en este caso alguna acción de consulta o de pago, y actuar en consecuencia.

* Good, because Es una arquitectura muy eficiente y escalable.
* Good, because Proporciona respuestas a eventos a tiempo real.
* Bad, because Es compleja de implementar.
* Bad, because Requiere un diseño cuidadoso de los eventos para garantizar la entrega y el procesamiento de los eventos.
