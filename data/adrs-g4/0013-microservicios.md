# Ejecución-Microservicios

* Status: accepted
* Date: 2023-11-09

## Context and Problem Statement

Necesitamos ejecutar los microervicios e indicarlo en el diseño del sistema.

## Decision Drivers

* RF01: Cambio de arquitectura.

## Considered Options

* 0013-1-Componente Microsoft Azure Functions
* 0013-2-Insertar capa de microservicios

## Decision Outcome

Chosen option: "0013-1-Componente Microsoft Azure Functions", because Tanto el diseño como la implementación se simplifican, además que al usar esta plataforma los problemas relacionados con la escalabilidad y optimización se solucionan al encargarse el propio software de ello.

### Positive Consequences

* Mayor escalabilidad y optimización.

## Pros and Cons of the Options

### 0013-1-Componente Microsoft Azure Functions

Infraestructura externa de Microsoft que permite ejecutar partes del código en la nube.

* Good, because Es una de las plataformas de "Serverless computing" más usadas.
* Good, because No es necesario implementar el software.
* Good, because Compatible con Java.
* Bad, because Posible problema de alta latencia de red en caso de tener una alta demanda.

### 0013-2-Insertar capa de microservicios

Incluir en el siseño un middleware que permita ejecutar los microservicios.

* Good, because Mayor claridad que estamos diseñando un sistema basado en microservicios.
* Bad, because Más complejo de diseñar e implementar.
* Bad, because Diseño UML con más clases y relaciones.
