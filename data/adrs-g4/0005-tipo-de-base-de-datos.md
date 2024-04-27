# Tipo-De-Base-De-Datos

* Status: accepted
* Date: 2023-10-26

## Context and Problem Statement

Necesitamos acceder a la base de datos de una manera rápida y eficiente separando la responsabilidad de las clases Reparto y Rutas y GestorPedidos.

## Decision Drivers

* RF02: Módulo clientes.
* RF03.1: Acceso datos pedidos.
* RF03.2: Gestión pedidos.

## Considered Options

* 0005-1-DataBase-Per-Service
* 0005-2-Shared-DataBase-Per-Service

## Decision Outcome

Chosen option: "0005-2-Shared-DataBase-Per-Service", because Es más sencillo su diseño y permite tener todas las consultas agrupadas en una misma clase.

## Pros and Cons of the Options

### 0005-1-DataBase-Per-Service

Cada base de datos cuenta con su propio microservicio que se utilza para realizar consultas y modificar la información.

* Good, because Se separa en cada microservicio las consultas específicas para cada base de datos.
* Good, because La implementación de las consultas se realiza en una clase específica para ello.
* Bad, because No es fácilmente escalable puesto que con cada base de datos nueva se necesita crear un nuevo microservicio.

### 0005-2-Shared-DataBase-Per-Service

Todas las bases de datos utilizan el mismo microservicio para realizar consultas y modificar información.

* Good, because Solo se necesita una clase que acceda a las bases de datos.
* Good, because La implementación de las consultas se realiza en una clase específica para ello.
* Bad, because Es más complicado de implementar.
