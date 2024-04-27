# Relocation of Gateway Component

* Status: accepted
* Date: 2023-10-31

## Context and Problem Statement

The gateway component should not be placed in the presentation layer since it is not something that the user can control.

## Decision Drivers

* RF08: Gateway Component.

## Considered Options

* 0006-1-Introduce it into the business logic layer
* 0006-2-Introduce it between the presentation layer and the business logic layer

## Decision Outcome

Chosen option: "0006-2-Introduce it between the presentation layer and the business logic layer", because we believe that this way the UML becomes cleaner and clearer.

### Positive Consequences

* Simpler UML.

### Negative Consequences

* It is not possible to know from the UML which classes use the component.

## Pros of the Options

### 0006-1-Introduce it into the business logic layer

Since it interacts with the classes of the business logic layer, it could be placed within it.

* It would be clearer which classes and packages use this component.

### 0006-2-Introduce it between the presentation layer and the business logic layer

It consists of placing the gateway component between the first two layers of the design.

* It is visually clearer.

## Cons of the Options

### 0006-1-Introduce it into the business logic layer

* Adds more relationships to the UML.

### 0006-2-Introduce it between the presentation layer and the business logic layer

* It is not known exactly which classes use the component.