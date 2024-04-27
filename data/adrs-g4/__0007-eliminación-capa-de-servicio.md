# Removal of Service Layer

* Status: accepted
* Date: 2023-10-31

## Context and Problem Statement

It is not necessary to have the Incidents and Statistics classes separated into a separate layer since they are part of the business logic.

## Decision Drivers

* RF05: Statistics module.
* RF06: Incidents module.

## Considered Options

* 0007-1-Introduce the classes into the business logic layer

## Decision Outcome

Chosen option: "0007-1-Introduce the classes into the business logic layer", because with this decision, we correct the initial mistake and obtain a design closer to the problem description.

### Positive Consequences

* Elimination of unnecessary layer.
* Correction of a design mistake.

### Negative Consequences

* More classes in the business logic layer.

## Pros of the Options

### 0007-1-Introduce the classes into the business logic layer

Since the functionalities of the two classes are related to business logic, we decide to introduce the classes into the corresponding layer.

* Eliminates unnecessary layer.
* Corrects a wrong design decision.

## Cons of the Options

### 0007-1-Introduce the classes into the business logic layer

* Adds more classes to the business logic layer.