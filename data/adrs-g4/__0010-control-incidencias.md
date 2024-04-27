# Incident Control

* Status: accepted
* Date: 2023-11-05

## Context and Problem Statement

We need to notify the type of incident in an efficient manner.

## Decision Drivers

* RF06: Incident module.

## Considered Options

* 0010-1-Publish and Subscribe Pattern
* 0010-2-Observer Pattern

## Decision Outcome

Chosen option: "0010-1-Publish and Subscribe Pattern", because it does not need to know who receives the notifications; the Order Manager class organizes it.

### Positive Consequences

* Order Manager is not called, allowing it to perform other actions before handling the queue.

### Negative Consequences

* Order Manager must always be listening for incidents.

## Pros of the Options

### 0010-1-Publish and Subscribe Pattern

Using the publish and subscribe pattern to notify incidents.

* Does not depend on who receives the notifications.

### 0010-2-Observer Pattern

Using the observer pattern to notify incidents.

* Calls the necessary classes when an event occurs.
* Requires observers to call.
* Calls immediately when an event occurs, invoking methods on the class.

## Cons of the Options

### 0010-1-Publish and Subscribe Pattern

* Does not send specific notifications for each class.

### 0010-2-Observer Pattern

* Requires observers to be set up.
* Invokes methods immediately upon event occurrence.