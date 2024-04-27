# Incident Module Connection

* Status: accepted
* Date: 2023-11-02

## Context and Problem Statement

The incidents module must report any type of incident to the route managers.

## Decision Drivers

* RF06: Incident module.

## Considered Options

* 0009-1-Connect the incidents module with the distribution and routes management.
* 0009-2-Connect the incidents module with the order manager.

## Decision Outcome

Chosen option: "0009-2-Connect the incidents module with the order manager.", because it is explicitly requested in the statement.

### Positive Consequences

* Delegates all logic service communications to the order manager.
* More scalable.

### Negative Consequences

* The order manager becomes more complex.
* Involves more classes unnecessarily.

## Pros of the Options

### 0009-1-Connect the incidents module with the distribution and routes management.

Since incidents occur during distribution, notifications are made in the distribution and routes module.

* More logical implementation of the problem.
* Relieves the order manager class of burden.

### 0009-2-Connect the incidents module with the order manager.

The problem statement describes that the order manager class acts as an intermediary between clients, orders, distribution, and incidents, communicating these functionalities.

* More faithful solution to the problem statement.
* Avoids communication issues as this class acts as an intermediary throughout the system.

## Cons of the Options

### 0009-1-Connect the incidents module with the distribution and routes management.

* Greater complexity in communicating incidents with the rest of the system.

### 0009-2-Connect the incidents module with the order manager.

* Greater complexity in implementing the order manager class.