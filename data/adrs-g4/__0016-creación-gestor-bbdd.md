# Database Manager Creation

* Status: accepted
* Date: 2023-11-12

## Context and Problem Statement

We need to ensure that only one instance of the database manager is created.

## Decision Drivers

* RF02.1: Client data access.
* RF03.1: Order data access.

## Considered Options

* 0016-1-Singleton Pattern

## Decision Outcome

Chosen option: "0016-1-Singleton Pattern", because it solves the problem in a straightforward manner.

## Pros of the Options

### 0016-1-Singleton Pattern

Creational design pattern used to ensure that only one instance of a class is created.

* Solves the problem in a straightforward manner.

## Cons of the Options

### 0016-1-Singleton Pattern

* Increases complexity in design.