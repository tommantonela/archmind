# Order Manager Creation

* Status: accepted
* Date: 2023-11-12

## Context and Problem Statement

We need to ensure that only one instance of the Order Manager is created.

## Decision Drivers

* RF03.2: Order manager.

## Considered Options

* 0015-1-Singleton Pattern

## Decision Outcome

Chosen option: "0015-1-Singleton Pattern", because by using this pattern, we solve the problem of having only one instance in a straightforward manner.

## Pros of the Options

### 0015-1-Singleton Pattern

It is a creational design pattern used to ensure that only one instance of a class is created.

* We achieve having only one instance of the Order Manager.
* Provides a straightforward solution.

## Cons of the Options

### 0015-1-Singleton Pattern

* Increases complexity in design.