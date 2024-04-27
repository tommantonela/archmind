# Statistics Component

* Status: accepted
* Date: 2023-11-07

## Context and Problem Statement

We need to create a statistics module that allows us to provide information about the status of orders and the real-time situation of trucks.

## Decision Drivers

* RF05: Statistics module.

## Considered Options

* 0011-1-Apache Common Maths Component
* 0011-2-Implement a Class

## Decision Outcome

Chosen option: "0011-1-Apache Common Maths Component", because it is easier to design and thus solves the problem.

### Positive Consequences

* Easy to design.
* Simpler solution.

### Negative Consequences

* Requires installation of the library and Apache Maven.

## Pros of the Options

### 0011-1-Apache Common Maths Component

Apache Common Maths is a library that allows statistical calculations.

* Compatible with Java.
* Self-contained.
* Lightweight.

### 0011-2-Implement a Class

Create a class that implements all statistical operations.

* We can select which queries can be performed.
* No need to install any external library.

## Cons of the Options

### 0011-1-Apache Common Maths Component

* Requires installation of Apache Maven as well.

### 0011-2-Implement a Class

* More complex to design and implement.