# Order Attempt

* Status: accepted
* Date: 2023-10-31

## Context and Problem Statement

We need the order to be blocked after a limited number of attempts.

## Decision Drivers

* RF3.3-Count number of order attempts.

## Considered Options

* 0008-1-Retry Pattern

## Decision Outcome

Chosen option: "0008-1-Retry Pattern", because it is the most optimal way to solve the problem since it is a design pattern that allows controlling the entire process of attempts when placing an order.

### Positive Consequences

* Efficient solution.
* Scalable.
* Allows controlling the number of attempts when placing an order.

### Negative Consequences

* More complex UML.

## Pros of the Options

### 0008-1-Retry Pattern

It is a stability pattern that consists of retrying a failed operation while also adding the number of attempts that can be made.

* Efficiently solves the design problem.

## Cons of the Options

### 0008-1-Retry Pattern

* Adds more complexity to the UML diagram.
