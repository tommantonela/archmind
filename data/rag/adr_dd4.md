# Architecture Decision Record

## Title
Utilizing the Strategy Pattern for Optimizing Delivery and Order Routing

## Motivation
The current system requires a module to optimize delivery and order routing based on delay, with the need for two different optimization algorithms. To address this requirement effectively within the microservices architecture, a design decision is needed to encapsulate and manage these algorithms efficiently.

## Decision Drivers
1. The system must provide a module to optimize delivery and order routing based on delay.
2. Two optimization algorithms for route assignment need to be implemented.

## Main Decision
The chosen design decision is to implement the Strategy Pattern for optimizing delivery and order routing with different algorithms. By encapsulating the algorithms within separate strategies, the Strategy Pattern enables easy swapping between the two optimization algorithms based on delay. This approach promotes flexibility, maintainability, and scalability within the system. The Strategy Pattern allows for the encapsulation of each algorithm, ensuring that changes or additions to the optimization strategies can be done independently without affecting the core routing logic. This design decision aligns well with the microservices architecture and the need for modular, interchangeable components.

## Alternatives
1. Factory Method Pattern: Defines an interface for creating objects but may introduce complexity and an additional layer of abstraction. It could lead to an explosion of subclasses if many products are required.

## Pros
### Strategy Pattern
- Encourages flexibility and maintainability by encapsulating algorithms separately
- Promotes easy swapping between optimization algorithms
- Aligns well with the microservices architecture and modular design principles

### Factory Method Pattern
- Promotes loose coupling by eliminating the need to bind application-specific classes into the code
- Allows subclasses to alter the type of objects that will be created
- Encapsulates object creation, providing a hook for extended behavior

## Cons
### Strategy Pattern
- Requires additional implementation effort to set up the strategy objects
- May introduce complexity if not managed properly

### Factory Method Pattern
- Can lead to an explosion of subclasses if many products are required
- May introduce complexity by adding an additional layer of abstraction