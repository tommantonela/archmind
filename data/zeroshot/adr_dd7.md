# Architecture Decision Record (ADR)

## Title
Selection of HTTP/REST API Pattern for Statistics and Incident Reporting Modules

## Motivation
The transition to a microservices architecture by the food company necessitates the adoption of a suitable pattern to provide statistics on orders, trucks, and customers, as well as to collect and report incidents. The HTTP/REST API pattern aligns well with the requirements, leveraging stateless and uniform communication interfaces crucial for these modules. This pattern also complements the use of HTTP/REST protocols and supports the scalability and simplicity needed for the specified functionalities.

## Decision Drivers
1. The system must include a module to provide statistics on orders, trucks, and customers.
2. The system must provide a module to collect and report incidents.

## Main Decision
The chosen design decision is to implement the HTTP/REST API pattern for the statistics and incident reporting modules. This pattern offers a stateless and uniform communication interface, which is essential for these functionalities. By utilizing HTTP/REST principles, the system can achieve scalability, simplicity, and platform-independent communication, meeting the specified requirements effectively. Assumptions are made regarding the effective access and transformation of existing SQL database data into the required format for this pattern.

## Alternatives
1. Gateway Pattern
2. Database per Service pattern
3. Microservices Architecture pattern
4. Command Query Responsibility Segregation (CQRS) pattern
5. RESTful API pattern
6. Event Sourcing pattern

## Pros
- **HTTP/REST API Pattern:**
  - Advantages:
    1. Statelessness and uniform interface for communication.
    2. Simplifies integration and scalability.
- **Alternative Patterns:**
  - Pros vary based on the specific pattern but include benefits like enhanced service independence, scalability, and performance improvements.

## Cons
- **HTTP/REST API Pattern:**
  - Disadvantages:
    1. Overhead of HTTP protocol.
    2. Limited support for complex interactions.
- **Alternative Patterns:**
  - Cons differ for each pattern but may involve challenges such as increased complexity, data consistency issues, and performance overhead.

This ADR outlines the rationale behind selecting the HTTP/REST API pattern for the statistics and incident reporting modules, considering the system context, requirements, and assessment provided.