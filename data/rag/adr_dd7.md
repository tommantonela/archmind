# Architecture Decision Record

## Title
Selection of HTTP/REST API Pattern for Statistics and Incident Reporting Modules

## Motivation
The food company is transitioning to a microservices architecture and requires a system that can provide statistics on orders, trucks, and customers, as well as collect and report incidents. The chosen pattern should align with the use of HTTP/REST protocols through a Gateway component and support the critical functionalities of the system.

## Decision Drivers
1. The system must include a module to provide statistics on orders, trucks, and customers.
2. The system must provide a module to collect and report incidents.

## Main Decision
The selected pattern is the HTTP/REST API pattern. This decision aligns well with the transition to microservices architecture and the use of HTTP/REST protocols. It provides a stateless and uniform interface for communication, which is crucial for modules like statistics on orders, trucks, and customers, as well as incident reporting. The simplicity and scalability of this pattern make it suitable for the specified requirements.

The HTTP/REST API pattern allows for easy integration with microservices, facilitates communication between different modules, and ensures a standardized approach to data exchange. By leveraging this pattern, the system can efficiently handle requests for statistics and incident reporting while maintaining a high level of flexibility and scalability.

## Alternatives
1. Event Sourcing Pattern:
   - Pros: Ensures data consistency and auditability, Enables rebuilding state at any point in time.
   - Cons: Complexity in managing event streams, Increased storage requirements.

2. Command Query Responsibility Segregation (CQRS) Pattern:
   - Pros: Optimizes read and write operations independently, Scalable for high-performance reads.
   - Cons: Introduces complexity with maintaining separate models, Requires synchronization mechanisms.

## Pros
- HTTP/REST API Pattern:
  - Provides a stateless and uniform interface for communication.
  - Aligns well with the transition to microservices architecture.
  - Simplifies integration and communication between modules.
  - Supports scalability and flexibility in handling requests.

- Event Sourcing Pattern:
  - Ensures data consistency and auditability.
  - Enables rebuilding state at any point in time.

- CQRS Pattern:
  - Optimizes read and write operations independently.
  - Supports high-performance reads.

## Cons
- HTTP/REST API Pattern:
  - May introduce overhead in managing API endpoints.
  - Could lead to increased network traffic if not optimized.

- Event Sourcing Pattern:
  - Complexity in managing event streams.
  - Increased storage requirements.

- CQRS Pattern:
  - Introduces complexity with maintaining separate models.
  - Requires synchronization mechanisms.

This ADR outlines the decision to utilize the HTTP/REST API pattern for implementing the statistics and incident reporting modules in the microservices architecture.