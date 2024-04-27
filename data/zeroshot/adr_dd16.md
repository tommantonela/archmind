# Architecture Decision Record

## Title
Microservices Pattern for System Architecture

## Motivation
The company aims to transition from a monolithic system to a microservices architecture to improve scalability, flexibility, and maintenance of the system. The existing data stored in SQL databases (Customers, Orders) needs to be accessible through the new architecture. The critical modules for Customers, Delivery & Routing, and Payments, as well as non-critical modules for Orders, Statistics, and Incidents, must be supported. The system must allow access to client data and order data efficiently.

## Decision Drivers
1. The system must allow access to client data.
2. The system must allow access to order data.

## Main Decision
The chosen design decision is to implement the Microservices Pattern. By decomposing the system into smaller, independent services, the Microservices Pattern aligns with the company's transition plan. It allows for better scalability, flexibility, and maintenance of the system. The OrderManager component will be implemented as a microservice to facilitate communication between different functionalities, ensuring efficient management of client and order data within the microservices architecture.

The transition to microservices may impact the existing data stored in SQL databases, but assuming effective migration, the Microservices Pattern can enhance scalability, flexibility, and maintenance. It can improve availability and fault tolerance, although it may introduce complexity in managing inter-service communication and deployment.

## Alternatives
1. Gateway Pattern: Acts as a centralized entry point for client requests, providing centralized access control and handling various protocols.
   - Pros: Centralized access control, protocol translation.
   - Cons: Single point of failure.
   
2. HTTP/REST API Pattern: Uses HTTP methods for resource manipulation, offering standardized communication.
   - Pros: Interoperability, simplicity, statelessness.
   - Cons: Limited functionality, potential performance issues.

## Pros
### Microservices Pattern
- Scalability
- Fault isolation
- Technology diversity

### Gateway Pattern
- Centralized access control
- Protocol translation

### HTTP/REST API Pattern
- Interoperability
- Simplicity
- Statelessness

## Cons
### Microservices Pattern
- Complexity
- Increased network communication

### Gateway Pattern
- Single point of failure

### HTTP/REST API Pattern
- Limited functionality
- Potential performance issues