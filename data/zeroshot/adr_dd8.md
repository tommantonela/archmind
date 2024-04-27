# Architecture Decision Record

## Title
Utilizing Retry Pattern for Handling Maximum Number of Order Placement Attempts

## Motivation
The current system transition from a monolithic architecture to microservices requires a mechanism to enforce a maximum number of order placement attempts for clients. This requirement aims to ensure system resilience and fault tolerance by allowing clients a specified number of retries in case of transient failures during the order placement process.

## Decision Drivers
- Clients must have a maximum number of attempts to place an order.

## Main Decision
The chosen design decision is to implement the Retry Pattern to address the requirement of clients having a maximum number of attempts to place an order. By incorporating the Retry Pattern, the system can provide fault tolerance and resilience to transient failures by allowing clients a predefined number of attempts to place an order. This approach enhances system availability and ensures that clients are not blocked after encountering temporary issues during the order placement process.

The system will track the number of order placement attempts per client and enforce the maximum limit set. Once the maximum number of attempts is reached, further actions will be defined, such as blocking additional attempts or notifying the client. Assumptions include the reliable tracking and enforcement of client attempts, with potential risks of abuse by clients attempting multiple orders. Trade-offs may arise between fault tolerance and system complexity due to the retry logic.

## Alternatives
1. Circuit Breaker Pattern
   - **Pros:** Protects downstream services from being overwhelmed.
   - **Cons:** May lead to service degradation if not configured properly.

## Pros
- Retry Pattern: Improves system resilience by handling transient failures effectively.
- Circuit Breaker Pattern: Protects downstream services from being overwhelmed.

## Cons
- Retry Pattern: May introduce delays in processing orders.
- Circuit Breaker Pattern: May lead to service degradation if not configured properly.