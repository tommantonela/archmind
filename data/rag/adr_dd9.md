# Architecture Decision Record

## Title
Utilizing Event Sourcing Pattern for Incident Tracking and Reporting

## Motivation
The system needs to provide a module for collecting and reporting incidents. To address this requirement effectively, it is essential to have a comprehensive audit log of changes that can be tracked and analyzed for incident reporting purposes. The Event Sourcing pattern captures all changes as events, enabling the system to maintain a detailed history of incidents and generate reports based on the event history.

## Decision Drivers
- The system must provide a module to collect and report incidents.

## Main Decision
The chosen design decision is to implement the Event Sourcing pattern for incident tracking and reporting. By capturing all changes as events, the system can maintain a detailed audit log of incidents, analyze their causes, and generate reports based on the event history. This approach aligns well with the requirement for incident collection and reporting, providing traceability and auditability of data changes.

The Event Sourcing pattern offers advantages such as providing a full history of changes to the system, enabling auditing and traceability of data changes, and supporting the rebuilding of application state at any point in time. However, it comes with drawbacks like increased storage requirements due to storing all events and added complexity in the system with event handling and replaying events.

## Alternatives
1. **Command Query Responsibility Segregation (CQRS) pattern**
   - Pros:
     - Improves performance by optimizing read and write operations
     - Enables scalability by independently scaling read and write models
     - Supports complex queries efficiently
   - Cons:
     - Introduces complexity with maintaining separate read and write models
     - Requires additional effort for synchronization between read and write models

## Pros
- Event Sourcing pattern provides a full history of changes to the system.
- Enables auditing and traceability of data changes.
- Supports rebuilding application state at any point in time.

## Cons
- Increases storage requirements due to storing all events.
- Adds complexity to the system with event handling and replaying events.