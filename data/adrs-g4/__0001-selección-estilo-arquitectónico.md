# Architectural Style Selection

* Status: accepted
* Date: 2023-10-20

## Context and Problem Statement

In the development process, choosing the appropriate architectural style is crucial for ensuring the scalability, maintainability, and efficiency of the system. Therefore, it is essential to evaluate different architectural options to determine the most suitable one for addressing the current problem.

## Decision Drivers

* RF01: Architecture change. The need for a new architectural style arises due to the evolving requirements of the system, requiring a reevaluation of the existing architecture.

## Considered Options

* 0001-1-Layered-Style
* 0001-2-REST-Style
* 0001-3-Event-Driven-Style

## Decision Outcome

Chosen option: "0001-1-Layered-Style", because it allows separating the different components into layers in an optimal way and enables us to implement the rest of the issues by applying design patterns.

### Negative Consequences

Changing one layer affects others.

## Pros of the Options

### 0001-1-Layered-Style

Architecture that separates the application components into 4 layers: presentation layer, business logic layer, data layer, and service layer.

* It is scalable, allowing each layer to scale independently according to demand.
* It allows the use of design patterns within each layer, promoting modularity and reusability.
* It provides an organized structure, enhancing maintainability and collaboration among developers.

### 0001-2-REST-Style

Architecture based on the client sending requests to the server and the server responding with the result of the query.

* System access will be via HTTP/REST protocols, providing a standardized and widely adopted approach for communication.
* It simplifies client-server interactions by leveraging uniform interfaces and stateless communication.

### 0001-3-Event-Driven-Style

Architecture that allows detecting events, such as query or payment actions, and acting accordingly.

* It is a very efficient and scalable architecture, particularly suitable for real-time processing and event-driven systems.
* It provides real-time responses to events, enabling timely reactions to changes in the system's environment.

## Cons of the Options

### 0001-1-Layered-Style

* A complex structure with many layers can hinder understanding and maintenance, especially for developers unfamiliar with the architecture.
* Communication between layers can be complex, requiring careful management of dependencies and interfaces.

### 0001-2-REST-Style

* The layered style may offer better modularity and decoupling between modules, making it preferable in certain scenarios.

### 0001-3-Event-Driven-Style

* It is complex to implement, requiring specialized knowledge and careful design to ensure robust event handling.
* It requires careful event design to ensure event delivery and processing, which can introduce additional overhead and complexity.