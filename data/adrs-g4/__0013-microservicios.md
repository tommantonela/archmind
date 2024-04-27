## Considered Options

* 0013-1-Microsoft Azure Functions Component
* 0013-2-Insert Microservices Layer

## Decision Outcome

Chosen option: "0013-1-Microsoft Azure Functions Component", because both the design and implementation are simplified. Additionally, using this platform solves scalability and optimization issues as the software itself takes care of it.

### Positive Consequences

* Greater scalability and optimization.

## Pros of the Options

### 0013-1-Microsoft Azure Functions Component

External infrastructure from Microsoft that allows running parts of the code in the cloud.

* It is one of the most widely used "Serverless computing" platforms.
* There is no need to implement the software.
* Compatible with Java.

### 0013-2-Insert Microservices Layer

Include a middleware in the design to execute the microservices.

* Provides greater clarity that we are designing a microservices-based system.
* More complex to design and implement.


## Cons of the Options

### 0013-1-Microsoft Azure Functions Component

* Possible network latency issue in case of high demand.

### 0013-2-Insert Microservices Layer

* UML design with more classes and relationships.