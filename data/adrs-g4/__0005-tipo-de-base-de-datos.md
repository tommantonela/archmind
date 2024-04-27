# Database Type

* Status: accepted
* Date: 2023-10-26

## Context and Problem Statement

We need to access the database quickly and efficiently, separating the responsibility of the Delivery and Routes classes and OrderManager.

## Decision Drivers

* RF02: Clients module.
* RF03.1: Access to order data.
* RF03.2: Order management.

## Considered Options

* 0005-1-DataBase-Per-Service
* 0005-2-Shared-DataBase-Per-Service

## Decision Outcome

Chosen option: "0005-2-Shared-DataBase-Per-Service", because its design is simpler and allows all queries to be grouped in a single class.

## Pros of the Options

### 0005-1-DataBase-Per-Service

Each database has its own microservice used to perform queries and modify information.

* Separates specific queries for each database into each microservice.
* Query implementation is done in a specific class for that purpose.

### 0005-2-Shared-DataBase-Per-Service

All databases use the same microservice to perform queries and modify information.

* Only requires one class to access databases.
* Query implementation is done in a specific class for that purpose.

## Cons of the Options

### 0005-1-DataBase-Per-Service

* Not easily scalable as a new microservice is needed for each new database.

### 0005-2-Shared-DataBase-Per-Service

* More complicated to implement.