# Modularizing Delivery and Routes

* Status: proposed
* Date: 2023-10-26

## Context and Problem Statement

We want to group all the classes used in managing delivery and routes into one package.

## Decision Drivers

* RF04: Delivery and Routes module.

## Considered Options

* 0003-1-Delivery and Routes Module

## Decision Outcome

Chosen option: "0003-1-Delivery and Routes Module", because it will facilitate the implementation of future decisions and allow us to gather everything related to delivery and routes in the same package.

## Pros of the Options

### 0003-1-Delivery and Routes Module

Creating a module that contains everything necessary to manage deliveries and routes.

* Scalable.
* Achieves more modularity.
* Facilitates future work.

## Cons of the Options

### 0003-1-Delivery and Routes Module

If too many packages are created, it may worsen the visibility of the design.