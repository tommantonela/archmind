# Modularizing Orders

* Status: accepted
* Date: 2023-10-26

## Context and Problem Statement

We want to group all the classes related to orders to consolidate their functionality.

## Decision Drivers

* RF03: Orders module.
* RF03.3: Counting number of order attempts.

## Considered Options

* 0002-1-Orders Module

## Decision Outcome

Chosen option: "0002-1-Orders Module", because it will facilitate the implementation of future decisions and allow us to gather everything related to orders in the same module.

### Positive Consequences

* This module will act as an intermediary between the client and the order manager.

## Pros of the Options

### 0002-1-Orders Module

Creating a module that contains the Orders class.

* Scalable.
* Achieves more modularity.

## Cons of the Options

### 0002-1-Orders Module

If too many packages are created, it may worsen the visibility of the design.