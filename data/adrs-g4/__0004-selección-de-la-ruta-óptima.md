# Optimal Route Selection

* Status: accepted
* Date: 2023-10-26

## Context and Problem Statement

The system has two optimization algorithms for delivery based on delay, and we need to minimize this time.

## Decision Drivers

* RF04: Delivery and Routes module.

## Considered Options

* 0003-1-Strategy Pattern

## Decision Outcome

Chosen option: "0003-1-Strategy Pattern", because it allows selecting the most optimal algorithm in a simple and efficient way through an interface inherited by different algorithms.

## Pros of the Options

### 0003-1-Strategy Pattern

Design pattern that allows defining algorithms by placing each one in a different class, thus enabling the ability to alternate between the desired algorithms.

* Allows implementation of multiple algorithms.
* Utilizes the algorithm that best suits the system.
* Easy to scale in case more algorithms need to be added.

## Cons of the Options

### 0003-1-Strategy Pattern

Each algorithm is completely independent, so if one needs information from another, they cannot obtain it.