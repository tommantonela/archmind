# Payment Gateway

* Status: accepted
* Date: 2023-11-07

## Context and Problem Statement

We need to choose the external component that will handle payments and connect with the payment gateway.

## Decision Drivers

* RF07.1: External gateway.

## Considered Options

* 0012-1-Redsys Rest Api

## Decision Outcome

Chosen option: "0012-1-Redsys Rest Api", because with this component, we can design the payment process in a straightforward manner since it is compatible with Java.

### Positive Consequences

* Compatible with Java.
* Easy to design.

## Pros of the Options

### 0012-1-Redsys Rest Api

Redsys is one of the most widely used web payment gateways, allowing payment via credit and debit cards.

* Compatible with Java.
* Reliable and proven.

## Cons of the Options

### 0012-1-Redsys Rest Api

* Requires registration to use.