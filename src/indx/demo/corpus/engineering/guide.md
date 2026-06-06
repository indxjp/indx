# Engineering Guide

How we build and ship software at Acme Robotics. New hires should read this during
[onboarding](../handbook/onboarding.md).

## Principles

- **Small reversible changes.** Prefer many small pull requests over one large one.
- **Trunk-based development.** Branch from `main`, merge back quickly.
- **Tests are part of the change.** A bug fix without a regression test is incomplete.

## Shipping a change

1. Open a branch and write the change with tests.
2. Open a pull request and request review (see [Code Review](code-review.md)).
3. Once approved and green, merge. Deploys are automatic from `main`.

## Tooling

We use typed Python with strict static checks. Formatting and linting run in CI, so
a green pull request is one that is already formatted and linted locally.
