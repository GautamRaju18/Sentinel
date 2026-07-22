# Runbook: Multiple services degraded simultaneously

**Applies to:** three or more services breaching SLOs within a few minutes.

## The trap

When several services alert at once, the instinct is to roll back whichever one
alerted first or is most visible. This is usually wrong. Simultaneous
degradation across independent services almost always means a **shared
dependency** failed, and none of the alerting services is the cause.

Rolling back an innocent service adds a second change to an ongoing incident.

## Find the origin by ordering, not by loudness

Build a timeline of when each service *started* degrading — not when it alerted.
Alerting thresholds differ, so alert order is not onset order.

**The component that degraded first is the origin.** Everything downstream of it
is fallout. In a cascade the ordering is usually:

```
shared dependency → direct consumer → consumer's consumers
```

Candidate shared dependencies: database, cache (Redis/Memcached), message
broker, service mesh, DNS, an internal auth service, a shared node pool, or the
network itself.

## Confirming a cascade

- Onset times are staggered by seconds to a couple of minutes, in dependency order
- Errors are timeouts and circuit-breaker trips rather than application errors
- `threads_blocked` or queue depth rising in the intermediate service
- **No deploys** anywhere near onset

## Redis-specific: `MISCONF` and fork failures

`MISCONF Redis is configured to save RDB snapshots but is unable to persist to
disk` combined with `fork failed: Cannot allocate memory` means Redis cannot
fork for background saving. Redis stalls command handling, and every consumer
blocks.

Cause: `vm.overcommit_memory` is not set to 1, or the host lacks free memory for
the fork (Redis forks a copy-on-write child; the kernel refuses without
overcommit).

**Fix:** set `vm.overcommit_memory=1` on the host and relieve memory pressure,
or disable RDB if AOF persistence is sufficient. Do not restart the consuming
services — they recover on their own once the dependency does.

## Rule

Do not roll back any service until you have established the onset ordering.
