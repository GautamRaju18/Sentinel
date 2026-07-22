# Runbook: Database connection errors

**Applies to:** pool timeouts, "connection is not available", exhausted pools.

## The critical distinction

Errors mention the database, so the database gets blamed. Usually wrongly.
Establish which of these you have before acting:

| Observation | Meaning |
|---|---|
| pool **active** count rose to its ceiling | genuine demand exceeds the pool |
| pool **active** count *fell* | **the ceiling was lowered** — a config change |
| database connections at server `max_connections` | the database is the limit |
| database CPU high, few connections | slow queries, not connection starvation |

**A pool active count that DROPS while errors rise is the signature of a config
change that reduced `maximumPoolSize`.** Load did not rise; the ceiling fell.
Check for a configmap or config reload event at onset.

## Investigation order

1. Config changes and config reload log lines at the onset minute.
2. Pool stats logs — they usually print `total=`, which is the ceiling.
3. Application request rate. Flat rate plus new pool errors means the pool
   changed, not the traffic.
4. Database health independently: its own CPU, connection count, slow queries.
   A healthy database with a struggling client exonerates the database.

## Fix

- Config regression → **revert to the previous config revision.** Do not invent
  a new pool size during an incident; restore the known-good value.
- Genuine demand growth → raise pool size, but verify the database's
  `max_connections` can absorb `pool_size × replica_count` first.
- Slow queries holding connections → fix the query; a larger pool just queues
  more work against the same bottleneck.

## Do not

Do not scale up replicas to fix pool exhaustion. Each replica opens its own
pool, multiplying pressure on the database.
