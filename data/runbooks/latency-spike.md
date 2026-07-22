# Runbook: Latency spike (p99 breach)

**Applies to:** any HTTP service breaching its p99 latency SLO.

## Triage

1. Confirm the breach is real and not a metrics artefact — check that request
   volume is non-zero. A p99 computed over three requests is noise.
2. Determine whether this service is the origin or a victim. Check the p99 of
   every service it depends on. **The service that degraded first is closer to
   the cause than the one that alerted.**

## Investigation order

1. **Deploys in the last 60 minutes.** Highest-yield check. Note that config
   changes (`cfg-` ids) count as deploys.
2. **Request rate.** If rps is flat while latency rose, the cause is internal —
   a code change, a config change, or a dependency. If rps rose sharply, you may
   simply be out of capacity.
3. **Downstream latency.** Database, cache, and upstream service latency.
4. **Query volume.** A latency rise paired with a large *query count* rise
   almost always means an N+1 query was introduced — an ORM change that removed
   eager loading is the classic cause. Look for a repeated identical query in
   the slow query log.
5. **Connection pool.** Rising pool wait time means requests are queueing for a
   connection, not executing slowly.

## Signature: N+1 query regression

- p99 latency: step change, not a ramp
- database queries_per_sec: large step increase (10x or more)
- application request rate: **flat**
- slow query log: the same `SELECT` repeated with different parameters
- a deploy touching serialization, ORM, or model code shortly before onset

**Fix:** roll back the implicated deploy. Scaling the application makes this
worse — more replicas means more concurrent queries against the same database.

## Escalation

Page the database on-call if database CPU is above 90% for more than 5 minutes.
