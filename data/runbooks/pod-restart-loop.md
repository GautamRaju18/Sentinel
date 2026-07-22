# Runbook: Pod restart loop / CrashLoopBackOff

**Applies to:** a service whose pods restart repeatedly.

## First: read the exit code

- **137** — OOMKilled. The container exceeded its memory limit. Go to
  *Memory exhaustion* below.
- **139** — segfault. Native code or a runtime bug.
- **143** — SIGTERM, graceful. Usually an eviction or a rollout, not a crash.
- **1 / 2** — the application exited on its own. Read the last 50 log lines
  before each restart; the cause is almost always printed there.

## Memory exhaustion

Check the memory metric's **shape**, not its current value:

- **Sawtooth** (climbs, resets, climbs) — a leak. The process is being killed
  and restarted on a cycle. Each tooth is one pod lifetime.
- **Steady ramp with no reset** — accumulation that has not yet hit the limit.
  You have time; act before it does.
- **Step change** — a configuration change altered the limit or the workload.

A rising GC pause alongside rising heap confirms a leak rather than a
legitimately larger working set.

## Finding the leak

The triggering change is **often not recent**. A leak deployed days ago only
pages you once it crosses the limit under production load. Search deploy history
for changes touching caches, session storage, connection pooling, or listener
registration — not just the last 24 hours.

Common causes:
- an in-memory cache with no eviction policy or size bound
- listeners or subscriptions registered per-request and never removed
- unbounded queues or buffers
- thread-local storage in a pooled-thread environment

## Mitigation vs fix

Raising the memory limit and adding replicas **buys time**; it does not fix a
leak — it lengthens the interval between crashes. Say so explicitly when
proposing it. The fix is bounding the growth or reverting the change.

Restarting a leaking service resets memory and buys roughly one leak-cycle of
time. Useful during an outage, never a resolution.
