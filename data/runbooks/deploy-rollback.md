# Runbook: Rolling back a deploy

**Blast radius: critical.** Requires operator approval.

## Before proposing a rollback

Satisfy all four. A rollback proposed on fewer is a guess.

1. The deploy landed **before** the onset, and close to it. A deploy after onset
   cannot be the cause.
2. The metric change is a **step**, aligned to the deploy time. A gradual ramp
   starting hours earlier is not explained by a deploy at onset.
3. The change is **plausibly capable** of the observed effect. "Refactor
   serializer" can cause an N+1 query; "update README" cannot.
4. **No better explanation** fits the same evidence.

## Correlation is not causation

Several deploys land every hour in an active system. The most recent one is not
automatically guilty. Ask specifically: *what mechanism connects this diff to
this symptom?* If you cannot name one, keep investigating.

## When a rollback is the wrong tool

- The cause is **time-based** (certificate expiry, cron, quota reset). Nothing
  to roll back.
- The cause is a **shared dependency**. Rolling back a victim adds a change to
  an incident.
- The deploy is **days old** and the failure is a slow leak. Rollback is still
  correct, but the deploy will not be the recent one — search wider.
- The deploy included a **database migration**. Rolling back application code
  against a migrated schema can be worse than the incident. Check for schema
  changes first.

## After rolling back

Verify recovery in metrics, not in hope. If metrics do not improve within a few
minutes, **the hypothesis was wrong** — say so, revert the rollback if it was
itself disruptive, and resume investigating. A rollback that changes nothing is
strong evidence against your diagnosis and should update it.
