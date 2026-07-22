# Runbook: Authentication outage

**Applies to:** login failures, token verification failures, SSO errors.

**Severity:** an auth outage is P1 by default. Nobody can use anything.

## Distinguishing feature: nothing was deployed

Auth outages are disproportionately **time bombs** rather than change-induced.
If the deploy history is empty around onset, that is a signal, not a dead end.
Time-triggered causes:

- **TLS or signing certificate expiry** — the most common by a wide margin
- OAuth client secret rotation or expiry
- SAML IdP metadata expiry
- clock skew breaking token validity windows
- a key rotation that did not propagate to every replica

## The CPU tell

During a certificate failure, **CPU usage falls**. The service is rejecting
handshakes instead of doing work. A struggling service works harder; a service
that cannot start a conversation works less. Falling CPU alongside rising errors
points at connection-establishment failure, not overload.

## Investigation order

1. `grep` logs for `x509`, `certificate`, `handshake`, `expired`. The error is
   usually explicit and names the expiry time.
2. Check certificate expiry on the service and every identity provider it talks
   to.
3. Confirm the onset time matches the expiry timestamp exactly. An exact match
   is conclusive.
4. Check whether downstream services showing auth errors are victims — a
   checkout service returning 401s because auth is down is not a second
   incident.

## Fix

Renew and roll out the certificate, then restart pods so they load it.
**Restarting alone does nothing** — the expired certificate on disk is still
expired, and pods will fail identically on startup. If a restart "did not help",
that is confirmation of the diagnosis, not a refutation.

## Follow-up (non-negotiable)

Add expiry alerting at 30, 14 and 7 days. A certificate expiry reaching
production is a monitoring failure, not a certificate failure.
