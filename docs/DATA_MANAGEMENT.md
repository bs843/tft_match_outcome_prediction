# Data Management and Privacy

- API credentials and HMAC keys exist only in `.env` or the process environment.
- Raw responses and metadata containing PUUIDs are private, access-controlled research data.
- Public analytical identifiers are project-specific HMAC-SHA256 pseudonyms. They remain pseudonymous, not anonymous.
- Request logs store endpoint templates and error classes; they never store credentials, PUUIDs, Riot IDs or response bodies.
- Raw responses are append-only. Every file receives a SHA-256 checksum and cannot be overwritten by a resume operation.
- Rejected candidates are retained privately for auditability and excluded from curated tables.
- Retrieval-time rank snapshots are private sampling metadata and never primary model features. Every accepted participant receives an explicit `found`, `unranked`, `not_found` or `request_failed` status; the snapshot table never stores PUUIDs.
- Publication packages contain code, schemas, configuration templates, aggregate results and sanitized fixtures—not raw responses or identifiers.
- Retention duration, access roles, encrypted backup location and secure deletion date must be completed in the institutional data-management plan before collection.
- Riot API terms and TFT policies must be reviewed immediately before the pilot and main run.
