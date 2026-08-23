# Patch 17.8 multi-region collection amendment

Collection configuration `tft-patch17.8-final-collection-v7` expands Patch 17.8
collection across multiple Riot platform regions while preserving GOLD–MASTER
strata and all eligibility rules. The change addresses insufficient unique
seed-player coverage observed in previous single-region collection attempts
without introducing cross-patch balance variation.

## Enabled platforms

| Platform | Match routing cluster | Ladder scope |
|---|---|---|
| ME1 | EUROPE | GOLD IV–I, PLATINUM IV–I, EMERALD IV–I, DIAMOND IV–I, MASTER |
| EUW1 | EUROPE | same |
| EUN1 | EUROPE | same |
| NA1 | AMERICAS | same |
| KR | ASIA | same |

Each non-apex division uses three ladder pages and up to 160 selected players per
populated page; MASTER uses one page. Three in-window match-history pages are
requested per seed. Platform pools are interleaved during match evaluation. This
balances discovery opportunity without imposing equal accepted-match quotas.

## Invariants

Patch 17.8, client patch 16.15, Set 17, queue 1100, standard completed matches,
eight participants, placements 1–8, checksums, HMAC pseudonymisation, end-result
validation, and all critical rejection rules are unchanged. At least 25,000
globally unique candidates are required before downloads; exactly 5,000 accepted
matches are required before an accepted manifest is frozen.

Every seed, source association, globally deduplicated candidate, collection-status
row, accepted-manifest entry and rank snapshot records explicit platform and route
provenance. Stored raw JSON may be reused only after checksum verification and is
revalidated for the new run. Acceptance is never inherited.

Match-history requests are cluster scoped, so a seed from one platform can return
a match whose ID is prefixed by another platform in that cluster. Configuration v6
uses the Riot match-ID prefix as authoritative download provenance, retains the seed
platform separately as discovery provenance, and excludes prefixes outside the five
enabled platforms. The candidate discovery report counts these exclusions explicitly.

Runs `20260820T152821Z-9df0ecbb`, `20260820T212450Z-31431b86`, and
`20260822T192049Z-84373556` remain immutable, audit-only failures and cannot be
resumed or merged.

Region is audit metadata for diagnostics and robustness checks. It is prohibited
from the primary feature matrix. Held-out-region modelling is not performed here.
