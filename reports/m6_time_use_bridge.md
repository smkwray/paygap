# M6 Time-Use Bridge

This artifact is the defensible `M6` layer for the project.
It is not a merged ACS+ATUS regression and should not be described that way.
Instead, it uses ATUS as a separate mechanism module to interpret the residual ACS/CPS wage gaps after `M5` and the selection-robustness checks.

## Why This Is Separate

- ACS/CPS identify realized wages, hours, and worker characteristics at scale.
- ATUS identifies daily time allocation, especially unpaid work and care burdens.
- ATUS cannot be merged person-by-person onto ACS/CPS in this repo's public-data workflow.
- So the right use is mechanism interpretation, not a literal control variable inside the wage regression.

## ATUS Burden Snapshot

- Female paid-work time gap: -67.96 minutes/day
- Female housework gap: 31.91 minutes/day
- Female childcare gap: 11.37 minutes/day
- Female commute gap: -8.07 minutes/day
- Female unpaid-work gap (housework + childcare): 43.28 minutes/day
- Female net paid+unpaid gap: -24.68 minutes/day

## Residual Gap Context

- Mean ACS `M5` female coefficient: -0.1450
- Latest ACS `M5` female coefficient: -0.1408
- Mean ACS `S2` IPW worker hourly gap: 19.73%
- Mean ACS `S2` combined expected annual-earnings gap: 38.48%
- Mean CPS `S2` IPW worker hourly gap: 19.25%
- Mean CPS `S2` combined expected annual-earnings gap: 31.99%

## Interpretation

- The ATUS evidence is directionally consistent with a family/schedule mechanism: women spend materially less time in paid work and materially more time in unpaid household and childcare work.
- That mechanism helps explain why annual-earnings gaps are larger than worker-only hourly wage gaps in the ACS/CPS selection surfaces.
- But the ATUS burden differences do not make the residual ACS worker-gap disappear; they coexist with a still-material post-`M5` residual gap.

## Use In The Final Project

Present this as a mechanism-aware bridge after `M5` and the selection results:
1. ACS/CPS establish the realized gap and residual worker-gap.
2. Selection robustness distinguishes worker gaps from broader employment and annual-earnings gaps.
3. ATUS explains one plausible channel: women carry more unpaid work and childcare time, which is consistent with schedule constraints and reduced paid-work time.
4. SCE adds a separate expectations/reservation-wage channel.

This is a stronger and more defensible story than pretending ATUS creates a literal merged `M6` wage regression.
