# NLSY Deep Dive

This note extends the NLSY lane beyond the older `g_proxy`-only framing. The cohort models now add blocks for skills and traits, family background, occupation sorting, family formation, and adult resources.

Important interpretation rule: these are not all exogenous causal controls. Some later-life blocks, especially family formation and adult resources, are better read as mechanism-sensitive accounting blocks than as clean pre-market controls.
All block comparisons below use the common complete-case sample for the cohort's final model, so the reductions are not being driven by stepwise sample changes.

## NLSY79

- Raw annual-earnings gap: 44.43%
- Final deep-model gap: 34.10%
- Common-sample observations: 2890
- Largest reduction comes from: occupation sorting (7.02 percentage points).
- Next-largest reduction: skills and noncognitive traits (5.62 points).
- Skills block reduction: 5.62 points.
- Family-background block reduction: 0.04 points.
- Occupation-sorting block reduction: 7.02 points.
- Family-formation block reduction: 0.18 points.
- Adult-resources block reduction: -0.95 points.
- Caution: the adult-resources block is post-market and should be read as mechanism-sensitive accounting, not a clean exogenous explanation.

## NLSY97

- Raw annual-earnings gap: 34.19%
- Final deep-model gap: 31.89%
- Common-sample observations: 2486
- Largest reduction comes from: adult resources (6.60 percentage points).
- Next-largest reduction: skills and school achievement (1.65 points).
- Skills block reduction: 1.65 points.
- Family-background block reduction: -0.06 points.
- Occupation-sorting block reduction: 0.00 points.
- Family-formation block reduction: -0.21 points.
- Adult-resources block reduction: 6.60 points.
- Caution: the adult-resources block is post-market and should be read as mechanism-sensitive accounting, not a clean exogenous explanation.

## Takeaway

NLSY is useful because it can test richer background and life-course factors than ACS or CPS. The main lesson from these cohort files is still not that one factor explains the gap. Skills, background, occupation, and family/resource variables each matter somewhat, but meaningful residual gaps remain after all of them are added.
