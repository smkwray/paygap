# Reports Guide

This guide explains what each dataset contributes, which observed factors matter most in the current data, what is missing, and where the repo can expand next.

## How to read the current results

- `Sequential OLS` tells you which blocks of observed variables reduce the residual gap the most.
- `Oaxaca-style contributions` tell you which measured characteristics account for the explained part of the gap in a simple decomposition.
- `Mechanism datasets` like ATUS and SCE do not replace the main wage regressions; they help interpret them.

## ACS

- Role: primary headline dataset for repeated-year worker-gap estimates.
- 2023 headline results: raw gap 16.81%, adjusted gap 13.23%.
- Key wage-gap variables in use: age, race/ethnicity, education, state, occupation, industry, class of worker, hours, work from home, commute time, marital status, children.
- Largest observed gap reduction comes from: job sorting (8.07 percentage points).
- Top simple explained-gap contributors: usual_hours_week (10.02%); age (-4.12%); age_sq (3.67%).
- What is missing: firm/pay-setting, tenure, exact employer, detailed within-job task mix, direct bargaining measures.
- Main reports: `reports/findings_summary.md`, `reports/defensibility_report.md`, `reports/oaxaca_review.md`.

## CPS ASEC

- Role: external cross-check on the realized worker gap and selection robustness.
- 2023 headline results: raw gap 15.77%, adjusted gap 16.96%.
- Key wage-gap variables in use: age, race/ethnicity, education, state, occupation, industry, class of worker, family variables.
- Largest observed gap reduction comes from: job sorting (3.16 percentage points).
- Top simple explained-gap contributors: usual_hours_week (9.06%); age (-5.81%); age_sq (4.66%).
- What is missing: commute, firm effects, direct bargaining measures, richer job-quality variables.
- Main reports: `reports/selection_robustness.md`, `reports/final_synthesis.md`.

## SIPP

- Role: validated public-use extension dataset with a first descriptive and adjusted-gap surface.
- 2023 headline results: raw gap 15.09%, adjusted gap 10.91%.
- Key wage-gap variables in use: month, occupation, industry, hours, paid-hourly status, multiple-jobholder status.
- Largest observed gap reduction comes from: hours/job structure (0.95 percentage points).
- Top simple explained-gap contributors: actual_hours_last_week (6.96%); usual_hours_week (6.96%); multiple_jobholder (1.18%).
- What is missing: state geography in current release, rich family controls, commute, firm/pay-setting variables.
- Main reports: `reports/sipp_validation.md`, `reports/sipp_snapshot.md`, `reports/sipp_models.md`.

## ATUS

- Role: mechanism evidence on time allocation, not a wage regression.
- Key observed channels: paid work (-67.96 min/day); housework (+31.91); childcare (+11.37).
- What is missing: no direct wage equation, no person-level merge to ACS/CPS/SIPP.
- Main report: `reports/m6_time_use_bridge.md`.

## SCE

- Role: mechanism evidence on expectations, reservation wages, and outside options.
- Key observed channels: expected offer gap (+17.46); reservation wage gap (+21.34).
- Subgroup read: the gender gaps are positive in every public wave, but the education and income spreads are even larger than the gender spread.
- What is missing: cannot merge to main wage files; smaller expectation-based survey.
- Main reports: `reports/sce_supplement.md`, `reports/sce_public_analysis.md`, `reports/sce_subgroup_analysis.md`.

## NLSY Sub-Analysis

- Role: separate cohort lane with richer background and life-course variables than the cross-sectional public files, not just `g_proxy`.
- NLSY79: the biggest measured reductions come from occupation sorting (7.02 points) and the skills/traits block (5.62 points); final deep-model gap is 34.10%.
- NLSY97: the biggest measured reductions come from adult resources (6.60 points) and the skills/achievement block (1.65 points); final deep-model gap is 31.89%.
- Caution: the NLSY97 adult-resources block is later-life and mechanism-sensitive, so it should not be read as a clean pre-market explanation.
- Family-background reductions are small in both cohorts (0.04 and -0.06 points).
- Interpretation: NLSY is most useful as a richer cohort/background check. Skills matter, but occupation and later-life structure still matter too, and the gap remains after all blocks are added.
- Main report: `reports/nlsy_deep_dive.md`.

## Future Variables To Test

- 1. firm and employer effects: likely large residual source within occupation/industry.
- 2. tenure and work-history interruptions: important for promotion and wage progression.
- 3. direct bargaining / reservation wages: closest to ask-wage channel; SCE helps here.
- 4. schedule predictability / flexibility: can change job sorting and wage penalties.
- 5. within-occupation task specialization: broad job titles can hide meaningful role differences.

## Bottom line

Yes, the repo can already extract a useful answer to 'what in the data explains part of the gap?'. In the current files, occupation/industry job sorting is the biggest measured reduction in ACS and CPS ASEC 2023, while hours and job-structure variables are the biggest measured reduction in SIPP. Commute time matters where observed, and the biggest remaining blind spots are employer effects, tenure/interruption histories, direct bargaining measures, and finer within-job differences.
