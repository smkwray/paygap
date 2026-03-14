# SCE Supplemental Expectations and Reservation-Wage Note

## Why this module exists

ACS, CPS ASEC, and ATUS can estimate realized earnings gaps, employment-selection sensitivity, and time-use channels, but they do not observe the wage workers say they would require before accepting a new job.
The New York Fed Survey of Consumer Expectations (SCE) Labor Market Survey is the best free public dataset in this project scope for that narrower question because it directly elicits reservation wages, expected offer wages, offer receipt, and offer acceptance outcomes.

## Bottom line for this repo

- Use SCE as supporting evidence on bargaining thresholds, outside options, and expectations.
- Do not treat SCE as a drop-in control variable inside the main ACS/CPS wage regressions.
- The main reason is identification, not convenience: SCE is a separate survey with a different sampling frame, different timing, and expectation-based measures that cannot be person-level merged onto ACS/CPS/SIPP.

## Official survey facts

- The SCE launched in 2013 as a New York Fed survey on expectations about inflation, labor markets, and household finance: https://www.newyorkfed.org/microeconomics/sce
- The Labor Market Survey module has been fielded since March 2014, first released in August 2017, and is fielded every four months in March, July, and November: https://www.newyorkfed.org/microeconomics/sce/sce-faq
- The Labor Market page states that the module collects experiences and expectations related to earnings, job transitions, and job offers, and offers chart data, a guide, questionnaire, and complete microdata downloads: https://www.newyorkfed.org/microeconomics/sce/labor
- Recent New York Fed releases describe the labor module as surveying about 1,000 panelists each wave and explicitly tracking reservation wages, expected offers, and realized offers: https://libertystreeteconomics.newyorkfed.org/2024/08/an-update-on-the-reservation-wages-in-the-sce-labor-market-survey/

## Why SCE matters for the current findings

- The rebuilt ACS year-by-year `M5` gap averages 13.77% and sits at 13.23% in 2023. That is a realized worker-gap result, not a bargaining-threshold result.
- ACS selection robustness shows a much larger combined expected annual-earnings gap (38.48% mean in `S2`) than the IPW worker hourly gap (19.73%).
- CPS selection shows the same pattern: combined expected annual-earnings gap 31.99% versus IPW worker hourly gap 19.25%.
- ATUS adds mechanism evidence in the same direction, with women averaging 68.0 fewer paid-work minutes per day, 31.9 more housework minutes, and 11.4 more childcare minutes.

Those results already show that realized earnings, hours, and labor-force attachment differ by sex. SCE complements them by getting closer to what workers expect, what offers they receive, and what minimum pay they say they require.

## What SCE can and cannot identify

### What it can add

- Whether women report systematically lower reservation wages than men.
- Whether women expect lower offer wages even before observed realized wage differences are measured.
- Whether offer receipt, acceptance, or rejection patterns differ by sex.
- Whether post-2020 changes in reservation wages track or diverge from realized wage-gap patterns.

### What it cannot do cleanly

- It cannot be merged person-by-person onto ACS, CPS ASEC, or ATUS.
- It cannot retroactively become a control in the current ACS/CPS wage regressions.
- It should not be used to claim that the adjusted gap disappears once we 'control for bargaining' unless the bargaining analysis is run inside SCE itself.

## Recommended empirical use

1. Keep ACS/CPS/ATUS as the main realized-gap evidence.
2. Use SCE as a separate supplemental section on expectations, reservation wages, and job-offer dynamics.
3. If SCE microdata are downloaded later, estimate sex differences in reservation wages and offer expectations within SCE itself, then compare their direction and magnitude to the ACS/CPS worker-gap results.
4. Present any SCE findings as mechanism or calibration evidence, not as the final adjusted-gap estimand.

## Measure map

### Reservation Wage
- What it measures: Lowest wage or salary the respondent says they would accept for a job they would consider.
- Why it matters: Closest public survey measure to the wage a worker says they require before accepting a new job.
- Best use here: Supporting evidence on bargaining thresholds and outside options.
- Merge directly into ACS/CPS: No
- Why not: Different survey, different sample, different cadence, and no crosswalk to ACS/CPS person records.
- Source: https://www.newyorkfed.org/medialibrary/media/research/microeconomics/interactive/downloads/sce-labor-questionnaire.pdf

### Expected Offer Wage
- What it measures: Expected wage or salary of future offers, conditional on receiving an offer.
- Why it matters: Captures what workers think the market will pay, which is distinct from realized current wages.
- Best use here: Expectation benchmark beside ACS/CPS realized worker gaps.
- Merge directly into ACS/CPS: No
- Why not: Expectation variable from a separate rotating module, not observed in ACS/CPS wage files.
- Source: https://www.newyorkfed.org/microeconomics/sce/labor

### Offer Receipt And Acceptance
- What it measures: Number of offers received, offer wages, and whether offers were accepted, rejected, or still under consideration.
- Why it matters: Lets the project discuss realized job-offer dynamics instead of treating bargaining as purely hypothetical.
- Best use here: Mechanism evidence on job search and offer conversion.
- Merge directly into ACS/CPS: No
- Why not: Observed over the last four months in a separate New York Fed panel, not in ACS/CPS.
- Source: https://www.newyorkfed.org/medialibrary/Interactives/sce/sce/downloads/data/SCE-Labor-Market-Survey-Data-Codebook.pdf?sc_lang=en

### Job Offer Expectations
- What it measures: Percent chance of receiving at least one offer and expected number of offers over the next four months.
- Why it matters: Measures perceived opportunity set, which can shape ask wages and acceptance thresholds.
- Best use here: Context for interpreting gender differences in outside options or bargaining leverage.
- Merge directly into ACS/CPS: No
- Why not: Forward-looking expectations from a smaller survey, not a realized wage control in the main files.
- Source: https://www.newyorkfed.org/microeconomics/sce/labor

### Job Search Intensity
- What it measures: Whether the respondent searched in the last four weeks, search channels used, and hours spent searching.
- Why it matters: Helps separate bargaining claims from search effort and labor-market attachment.
- Best use here: Supporting mechanism evidence next to ACS/CPS selection robustness.
- Merge directly into ACS/CPS: No
- Why not: Search-intensity module is not available person-for-person in ACS/CPS.
- Source: https://www.newyorkfed.org/medialibrary/media/research/microeconomics/interactive/downloads/sce-labor-questionnaire.pdf

### Retirement And Transition Expectations
- What it measures: Probabilities of future employment states and working beyond ages 62 and 67.
- Why it matters: Adds forward-looking labor-supply expectations that can differ from realized annual earnings.
- Best use here: Supplemental interpretation for extensive-margin and lifecycle channels.
- Merge directly into ACS/CPS: No
- Why not: Expectation measures live in SCE only and are not aligned to ACS/CPS survey timing.
- Source: https://www.newyorkfed.org/microeconomics/sce/sce-faq

## Download and staging

- The repo downloader writes acquisition instructions into `data/raw/sce_labor_market`.
- Databank / microdata access: https://www.newyorkfed.org/microeconomics/databank.html
- Questionnaire: https://www.newyorkfed.org/medialibrary/media/research/microeconomics/interactive/downloads/sce-labor-questionnaire.pdf
- Codebook: https://www.newyorkfed.org/medialibrary/Interactives/sce/sce/downloads/data/SCE-Labor-Market-Survey-Data-Codebook.pdf?sc_lang=en

## Practical interpretation

If SCE shows lower female reservation wages or lower expected offer wages, that would support a bargaining/expectations channel. It still would not imply that the realized ACS/CPS worker gap is explained away; it would imply that part of the mechanism may run through expectations and outside options rather than only realized job sorting or hours.

That is the defensible way to use SCE in this project.
