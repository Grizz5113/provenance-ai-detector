# Human Essay Collection Protocol

## Purpose

Human essays provide the human-writing examples required
for training and evaluating the Provenance detector.

## Eligibility

Each essay must be:

- written primarily by a human;
- voluntarily provided or appropriately licensed;
- suitable for use in this research project;
- an admissions-style essay or response to an equivalent
  personal/reflective prompt.

Private admissions essays must not be collected without
appropriate permission.

## Required Metadata

Every human essay records:

- essay ID
- source
- topic
- source group
- author language background, when voluntarily provided
- notes

## Privacy

Do not collect:

- real names unless necessary;
- addresses;
- phone numbers;
- email addresses;
- student IDs;
- application IDs;
- other unnecessary personal information.

Essays should be anonymized before entering the dataset.

## Language Background

If voluntarily provided, language background may be recorded
for later error analysis.

It must NOT be used as an input feature for the detector.

The purpose is to determine whether the detector produces
disproportionate false positives for writers who learned
English as a second language.

## Minimum Length

Human essays should contain at least 50 words.

The target length for the pilot is approximately 500–700 words,
but shorter naturally occurring examples may be retained
during exploratory analysis if their limitations are documented.

## Source Groups

Human essays that will later be transformed into hybrid
examples must receive a unique source group.

Example:

human_001
source_group = group_001

The corresponding hybrid version will use:

hybrid_001
source_group = group_001