# Dataset Collection Protocol

## Pilot Dataset

The first pilot dataset contains:

- 10 human essays
- 10 AI-generated essays
- 10 hybrid essays

Total: 30 essays.

The pilot dataset is intended for feature inspection and
pipeline validation. It is not large enough to support a
strong claim about detector accuracy.

## Topics

The pilot dataset covers five admissions-style topics:

1. Challenge / failure
2. Academic curiosity
3. Leadership
4. Community involvement
5. Personal growth

Each category should contain examples from multiple topics.

## Human Essays

Human essays must come from:

- appropriately licensed/public sources, or
- contributors who explicitly permit their writing to be
  used for this research project.

Private admissions essays must not be collected without
appropriate permission.

## AI Essays

AI-generated essays must record:

- model
- generation prompt
- topic
- generation date/version when available
- requested approximate length

The detector must not use the model's own judgement as the
classification result.

## Hybrid Essays

Hybrid essays begin with a human-written source.

The AI is then used to modify the text.

The original human version must be preserved.

The modification level should be recorded as:

- light
- moderate
- heavy

The relationship between the original and modified versions
is represented using `source_group`.

## Data Leakage

Related documents must remain in the same dataset split.

For example:

human_001
hybrid_001

must never be separated between training and testing.

## Evaluation

The pilot dataset is not the final evaluation set.

A later held-out test set will be created and kept separate
from model development.

The final evaluation will report:

- accuracy
- precision
- recall
- F1
- confusion matrix
- confidently incorrect examples
- false-positive analysis
- false-negative analysis
- English-language-background error analysis

## Limitations

The dataset will not represent every form of human writing
or every language model.

Results may vary with:

- topic
- essay length
- author
- writing proficiency
- English-language background
- generation model
- prompting strategy
- amount of AI intervention