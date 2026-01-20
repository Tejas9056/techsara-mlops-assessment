# techsara-mlops-assessment
AI/ML Engineer Technical Assessment – MLOps pipeline with conditional model promotion

# AI/ML Engineer – Technical Assessment (MLOps Extension)

## Project Overview
This project demonstrates a reproducible machine learning workflow with evaluation
and a conditional model promotion mechanism following MLOps best practices.

## What I Implemented
I implemented an end-to-end ML pipeline using a Random Forest classifier. The model
is evaluated using F1-score and compared against a baseline production model stored
in a JSON file. The new model is promoted only if it meets or exceeds the baseline
performance threshold.

## How to Run
1. Install dependencies:
   pip install scikit-learn pandas

2. Run the pipeline:
   python prediction.py

## Assumptions & Limitations
The production model registry is simulated using a local JSON file. Cloud deployment,
CI/CD pipelines, and monitoring are outside the scope of this assessment.

## Reflection

Using a coding assistant helped me significantly speed up development, especially
when structuring the ML pipeline and implementing the conditional deployment logic.
It was particularly useful for debugging errors and understanding MLOps best
practices conceptually. In a few cases, the assistant’s suggestions required manual
adjustments to suit the execution environment, which highlighted the importance of
developer judgment. Overall, the assistant improved productivity while still requiring
me to validate logic and decisions carefully.

