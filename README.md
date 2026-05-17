# MetS Risk Score - Multimodal Prediction in Young Adults

## Research Question
Can we predict continuous Metabolic Syndrome severity in young adults (18–45)
using secondary biochemistry markers and multimodal lifestyle data, without
relying on the five standard diagnostic components?

## Data Source
NHANES 2017–March 2020 Pre-Pandemic File (CDC)

## Domains
- Secondary Biochemistry (liver, kidney, electrolytes, CBC)
- Dietary Intake
- Physical Activity
- Sleep
- Mental Health (PHQ-9)
- Smoking & Alcohol
- Anthropometrics (non-leakage)

## Target Variable
Continuous MetS Severity Score (DeBoer-Gurka, sex/race-specific)

## Methods
MICE Imputation · Stratified K-Fold CV · NHANES Survey Weights
ElasticNet · XGBoost · LightGBM · SHAP Explainability · Calibration Analysis

## Status
🔧 In Progress

## Author
Jayeshkumar Narsingani — M.S. Applied Mathematics
