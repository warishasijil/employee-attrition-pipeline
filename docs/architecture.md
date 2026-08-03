# Employee Attrition Pipeline Architecture

## Overall Flow

```text
main.py
    ↓
EmployeeAttritionPipeline
    ↓
Config
    ↓
DataLoader
    ↓
DataValidator
    ↓
DataCleaner
    ↓
EDAAnalyzer
    ↓
FeatureEngineer
    ↓
Preprocessor
    ↓
ModelTrainer
    ↓
ModelEvaluator
    ↓
ModelSaver
    ↓
Predictor