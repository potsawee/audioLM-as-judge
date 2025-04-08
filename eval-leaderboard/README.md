# Experiment: Speech-in Speech-out leaderboard (based on AudioLLM-as-a-Judge) like AlpacaEval
- Data is derived from the `advanced-voice-gen-task-v1` experiment

## Structure

### notebooks
Jupyter notebooks for data processing and analysis

### experiments
- Inference output cache files from running scripts
- Judge output cache files from running scripts

### scripts
- `inference`: to run inference on the data (from the `advanced-voice-gen-task-v1` experiment)
- `evaluate`: to run AudioLLM-as-a-Judge on the generated speech (from the `inference` step)