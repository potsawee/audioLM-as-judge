# Experiment: Speech Qaulity Assessment
- Speech Quality Assessment, Mean Opinion score (MOS) score prediction (natruralness, intelligibility, etc)
- Data derived from SOMOS, TMHINT, ThaiMOS

## Structure

### data
data

### notebooks
Jupyter notebooks for data processing and analysis

### experiments
Output cache files from running scripts

### scripts
```
{exp_name}_{dataset}_{judge_llm}_{input_modal}_{prompt}.py
```
- `exp_name`: exp1 = pairwise judge (i.e., A/B testing); exp2 = pointwise (inital results were not great, and I didn't investigate further)
- `dataset`: dataset name (e.g., sommos, tmhintqi, thaimos)
- `judge_llm`: judge LLM (e.g., gpt, qwen2, typhoon2, etc)
- `input_modal`: input modality (e.g., text, audio). There is no output as this task involves assessment the quality of TTS synthesized speech
- `prompt`: prompting method / experimental setup, e.g., multiturn (showing speechA and speechB in different turns), cot (CoT promtping); vanilla (most basic prompting).

For example:
- SOMOS, A/B testing
    - all pairs where diff > 0.0 (3567 pairs in total) -- to be continued
    ```
    python scripts/exp1_somos_gpt_multiturn.py --data data/data_somos_pairwise_diffall.json --output_path experiments/somos/ab_testing/diffall_prompt2.txt --order ab --message_format 2
    python scripts/exp1_somos_gpt_multiturn.py --data data/data_somos_pairwise_diffall.json --output_path experiments/somos/ab_testing/diffall_prompt2_BA.txt --order ba --message_format 2
    ```
### legacy
Old scripts and notebooks that are not used in the current experiments. These are kept for reference and may be useful for future experiments.