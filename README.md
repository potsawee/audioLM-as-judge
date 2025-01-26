# audioLM-as-judge


# Experiments
- exp{1/2}\_{dataset}\_{model}\_{variant}.py
    - exp1 = pairwise
    - exp2 = pointwise

- SOMOS, A/B testing
    - all pairs where diff > 0.0 (3567 pairs in total) -- to be continued
    ```
    python exp1_somos_gpt_multiturn.py --data data/data_somos_pairwise_diffall.json --output_path experiments/somos/ab_testing/diffall_prompt2.txt --order ab --message_format 2
    python exp1_somos_gpt_multiturn.py --data data/data_somos_pairwise_diffall.json --output_path experiments/somos/ab_testing/diffall_prompt2_BA.txt --order ba --message_format 2
    ```