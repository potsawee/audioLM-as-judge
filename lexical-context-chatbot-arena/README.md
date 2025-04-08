# Experiment: Chatbot Arena (Lexical Content Evaluation)
- Lexical content evaluation
- Speech-based ChatbotArena (i.e., speech is synthesized by TTS) but the contents were taken from the original Chatbot Arena

## Structure

### data
data

### notebooks
Jupyter notebooks for data processing and analysis

### experiments
Output cache files from running scripts

### scripts
```
{exp_name}_{judge_llm}_{input_modal}_{output_modal}_{...}.py
```
- `exp_name`: exp1 chatbot arena = pairwise chatbot arena (there is only one exp at the moment)
- `judge_llm`: judge LLM (e.g., gpt, qwen2, typhoon2, etc)
- `input_modal`: input modality (e.g., text, audio). text = ground-truth text, asr = asr transcript, audio = TTS synthesized speech
- `output_modal`: output modality (e.g., text, audio). text = ground-truth text, asr = asr transcript, audio = TTS synthesized speech
- `...`: other additional experiments (e.g., `AWGN` = Additive White Gaussian Noise ablation, `style_only` = Style bias experiment)

### Others
`process_chatbot_arena`: old code for processing chatbot arena data