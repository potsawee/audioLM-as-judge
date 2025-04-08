


# Experiment: Paralinguistic Assessment
- Assessing the abilities of Audio LLMs as judges for paralinguistic features
- Speech-based ChatbotArena (i.e., speech is synthesized by TTS) but using speech generation systems (ElevenLabs or GPT-4o-Audio) to generate speech with paralinguistic features. Then judges are assessed whether they can detect which speech contains the paralinguistic features (e.g., style, emotion, etc)

## Structure

### data
- `data-gpt-style`: Paralinguistic **data preparation** using GPT-4o-Audio
- `data-elevenLabs`: Paralinguistic **data preparation** using ElevenLabs API

### notebooks
Jupyter notebooks for data processing and analysis

### experiments
Output cache files from running scripts

### scripts
```
{exp_name}_{judge_llm}_{input_modal}_{output_modal}_{type}.py
```
- `exp_name`: exp1_chatbotarenastyle = pairwise experiment where the contents (text) are taken from the original Chatbot Arena
- `judge_llm`: judge LLM (e.g., gpt, qwen2, typhoon2, etc)
- `input_modal`: input modality -- only `audio` = synthesized speech
- `output_modal`: output modality -- only `audio` = synthesized speech
- `type`: `styleonly` to assess the paralinguistic (instead of the lexical content)
