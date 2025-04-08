# audioLM-as-judge

## Structure
Each experiment (e.g., lexical content evaluation on chatbot arena evaluation) is organized into a separate folder. Each folder (roughly) contains the following subfolders:
- `data`: Data used in the experiment
- `notebooks`: Jupyter notebooks for data processing and analysis
- `experiments`: Output cache files from running scripts
- `scripts`: Python scripts for running the experiments
----------------------------------------------------

## Experiments
### Part1: Assessing AudioLLM as a Judge
- `lexical-content-chatbot-arena`: Lexical content evaluation on chatbot arena evaluation
- `speech-quality`: Speech quality assessment, Mean Opinion score (MOS) score prediction (naturalness, intelligibility, etc)
- `paralinguistic`: Paralinguistic evaluation, including data synthesized from ElevenLabs and GPT-4o-Audio (based on content from ChatbotArena)

### Part2: Benchmarking Speech Generation Ability 
- `advanced-voice-gen-task-v1`: *Data preparation* for advanced voice generation task v1 (the first version of SpeakBench)
- `eval-leaderboard`: Run inference for existing speech-in speech-out systems on the data from `advanced-voice-gen-task-v1`, and evaluate the outputs in a pairwise setup (like AlpacaEval) using AudioLLM-as-a-Judge.

### Others
- `kokoroTTS`: Scripts to run kokoroTTS for various experiments
- `legacy-exp`: Legacy code for previous experiments