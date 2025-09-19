Instructions to Set Up Offline Blacknox Assistant

1. Download GPT-2 Small Model files from:
   https://huggingface.co/gpt2/tree/main

2. Place these files into the folder named 'your-model/':
   - config.json
   - merges.txt
   - pytorch_model.bin
   - special_tokens_map.json
   - tokenizer_config.json
   - vocab.json

3. Install dependencies:
   pip install -r requirements.txt

4. Run the assistant:
   python agent_offline.py

No internet is required after the model files are in place.