from transformers import AutoProcessor, SeamlessM4TModel
from IPython.display import Audio
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium", use_fast=False)
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

from datasets import load_dataset
# dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
dataset = load_dataset("eugenetanjc/speech_accent_english_100")
audio_sample = dataset["train"][0]
audio_inputs = processor(audios=audio_sample["audio"], return_tensors="pt")
print(type(audio_inputs))
Audio(audio_inputs['input_features'][0],rate=44100)
output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

from torchmetrics.text import WordErrorRate
metric = WordErrorRate()
print(translated_text_from_audio)
print(dataset["train"][0]["text"])
