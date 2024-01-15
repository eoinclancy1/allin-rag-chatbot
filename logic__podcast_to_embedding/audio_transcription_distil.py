# Speech to text using a seemingly faster model - upon testing though, it seemed to take longer than the whisper-small model

#https://huggingface.co/distil-whisper/distil-large-v2
#https://twitter.com/alphasignalai/status/1724498003585900912?s=46
#https://huggingface.co/spaces/SteveDigital/free-fast-youtube-url-video-to-text-using-openai-whisper (alternative)
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe("audio_english.mp3")
print(result["text"])

with open('readme.txt', 'w') as f:
    f.write(result["text"])


presidential_pods = {
    "recordings/phillips.mp3": "transcriptions/phillips.txt",
    "recordings/christie.mp3": "transcriptions/christie.txt",
    "recordings/vivek.mp3": "transcriptions/vivek.txt",
    "recordings/kennedy.mp3": "transcriptions/kennedy.txt"
}

for pod in presidential_pods:
    result = pipe(pod)
    with open(presidential_pods[pod], 'w') as f:
        f.write(result["text"])