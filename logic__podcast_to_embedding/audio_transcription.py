# Speech to text - convert the mp3 files to text using openAI whisper
# This is actually what I used within my application

# https://medium.com/nlplanet/transcribing-youtube-videos-with-whisper-in-a-few-lines-of-code-f57f27596a55
# https://huggingface.co/spaces/SteveDigital/free-fast-youtube-url-video-to-text-using-openai-whisper (alternative)
import pytube as pt
import whisper

model = whisper.load_model("small")

# Outline the location of the files already downloaded
presidential_pods = {
    "recordings/phillips.mp3": "transcriptions/phillips.txt",
    "recordings/christie.mp3": "transcriptions/christie.txt",
    "recordings/vivek.mp3": "transcriptions/vivek.txt",
    "recordings/kennedy.mp3": "transcriptions/kennedy.txt"
}

for pod in presidential_pods:
    
    result = model.transcribe(pod)
    with open(presidential_pods[pod], 'w') as f:
        f.write(result["text"])