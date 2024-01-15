import pytube as pt
import whisper

# download mp3 from youtube video (Two Minute Papers)
#yt = pt.YouTube("https://www.youtube.com/watch?v=dd1kN_myNDs")
#stream = yt.streams.filter(only_audio=True)[0]
#stream.download(filename="audio_english.mp3")

model = whisper.load_model("base")

result = model.transcribe("audio_english.mp3")
print(result["text"])