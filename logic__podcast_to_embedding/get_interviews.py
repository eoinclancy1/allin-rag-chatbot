#Given youtube links, grab the raw .mp3 file from youtube

import pytube as pt

# https://medium.com/nlplanet/transcribing-youtube-videos-with-whisper-in-a-few-lines-of-code-f57f27596a55 
presidential_pods = {
    "phillips.mp3": "https://www.youtube.com/watch?v=1hh8lcoJ1NA&t",
    "christie.mp3": "https://www.youtube.com/watch?v=odWe7qsrrGk",
    "vivek.mp3": "https://www.youtube.com/watch?v=mpC6c6iYji8",
    "kennedy.mp3": "https://www.youtube.com/watch?v=nA0OXZuaG0g"
}

# download mp3 from youtube video (Two Minute Papers)
for pod in presidential_pods:
    yt = pt.YouTube(presidential_pods[pod])
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename=pod)
