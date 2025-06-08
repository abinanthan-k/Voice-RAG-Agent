import speech_recognition as sr

def listen_to_user(wav_io):
    ans = ""
    r = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio = r.record(source)
    try:
        ans = r.recognize_google(audio)
    except Exception:
        print('Sorry, I could not understand. Could you please say that again?')
    print(f"You said: {ans}")
    return ans

