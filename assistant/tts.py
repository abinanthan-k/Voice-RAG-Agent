import pyttsx3

def initialize_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  
    engine.setProperty('volume', 0.9)  
    return engine

def speak(engine, text):
    engine.say(text)
    engine.runAndWait()
