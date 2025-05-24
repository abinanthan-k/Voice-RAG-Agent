import speech_recognition as sr

recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("Listening..")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said:\n", text)
    except sr.UnknownValueError:
        text = ""
        print("Sorry, could not understand audio")
    except sr.RequestError as e:
        text = ""
        print("Could not request results; {0}".format(e))
    return text

