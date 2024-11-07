import speech_recognition as sr
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import emoji
from keras.models import load_model
import librosa
import nltk
import re
from nltk.stem import WordNetLemmatizer
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import sounddevice as sd
import scipy.io.wavfile as wavfile
import threading






# Loading Encoders and Tokenizer
with open('text_tokenizer.pickle', 'rb') as fl:
    tokenizer=pickle.load(fl)

with open('text_Encoder.pickle', 'rb') as fd:
    txt_encoder=pickle.load(fd)

with open('speech_tokenizer.pickle', 'rb') as df:
    speech_encoder=pickle.load(df)


# Loading model
txt_model=load_model("text_emotion_model.keras")

speech_model=load_model("speechemotionmodel.keras")


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate):
    return librosa.effects.time_stretch(data,rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampl_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampl_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sampl_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result

def get_features(path):
    global sampl_rate
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sampl_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) 

    # data with stretching and pitching
    new_data = stretch(data,rate=0.8)
    data_stretch_pitch = pitch(new_data, sampl_rate,pitch_factor=2)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))

    return result

def speech_emotion(path):
    data=get_features(path)
    result1=speech_model.predict([data])
    result=speech_encoder.inverse_transform(result1)[0][0]
    return result



lemmatizer=WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"wasn't","was not",text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"n't", "not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"href", "",text)
    text = re.sub(r"shouldnt", "should not",text)
    text = re.sub(r"hadnt", "had not",text)
    text = re.sub(r"youve", "you have",text)
    text = re.sub(r"nofollow", "no follow",text)
    text = re.sub(r"hadn", "had in",text)
    text = re.sub(r"werent", "were not",text)
    text = re.sub(r"theyve", "they we",text)
    
    
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)


    # Tokenize the text
    words = nltk.word_tokenize(text)
     
    # Lemmatize and remove stop words
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    #words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    
    return ' '.join(words)


def txt_emotion(user_input):
    maxlen=66
    test=[clean_text(text) for text in user_input]
    
    test_seq = tokenizer.texts_to_sequences(test)
    Xtest = pad_sequences(test_seq, maxlen = maxlen, padding = 'post', truncating = 'post')

    y_preder = txt_model.predict(Xtest)
    y_pred = np.argmax(y_preder)
    em_result=txt_encoder.inverse_transform(y_preder)[0][0]
    return em_result

    #print(test[0] +" : "+ em_result + label_to_emoji(y_pred))



global uploaded_file
def get_result():
    user_input = text_box.get("1.0", END).strip()
    if user_input:
        emo = txt_emotion([user_input])
        result_box.config(state=NORMAL)
        result_box.delete("1.0", END)
        result_box.insert(END, f"Text Emotion: {emo} ")
        result_box.config(state=DISABLED)
    elif 'audio.wav' in globals():
        emo = speech_emotion('audio.wav')
        result_box.config(state=NORMAL)
        result_box.delete("1.0", END)
        result_box.insert(END, f"Speech Emotion: {emo[0]} ")
        result_box.config(state=DISABLED)
    elif uploaded_file:
        emo = speech_emotion(uploaded_file)
        result_box.config(state=NORMAL)
        result_box.delete("1.0", END)
        result_box.insert(END, f"Uploaded Audio Emotion: {emo[0]} ")
        result_box.config(state=DISABLED)
    else:
        result_box.config(state=NORMAL)
        result_box.delete("1.0", END)
        result_box.insert(END, "No input provided.")
        result_box.config(state=DISABLED)



def upload_audio():
    global uploaded_file
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        uploaded_file = file_path
        result_box.config(state=NORMAL)
        result_box.delete("1.0", END)
        result_box.insert(END, "Uploaded Successfully")
        result_box.config(state=DISABLED)

def record_audio():
    fs = 44100
    duration = 3
    result_box.config(state=NORMAL)
    result_box.delete("1.0", END)
    result_box.insert(END, "Recording...")
    result_box.config(state=DISABLED)

    def record():
        global audio_file
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        wavfile.write('audio.wav', fs, myrecording)
        audio_file = 'audio.wav'
        result_box.config(state=NORMAL)
        result_box.delete("1.0", END)
        result_box.insert(END, "Recording saved as audio.wav")
        result_box.config(state=DISABLED)


    # Use a separate thread to avoid blocking the main GUI thread
    threading.Thread(target=record).start()

def all_result():
    get_result()
    record_audio()
    upload_audio()

emotionwindow = Tk()
emotionwindow.geometry("874x494+50+50")
emotionwindow.title("Emotion Recognition from Text and Speech")

# Load and resize the image
image = Image.open("bg.jpeg")
image = image.resize((874, 494))  # Resize the image to match the window size
bgimage = ImageTk.PhotoImage(image)

# Create a canvas for the background image
canvas = Canvas(emotionwindow, width=874, height=494)
canvas.pack()

# Place the background image on the canvas
canvas.create_image(0, 0, image=bgimage, anchor=NW)

# Add title label
canvas.create_text(437, 60, text="Emotion Recognition from Text and Speech", fill="white", font=('times', 20, 'bold'))

# Create a frame to contain the text box and mic button
input_frame = Frame(emotionwindow)
input_frame.place(relx=0.5, rely=0.5, anchor="center")

# Create the text box with black border
text_box = Text(input_frame, width=40, height=5, bd=2, relief="solid")
text_box.pack(side=LEFT)

# Create the Google microphone symbol button (increased size)
mic_button = Button(input_frame, text="🎙", command=record_audio, font=("Arial", 20))
mic_button.pack(side=LEFT, padx=10)  # Added padding for better spacing

# Create the upload audio button
upload_button = Button(emotionwindow, text="Upload Audio", command=upload_audio)
upload_button.place(relx=0.5, rely=0.65, anchor="center")

# Create the result button
result_button = Button(emotionwindow, text="Get Result", command=get_result)
result_button.place(relx=0.5, rely=0.75, anchor="center")

# Create the result display box
result_box = Text(emotionwindow, width=50, height=5, state=DISABLED)
result_box.place(relx=0.5, rely=0.85, anchor="center")

emotionwindow.mainloop()