# importer les bibliotheques necessires
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from keras.models import load_model



#importer la base de connaissances pour le pré-traitement des données

words=[]
tags = []
words_tags = []
ignore_words = ['?', '!']
data_file = open(r'Rabat_Chatbot/rabat.json').read()
data = json.loads(data_file)
# preprocessing the json data
# tokenization

for intent in data['intents']:
    for pattern in intent['patterns']:
        #tokenisation de chaque mot
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        #ajout des tokens (mots cles) et le tag correspondant a words_tags
        words_tags.append((tokens, intent['tag']))
        # ajouter le tag actuelle a la liste des tags
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# creer an object de WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# rendre miniscule puis lemmatiser les mots cles et supprimer les doublons
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# creating a pickle file to store the Python objects which we will use while predicting
# stocker les resultats de traitement de text(words et tags) a deux fichier pickle
pickle.dump(words,open('words.pkl','wb')) 
pickle.dump(tags,open('tags.pkl','wb'))
# creation des données d'entrainement
training = []

# creation d'un tableau vide pour le resultat [create an empty array for our output]
output_empty = [0] * len(tags)

# ensemble d'entraînement, bag (sac) de mots pour chaques phrase
for wrd in words_tags:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = wrd[0]
   
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[tags.index(wrd[1])] = 1
    training.append([bag, output_row])

# shuffle features and converting it into numpy arrays
# melanger les donnees et les convertir en un tableau numpy
random.shuffle(training)
training = np.array(training)

# creer deux liste d'entrainement et de teste
train_input = list(training[:,0]) #input
train_output = list(training[:,1]) #output

print("Training data created")
# creation d'un modele de reseau de neurones pour predir les reponses
model = Sequential()
model.add(Dense(256, input_shape=(len(train_input[0]),), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(train_output[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# Compile model. Adam donne des bonnes resultats pour ce modèle
optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#fitting and saving the model 
history = model.fit(np.array(train_input), np.array(train_output), epochs=20, batch_size=64, verbose=1)
model.save('chatbot.h5', history) # we will pickle this model to use in the future
print("\nModèle crée avec succés!")

# appeler le modele qu'on a enregistré
model = load_model('chatbot.h5')
data = json.loads(open(r'Rabat_Chatbot/rabat.json').read())
words = pickle.load(open('words.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenisation (separation des mots dnas un tableau) et lemmatisation aprés
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words):
    # tokenisation
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:               
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model): 
    # filter out predictions below a threshold
    p = bow(sentence, words)
    print("predict type: ", model.predict(np.array([p])))
    res = model.predict(np.array([p]))[0]
    print(res)
    error = 0.25
    # le resultat est probable de plus de 25%
    results = [[i,r] for i,r in enumerate(res) if r>error]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": tags[r[0]], "probability": str(r[1])})
    return return_list
# function to get the response from the model

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if(intent['tag']== tag):
            result = random.choice(intent['responses'])
            break
    return result

# function to predict the class and get the response

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, data)
    return res
# function to start the chat bot which will continue till the user type 'end'

def start_chat():
    print("Bot: This is Tourism! Your Virtual Assistant.\n\n")
    while True:
        inp = str(input()).lower()
        if inp.lower()=="end":
            break
        if inp.lower()== '' or inp.lower()== '*':
            print('Please re-phrase your query!')
            print("-"*50)
        else:
            print(f"Bot: {chatbot_response(inp)}"+'\n')
            print("-"*50)
            
# start the chat bot
# start_chat()
import random
import tkinter as tk
from tkinter import *

root=tk.Tk()
root.title(f"Ibn Battuta ChatBot")
root.geometry('500x400')
root.resizable(False, False)
message=tk.StringVar()

chat_win=Frame(root,bd=1,bg='white',width=50,height=8)
chat_win.place(x=6,y=6,height=300,width=488)
textcon=tk.Text(chat_win,bd=1,bg='white',width=50,height=8)
textcon.pack(fill="both",expand=True)
# ///////////////////////////
mes_win=Entry(root,width=30,xscrollcommand=True,textvariable=message)
mes_win.place(x=6,y=310,height=60,width=380)
mes_win.focus()

textcon.config(fg='black')
textcon.tag_config('usr',foreground='black')
textcon.insert(END,"Bot: This is Ibn Battuta! Your Personal Assistant.\nIf you want to know where to stay in Rabat or where to visit just ask me :)\n")
mssg=mes_win.get()

exit_list = ['exit','break','quit','see you later','chat with you later','end the chat','bye','ok bye']

def greet_res(text):
    text=text.lower()
    bot_greet=['hi there','hello there','hey there']
    usr_greet=['hi','hey','hello','bonjour','greetings','salut','wassup','whats up']
    for word in text.split():
        if word in usr_greet:
            return random.choice(bot_greet)

def send_msz(event=None):
    usr_input = message.get().lower()
    textcon.insert(END, f'You: {usr_input}'+'\n','usr')
    if usr_input in exit_list:
        return root.destroy()
    else:
        textcon.config(fg='green')
        if greet_res(usr_input) != None:
            lab=f"Bot: {greet_res(usr_input)}"+'\n'
            textcon.insert(END,lab)
            mes_win.delete(0,END)
        else:
            lab = f"Bot: {chatbot_response(usr_input)}"+'\n'
            textcon.insert(END,lab)
            mes_win.delete(0,END)

button_send=Button(root,text='Send',bg='dark green',activebackground='grey',command=send_msz,width=12,height=5,font=('Arial'))
button_send.place(x=376,y=310,height=60,width=110)
root.bind('<Return>', send_msz,button_send)
root.mainloop()