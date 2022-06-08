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
data_file = open(r'C:\Users\hicha\datascience\Rabat_Chatbot\rabat.json').read()
data = json.loads(data_file)

words=[]
tags = []
words_tags = []
ignore_words = ['?', '!','.','|','-','*','+','/','#','@','%','`']

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

# créer an object de WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# rendre miniscule puis lemmatiser les mots cles et supprimer les doublons
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# stocker les resultats de traitement de text(words et tags) a deux fichier pickle a utiliser lors de la prediction
pickle.dump(words,open('words.pkl','wb')) 
pickle.dump(tags,open('tags.pkl','wb'))
# creation des données d'entrainement
training = []

# creation d'un tableau vide pour le resultat [create an empty array for our output]
output_empty = [0] * len(tags)

# ensemble d'entraînement, bag (sac) de mots pour chaques phrase
for wrd in words_tags:
    # initialisation du bag (sac)
    bag = []
    # liste des mot cles tokenisés
    pattern_words = wrd[0]
   
    # lemmatisation des mots afin de representer les mots en relation
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # creer un tableau de mots (bag of words) et on y ajoute 1 si le mot est trouve dans le pattern actuel et 0 sinon
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # la sortie est 0 pour chaque autre tag et 1 pour le tag actuel (pour chaque pattern)
    output_row = list(output_empty)
    output_row[tags.index(wrd[1])] = 1
    training.append([bag, output_row])

# melanger les donnees et les convertir en un tableau numpy
random.shuffle(training)
training = np.array(training)

# ***************************************************************************************************************
# creer deux liste d'entrainement et de teste
train_input = list(training[:,0]) #contient les bags
train_output = list(training[:,1]) #contient les tags

# creation d'un modele de reseau de neurones pour predir les reponses
model = Sequential()
model.add(Dense(256, input_shape=(len(train_input[0]),), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(train_output[0]), activation='softmax'))

# Compile model. Adam donne des bonnes resultats pour ce modèle
# optimiser le modele en changeant les poids au fure et a mesure
optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#ici on commence notre training
history = model.fit(np.array(train_input), np.array(train_output), epochs=20, batch_size=64, verbose=1)
# on a enregistre le modele en format h5 (Hierarchical Data Format (HDF))
model.save('chatbot.h5', history) 

# *************************************************************************************************************

# on appele le modele enregistre
model = load_model('chatbot.h5')
data = json.loads(open(r'C:\Users\hicha\datascience\Rabat_Chatbot\rabat.json').read())
words = pickle.load(open('words.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenisation (separation des mots dnas un tableau) et lemmatisation aprés
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# retourne un tableau de mots: 0 ou 1 si le mot dans le bag existe dans la phrase
def bow(sentence, words):
    # tokenisation
    sentence_words = clean_up_sentence(sentence)
    # sac de mots, matrice de N mots, matrice de vocabulaire
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:               
                #  1 si le mot s figure dans le vocabulare
                bag[i] = 1
    return(np.array(bag))

def predict_tag(sentence, model): 
    p = bow(sentence, words) #tableau de 0 et 1 lorsque un mots figure dans le vocab
    res = model.predict(np.array([p]))[0]
    # print(res)
    error = 0.25
    # filtrer les predictions
    # le resultat est probable de plus de 25%
    results = [[i,r] for i,r in enumerate(res) if r>error]
    # print("results", results)

    # ordonner par probabilité dcroissante
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        # print("r " , r)
        return_list.append({"tag": tags[r[0]], "probability": str(r[1])})
    return return_list

# pour recevoir la reponse du modèle
def getResponse(tag_prob, intents_json):
    tag = tag_prob[0]['tag']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if(intent['tag']== tag):
            result = random.choice(intent['responses'])
            break
    return result

# pour predire le tag et recevoir la reponse

def chatbot_response(text):
    tag_prob = predict_tag(text, model)
    res = getResponse(tag_prob, data)
    return res
            
# ***********************************************************************************************************
# Demarrer le chatbot
# interface avec tkinter

import random
import tkinter as tk
from tkinter import *
# creer la fenetre
root=tk.Tk()
root.title(f"Ibn Battuta ChatBot")
root.geometry('500x400')
root.resizable(False, False)

message=tk.StringVar()
# styler le frame, c'est comme un container
chat_win=Frame(root,bd=1,bg='white',width=50,height=8)
#  gesrer la géométrie et organiser les widgets 
chat_win.place(x=6,y=6,height=300,width=488)
textcon=tk.Text(chat_win,bd=1,bg='white',width=50,height=8)
textcon.pack(fill="both",expand=True)
# pour ecrire un message
mes_win=Entry(root,width=30,xscrollcommand=True,textvariable=message)
mes_win.place(x=6,y=310,height=60,width=380)
mes_win.focus() # place le curseur dans le champ de text automatiquement

textcon.tag_config('usr',foreground='black')
textcon.insert(END,"Bot: This is Ibn Battuta! Your Personal Assistant.\nIf you want to know where to stay in Rabat or where to visit just ask me :)\nTO EXIT TYPE exit :(\n")

exit_list = ['exit','break','quit','see you later','chat with you later','end the chat','bye','ok bye']

def send_msz(event=None):
    usr_input = message.get().lower()
    textcon.insert(END, f'You: {usr_input}'+'\n','usr')
    if usr_input in exit_list:
        return root.destroy()
    else:
        textcon.config(fg='green')
        # message
        lab = f"Bot: {chatbot_response(usr_input)}"+'\n'
        textcon.insert(END,lab)
        mes_win.delete(0,END)

button_send=Button(root,text='Send',bg='dark green',activebackground='grey',command=send_msz,width=12,height=5,font=('Arial'))
button_send.place(x=376,y=310,height=60,width=110)
root.bind('<Return>', send_msz,button_send)
root.mainloop()
