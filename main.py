import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from googletrans import Translator
import random
import time
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

translator = Translator()

sia = SentimentIntensityAnalyzer()

def translate_to_english(text):
    return translator.translate(text, src="tr", dest="en").text

def translate_to_turkish(text):
    return translator.translate(text, src="en", dest="tr").text

def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Pozitif"
    elif score['compound'] <= -0.05:
        return "Negatif"
    else:
        return "Nötr"
    
def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Günaydın"
    elif hour < 18:
        return "İyi öğleden sonralar"
    else:
        return "İyi akşamlar"

def trim_history(history, max_length=3):
    return history[-max_length:]

def generate_response(input_text, history=None):
    if history is None:
        history = []
    
    if "adın ne" in input_text.lower():
        return "Benim adım Chatbot. Size nasıl yardımcı olabilirim?", history
    elif "nasılsın" in input_text.lower():
        return "Ben bir yapay zeka olduğum için duygularım yok, ama size yardımcı olmak için buradayım!", history

    sentiment = analyze_sentiment(input_text)
    
    english_text = translate_to_english(input_text)
    
    history = trim_history(history)
    
    full_input = " ".join(history) + " " + english_text
    
    inputs = tokenizer(full_input, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=3, no_repeat_ngram_size=2)
    
    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(response)
    
    selected_response = random.choice(responses)
    
    turkish_response = translate_to_turkish(selected_response)
    
    history.append(english_text)
    history.append(selected_response)
    
    return f"{get_greeting()}, Duygu Durumunuz: {sentiment}. {turkish_response}", history

def chat():
    print("Chatbot'a hoş geldiniz! Çıkmak için 'çıkış' yazabilirsiniz.")
    history = []
    while True:
        user_input = input("Siz: ")
        if user_input.lower() == "çıkış":
            print("Chatbot: Görüşmek üzere!")
            break
        
        response, history = generate_response(user_input, history)
        print("Chatbot:", response)

if __name__ == "__main__":
    chat()