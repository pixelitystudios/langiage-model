from Pixelity.Model import LanguageModel
import tensorflow as tf

with open('./data/Q1.csv', 'r', encoding='utf8') as f:
    questions = f.read().split('\n')

with open('./data/Q2.csv', 'r', encoding='utf8') as f:
    answers = f.read().split('\n')

model = LanguageModel(path='./')
model.compile(questions=questions,answers=answers)
model.fit(epochs=1)
model.save_model()