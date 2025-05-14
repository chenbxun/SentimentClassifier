import torch
import pickle
import csv
from model import SentimentClassifier
import sys
sys.path.append(r"./data")
from data_u import handle_data, DATA_PATH
import os

if __name__ == '__main__':
    epoch = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f'save/model_epoch{epoch}.pkl', map_location=device)
    model = model.to(device)
    model.eval()  

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_val = pickle.load(inp)
        y_val = pickle.load(inp)

    sentences, labels = handle_data("test")

    predictions = []
    correct_predictions = 0
    
    for i, sentence in enumerate(sentences):
        x = [word2id.get(word, len(word2id)) for word in sentence]
        length = [len(x)]
        x = torch.LongTensor([x]).to(device)
        
        with torch.no_grad():
            log_probs = model(x, length)
            predict = log_probs.argmax(dim=1).item()
        
        original_sentence = ''.join(sentence)
        predictions.append((original_sentence, id2tag[predict]))

        if predict == labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(sentences)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    with open(os.path.join('data', DATA_PATH, 'predictions.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sentence', 'Predicted_Label'])  
        for sentence, label in predictions:
            writer.writerow([sentence, label])
