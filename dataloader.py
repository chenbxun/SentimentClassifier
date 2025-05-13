import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Sentence(Dataset):
    def __init__(self, x, y, vocab_size, batch_size=10):
        self.x = x
        self.y = y
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @staticmethod
    def collate_fn(train_data, vocab_size): 
        # train_data: [batch_size, (input_sequence, target_sequence)]
        train_data.sort(key=lambda data: len(data[0]), reverse=True)

        data_x = [torch.LongTensor(data[0]) for data in train_data]
        data_y = [data[1] for data in train_data] 

        data_length = [len(x) for x in data_x]
        data_x = pad_sequence(data_x, batch_first=True, padding_value=vocab_size+1)
        data_y = torch.LongTensor(data_y)
        return data_x, data_y, data_length

if __name__ == '__main__':
    # test
    with open('./data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_val = pickle.load(inp)
        y_val = pickle.load(inp)
    
    vocab_size = len(word2id)
    dataset = Sentence(x_train, y_train, vocab_size=vocab_size)

    train_dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=lambda batch: Sentence.collate_fn(batch, vocab_size=vocab_size)
    )

    for input, label, length in train_dataloader:
        print(input, label, length)
        break