import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentimentClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super(SentimentClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # Ƕ��㣺������Ĵʻ�����ת��ΪǶ������
        self.word_embeds = nn.Embedding(vocab_size + 2, embedding_dim) # +2 ����Ϊ���ڴʵ���Ĵ�����Ϊvocab_size�����ֵΪvocab_size+1
        # LSTM�㣺��׽�ı��е���������Ϣ
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        # Dropout�㣺��ֹ�����
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # ȫ���Ӳ㣺��LSTM�����ӳ�䵽������ռ�
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Softmax�㣺�������ڸ������ĸ���
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))
    
    def forward(self, sentence, length):
        batch_size = sentence.size(0)
        self.hidden = self.init_hidden(batch_size, sentence.device)
        embeds = self.word_embeds(sentence)
        # ��������Դ���ͬ���ȵ�����
        packed_input = pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)
        lstm_out, self.hidden = self.lstm(packed_input, self.hidden)
        # �������
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # ʹ�����һ��ʱ�䲽������״̬��Ϊ���ӱ�ʾ
        last_hidden_state = self.hidden[0].transpose(0, 1).reshape(batch_size, -1) # ���ά�ȣ�(batch_size, hidden_dim)
        out = self.fc(self.dropout(last_hidden_state))
        log_probs = self.softmax(out)
        return log_probs
