import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentimentClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super(SentimentClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # 嵌入层：将输入的词汇索引转化为嵌入向量
        self.word_embeds = nn.Embedding(vocab_size + 2, embedding_dim) # +2 是因为不在词典里的词索引为vocab_size，填充值为vocab_size+1
        # LSTM层：捕捉文本中的上下文信息
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 全连接层：将LSTM的输出映射到情感类别空间
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Softmax层：计算属于各个类别的概率
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))
    
    def forward(self, sentence, length):
        batch_size = sentence.size(0)
        self.hidden = self.init_hidden(batch_size, sentence.device)
        embeds = self.word_embeds(sentence)
        # 打包序列以处理不同长度的输入
        packed_input = pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)
        lstm_out, self.hidden = self.lstm(packed_input, self.hidden)
        # 解包序列
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # 使用最后一个时间步的隐藏状态作为句子表示
        last_hidden_state = self.hidden[0].transpose(0, 1).reshape(batch_size, -1) # 输出维度：(batch_size, hidden_dim)
        out = self.fc(self.dropout(last_hidden_state))
        log_probs = self.softmax(out)
        return log_probs
