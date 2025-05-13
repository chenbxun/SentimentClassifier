import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import SentimentClassifier
from dataloader import Sentence
import torch.nn as nn

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--output_dim', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cuda', action='store_true', default=False)
    return parser.parse_args()

def set_logger():
    log_file = os.path.join('save', 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_val = pickle.load(inp)
        y_val = pickle.load(inp)

    model = SentimentClassifier(len(word2id), args.embedding_dim, args.hidden_dim, args.output_dim, args.dropout)
    if use_cuda:
        model = model.cuda()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr)

    train_data = DataLoader(
        dataset=Sentence(x_train, y_train, vocab_size=len(word2id)),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=lambda batch: Sentence.collate_fn(batch, vocab_size=len(word2id)),
        drop_last=False,
    )

    val_data = DataLoader(
        dataset=Sentence(x_val, y_val, vocab_size=len(word2id)),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=lambda batch: Sentence.collate_fn(batch, vocab_size=len(word2id)),
        drop_last=False,
    )

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        for sentence, label, length in train_data:
            if use_cuda:
                sentence = sentence.cuda()
                label = label.cuda()
                mask = mask.cuda()

            # forward
            output = model(sentence, length) 
            loss = nn.NLLLoss()(output, label) 
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []

        # test
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for sentence, label, length in val_data:
                if use_cuda:
                    sentence = sentence.cuda()
                    label = label.cuda()
                output = model(sentence, length)
                pred = output.argmax(dim=1)
                total_correct += (pred == label).sum().item()
                total_samples += label.size(0)
            logging.info('Validation accuracy: %.4f' % (total_correct / total_samples))
        model.train()

        path_name = "./save/model_epoch" + str(epoch) + ".pkl"
        torch.save(model, path_name)
        logging.info("model has been saved in  %s" % path_name)


if __name__ == '__main__':
    set_logger()
    main(get_param())
