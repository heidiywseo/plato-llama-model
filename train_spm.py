import sentencepiece as spm

def train_sentencepiece(input_file, model_prefix, vocab_size = 100):
    spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix, vocab_size=vocab_size)

def tokenize_text(model_prefix, input_file):
    sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
    with open(input_file, 'r') as f:
        data = f.read().splitlines()
    tokenized_data = [sp.encode(line, out_type=int, add_bos=True, add_eos=True) for line in data]
    return tokenized_data, sp.pad_id()

input_file = 'plato.txt'
model_prefix = 'tokenizer'
train_sentencepiece(input_file, model_prefix)
tokenized_data, pad_id = tokenize_text(model_prefix, input_file)

import pickle
with open('tokenized_data.pkl', 'wb') as f:
    pickle.dump((tokenized_data, pad_id), f)

