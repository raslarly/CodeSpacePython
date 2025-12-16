"""
problem tanimi: lstm ile metin turetme
"""

# lstm malakia

import torch
import torch.nn as nn
import torch.optim as optim
# kelime frekanslarini hesaplamak icin import edildi
from collections import Counter
# grid search icin kombinasyon olusturmak amaci ile import edildi
from itertools import product

# %% veri yukleme ve on isleme (preprocessing)

# forumda olan metin
text = """
I really do not understand any complex editors.
Notepad or nano are actually fine with me.
Only recently I started requiring line number on every line, 
perhaps the old time returning to me when we had a line number in positions 
of 72-80 of a punch card. I also learned that any algorithm should not be more
complicated than 150 lines. Anyway every developer chooses an editor according
own personality. It is like - show me your IDE and I will tell who you are
"""

# veri on isleme: noktalama isaretlerinden kurtul, kucuk harf donusumu yap
# kelimeleri bol

words = text.replace('.','').replace('!','').lower().split()

# kelime frekanslarini hesapla ve indeksleme olustur

word_count = Counter(words)
# kelime frekansini buyukten kucuge sirala 
vocab = sorted(word_count, key=word_count.get, reverse=True)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# egitim verisini hazirlama

data = [(words[i], words[i+1]) for i in range(len(words)-1)]


# %% lstm modeli tanimlama

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        # bir ust sinifin constructor i 
        super(LSTM, self).__init__()
        # embedding katmani
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM katmani
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    # ileri besleme fonksiyonu
    def forward(self, x):
        # input -> embedding
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x.view(1,1,-1))
        output = self.fc(lstm_out.view(1,-1))
        return output

model = LSTM(len(vocab), embedding_dim=8, hidden_dim=32)

# %% hyperparameter tuning

# %% lstm training

# %% test ve degerlendirme






















