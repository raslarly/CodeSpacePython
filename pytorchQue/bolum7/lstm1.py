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
I started learning programming because I wanted to understand how things work, not because I wanted quick money. At first everything felt confusing. Variables made no sense, loops felt pointless, and errors appeared constantly. The more I tried to fix things, the more broken everything looked.

Over time I realized something uncomfortable: confusion is not a sign of failure, it is the default state of learning. Anyone who claims otherwise is either lying or has forgotten what it feels like to be a beginner. Progress happened slowly, almost invisibly. One day I could not write a function. A week later I could. There was no dramatic moment, no sudden breakthrough.

People often talk about talent, but they ignore repetition. Writing bad code over and over again was what actually helped. Copying examples, breaking them, fixing them, and breaking them again taught me more than any tutorial ever did. The keyboard became familiar. Errors stopped feeling personal.

There were days where motivation disappeared completely. On those days discipline mattered more than passion. Even writing ten lines of useless code kept the habit alive. Consistency turned out to be more important than intensity.

Eventually I understood that learning never really ends. New tools appear, old ones change, and yesterday’s best practice becomes today’s mistake. The goal is not to know everything, but to stay comfortable with not knowing and still moving forward.

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

# kelime listesi -> tensor
def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype = torch.long)

# hyperparameter tuning kombinasyonlarini belirle
# denenecek gizli katman boyutlari
embedding_size =  [8, 16]
# denenecek gizli katman boyutlari
hidden_sizes = [32,64]
# denenecek ogrenme oranlari
learning_rate = [0.01, 0.005]
# en dusuk kayip degerini saklamak icin bir degisken
best_lost = float('inf')
# en iyi parametreleri saklamak icin bos bir dictionary
best_params = {}

print(' hyperparameter tuning is basliyooor ...')

# grid search

for embSize,hiddenSize, lr in product(embedding_size,hidden_sizes,
                                      learning_rate):
    print(f'Embedding: {embSize}, hidden: {hiddenSize}, learninhg: {lr}')
    # secilen parametreler ile model olustur
    model = LSTM(len(vocab), embSize, hiddenSize)
    # Entropi kayip fonksiyonu
    loss_function = nn.CrossEntropyLoss()
    # secilen lr ile adam optimizeri
    optimizer = optim.Adam(model.parameters(), lr= lr)

    epochs = 50
    total_loss = 0
        
    for epoch in range(epochs):
        # epoch baslangicinda kaybi sififrlayalim
        epoch_loss = 0
        for word, next_word in data:
            # gradyanlari sifirla
            model.zero_grad()
            # girdiyi tensor'e cevir
            input_tensor = prepare_sequence([word], word_to_ix)
            # hedef kelimeyi tensore donustur
            target_tensor = prepare_sequence([next_word], word_to_ix)
            # prediction
            output = model(input_tensor)
            loss = loss_function(output, target_tensor)
            # geri yayilim islemi uygula
            loss.backward()
            # parametreleri guncelle
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, loss: {epoch_loss:.5f}')
            
        total_loss = epoch_loss
        
    # en iyiy modeli kaydet
    if total_loss < best_lost:
        best_lost = total_loss
        best_params = {'embedding_dim':embSize, 'hidden_dim':hiddenSize,
                       'learning rate':lr}
    print()
    
print(f'Best params: {best_params}')

# %% lstm training

finalModel = LSTM(len(vocab), best_params['embedding_dim'],
                  best_params['hidden_dim'])
optimizer = optim.Adam(finalModel.parameters(),
                       lr=best_params['learning rate'])

# %% test ve degerlendirme

print('final model training')
epochs= 100
for epoch in range(epochs):
    epoch_loss = 0
    for word, next_word in data:
        finalModel.zero_grad()
        input_tensor = prepare_sequence([word],word_to_ix)
        target_tensor = prepare_sequence([next_word],word_to_ix)
        output = finalModel(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch %10 == 0:
        print(f'Final Model Epoch: {epoch}, losss: {epoch_loss:.5f}')

# %% test ve degerlendirme

# kelime tahmini fonksiyonu: baslangic kelimesi ve n adet kelime uretmesini
# sagla

def predict_sequence(start_word, num_words):
    # su anki kelime baslangic kelimesi olarak ayarlanir
    current_word = start_word
    # cikti dizisi
    output_sequence = [current_word]
    
    # belirtilen sayida kelime tahmini
    for _ in range(num_words):
        # gradyan hesaplamasi yapmadan
        with torch.no_grad():
            # kelime -> tensor
            input_tensor = prepare_sequence([current_word], word_to_ix)
            output = finalModel(input_tensor)
            # en yuksek olasiliga sahip kelimenin indexi
            predicted_idx = torch.argmax(output).item()
            # indexe karsilik gelen kelimeyi return eder
            predicted_word = ix_to_word[predicted_idx]
            output_sequence.append(predicted_word)
            # bir sonraki tahmin icin mevcut kelimeleri guncelle
            current_word = predicted_word
    # tahmin edilen kelime dizisi return edilir
    return output_sequence

modelCiktisi = predict_sequence('because',6)
print(' '.join(modelCiktisi))














