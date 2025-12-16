"""
rnn: tekrarlayan sinir aglari: zaman serilerinde kullaniyorduk: kisa ozet
veriseti secme 
"""

# %% veriyi olustur ve gorsellestir

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def generate_data(seq_length = 50, num_samples=1000):
    # 0 ile 100 arasinda num_samples adet veri olustur
    x = np.linspace(0,100, num_samples)
    y = np.sin(x)
    # giris dizilerini saklamak icin bos liste
    sequence =[]
    # hedef degerleri saklamak icin
    targets = []
    
    for i in range(len(x) - seq_length):
        sequence.append(y[i:i+seq_length])
        targets.append(y[i + seq_length])
        
        """
        example 3 lu paket
        sequence: [2,3,4]
        target: [5]
        """
    
    plt.figure(figsize=(8,4))
    plt.plot(x,y,label='sin(x)',color='b',linewidth=2)
    plt.title('sinus dalga grafigi')
    plt.xlabel('Zaman(radyan')
    plt.ylabel('Genlik')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return np.array(sequence), np.array(targets)

sequence, targets = generate_data()

# %% rnn modelini olustur

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
            RNN: recurrent neural network
            RNN -> Linear (output)
        """
        super(RNN, self).__init__()
        # RNN layer
        # input_size: giris boyutu, hidden_size: rnn gizli katman hucre sayisi
        # num_layers: rnn layer sayisi
        self.rnn=nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # fully connected layer: output , output_size = cikti boyutu
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # rnn'e girdiyi ver ciktiyi al
        out, _ = self.rnn(x)
        # son zaman adiminda ki ciktiyi al ve fully connected layer a bagla
        out = self.fc(out[:,-1,:])
        
        return out

model = RNN(1, 16, 1, 1)
# %% rnn training

# hiperparametreler

# input dizisinin boyutu
seq_length = 50
# input dizisinin boyutu
input_size = 1
# rnn gizli katmanlarinda ki dugum sayisi
hidden_size = 16
# output boyutu veya tahmin edilen deger
output_size = 1
# rnn katman sayisi
num_layers = 1
# modelin kac kez tum veriseti uzerinde egitilecegi
epochs = 20
# her bir egitim adiminda kac ornegin kullanilacagi
batch_size = 32
# optimizasyon algoritmasi icin ogrenme orani ya da hizi
learning_rate = 0.001

# veriyi hazirla 

x,y = generate_data(seq_length)
# pytorch tensorune cevir ve boyut ekle
x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
# pytorch tensorune cevir ve boyut ekle
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
# pytorch da dataset olsuturma
dataset = torch.utils.data.TensorDataset(x,y)
# veri yukleyici olustur
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

# modeli tanimla
model = RNN(input_size, hidden_size, output_size, num_layers)
# loss function: mean square error (ortalama kare hata)
criterion = nn.MSELoss()
# optimization = adaptive momentum
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def malakianEpo():
    for epoch in range(epochs):
        for batch_x, batch_y in dataLoader:
            # gradyanlari sifirla
            optimizer.zero_grad()
            # modelden tahmini al
            pred_y = model(batch_x)
            # model tahmini ile gercekte olani karsilastir ve loss hesapla
            loss = criterion(pred_y, batch_y)
            # geri yayilim ile gradyanlari hesapla
            loss.backward()
            # optimizasyonlari uygula agirliklari guncelle
            optimizer.step()
        print(f'Epoch: {epoch+1}/{epochs} Loss: {loss.item():.4f}')
malakianEpo()

# %% rnn test and evaluation

# test icin veri olustur
# ilk test verimiz
xTest = np.linspace(100, 110, seq_length).reshape(1, -1)
# test verimizin gercek sonucu
yTest = np.sin(xTest)

# ikinci test verimiz
xTest2 = np.linspace(120,130,seq_length).reshape(1, -1)
yTest2 = np.sin(xTest2)

# from numpy to tensor 

yTest = torch.tensor(yTest, dtype=torch.float32).unsqueeze(-1)
yTest2 = torch.tensor(yTest2, dtype=torch.float32).unsqueeze(-1)

# modeli kullanarak predicton yap

model.eval()
# ilk test verisi icin tahmin yapma
prediction1 = model(yTest).detach().numpy()
prediction2 = model(yTest2).detach().numpy()

# sonuclari gorsellestir

plt.figure()
plt.plot(np.linspace(0, 100, len(y)), y, marker='o', label='Training dataset')
plt.plot(yTest.numpy().flatten(), marker='o', label='Test1')
plt.plot(yTest2.numpy().flatten(), marker='o', label='Test2')
plt.plot(np.arange(seq_length, seq_length+1),prediction1.flatten(),'ro',
         label='prediction1')
plt.plot(np.arange(seq_length, seq_length+1),prediction2.flatten(), 'ro',
         label='prediction2')
plt.legend()
plt.show()

