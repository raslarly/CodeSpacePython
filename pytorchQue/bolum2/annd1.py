'''
mnist veriseti kullanılarak rakam sınıflandırması yapmak
MNIST
ANN: yapay sinir aglari
'''

# # %% Kutuphaneleri tanimla
# import os 
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch # pytorch kutuphanesi tensor islemleri
import torch.nn as nn # yapay sinir agi katmanlarini tanimlamak icin kullanicaz
import torch.optim as optim # optimizasyon algoritmalarini iceren modul
import torchvision # goruntu isleme ve preTrained modelleri icerir
import torchvision.transforms as transforms # goruntu donusumleri yapmak
#import seaborn as sns # veri gorsellestirmeleri yapmak icin
import matplotlib.pyplot as plt
# optional: cihazi belirle 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Veriseti yukleme Data loading

def get_data_loaders(batch_size=64): # her iterasyonda islenecek veri miktari
    
    transform = transforms.Compose([
        transforms.ToTensor(), # Goruntuyu tensore cevirir ve 0-255 , 0-1 
        # olceklendirir
        transforms.Normalize((0.5,), (0.5,)) # pixel degerlerini -1,1 arasina
        # olcekler
        
    ])
    
    # MNIST veri setini indir ve egitim seti kumelerini olustur
    
    trainSet = torchvision.datasets.MNIST(root='./data',train=True,
                                          download=True,transform=transform)
    testSet = torchvision.datasets.MNIST(root='./data',train=False,
                                         download=True, transform=transform)
    
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
                                          shuffle = True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size,
                                             shuffle=False)
    return trainLoader, testLoader
    
# pytorch veri yukleyicisini olustur

trainLoader, testLoader = get_data_loaders()

# Data visualization

def visualize_samples(loader, n):
    images, labels = next(iter(loader))# ilk batchden goruntu ve etiket alalim
    fig, axes = plt.subplots(1, n, figsize=(10,5))# n farkli goruntu icin
    print(images[0].shape)
    # gorsellestirme sagliyoruz
    for i in range(n):
        axes[i].imshow(images[i].squeeze(),cmap='gray')#gorseli gritonda goster
        axes[i].set_title(f'Label: {labels[i].item()}')#goruntuye ait etiketi
        # Baslik olarak almak icin yazilan kod
        axes[i].axis('off') # eksenleri gizle
    plt.show()
visualize_samples(trainLoader, 4)

# %% Define Artifical neural network

# yapay sinir agi classi
class NeuralNetwork(nn.Module): # pytorch un nn.module sinifindan miras aliyor
    
    def __init__(self): # nn insa etmek icin gerekli olan bilesenleri tanimla
        super(NeuralNetwork, self).__init__()
        # elimizde bulunan goruntuleri (2D) vektor haline cevirecegiz (1D)
        self.flatten = nn.Flatten()
        # ilk tam bagli katmani olusturacagiz 
        self.fc1 = nn.Linear(28*28, 128) # 784 input, 128 output boyutu
        
        # aktivasyon fonksiyonu olustur
        self.relu = nn.ReLU()
        
        # ikinci tam bagli katmani olustur
        self.fc2 = nn.Linear(128, 64) # 128 = input size, 64 = output size
        
        # cikti katmani olustur 
        
        self.fc3 = nn.Linear(64, 10) # 64 = input size , 10= output size
        # output size su anda elde ki sinif sayisina esittir [0-9]       
        
    def forward(self, x): # forward propogation: ileri yayilim x = goruntu
        # initial x = 28*28 lik bir goruntu duzlestir ve 784'luk hale getir
        x = self.flatten(x)
        x = self.fc1(x) # birinci bagli katman
        x = self.relu(x) # aktivasyon foknksiyonu
        x = self.fc2(x) # ikinci bagli katman
        x = self.relu(x) # aktivasyon fonksiyonu
        x = self.fc3(x) # output katmani
        
        return x # model ciktisi return ediliyor

# Create model and compile 
model = NeuralNetwork().to(device)

# kayip fonksiyonu ve optimizasyon algoritmasini belirle
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # Cok sinifli siniflandirma icin kullanilan 
    # siniflandirma fonksiyonu multi class classification problem loss function
    optim.Adam(model.parameters(), lr = 0.001)) # update weights with adam

criterion, optimizer = define_loss_and_optimizer(model)

# %% Train

def train_model(model, trainLoader, criterion, optimizer,epochs=10):
    # modeli egitim moduna alalim
    model.train()
    # her bir epoch sonucunda elde edilen loss degerlerini saklamak icin liste
    trainLosses = []
    # belirtilen epoch sayisi kadar egitim yapilacak
    for epoch in range(epochs):
        # toplam kayip degeri
        total_loss = 0
        # tum egitim verileri uzerinde iterasyon gerceklesir
        for images, labels in trainLoader:
            # verileri cihaza tasi
            images, labels = images.to(device), labels.to(device)
            # gradyanlari sifirla
            optimizer.zero_grad()
            # modeli uygula, forward propogation 
            predictions = model(images)
            # loss hesaplama -> yPred ile yReal
            loss = criterion(predictions, labels)
            # geri yayilim yani gradyan hesaplama
            loss.backward()
            # update weights
            optimizer.step()
            total_loss = total_loss + loss.item()
            
        # ortalama kayip hesaplama
        avg_loss = total_loss / len(trainLoader)
        trainLosses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f}')
    
    # loss graph 
    plt.figure()
    plt.plot(range(1,epochs+1),trainLosses, marker='o',linestyle='-',
             label='trainLoss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()
    
train_model(model, trainLoader,criterion, optimizer)

# %% Test
def test_model(model, testLoader):
    model.eval()
    correct = 0 # dogru tahmin sayaci
    total = 0 # toplam veri sayaci
    
    # gradyan hesaplama gereksiz oldugundan kapattik
    with torch.no_grad():
        # test veri kumesini donguye al
        for images, labels in testLoader:
            # verileri cihaza tasi
            images, labels = images.to(device),labels.to(device)
            predictions = model(images)
            # en yuksek olsilikli siniifn etiketini bul
            _, predicted = torch.max(predictions, 1)
            # toplam veri sayisini guncelle
            total += labels.size(0)
            # dogru tahminleri say
            correct += (predicted==labels).sum().item()
            
    print(f'Test accuracy: {100*correct/total:.3f}%')

test_model(model, testLoader)

# %% main
# Visualize the test results

if __name__ == '__main__':
    # veri yukleyicileri al
    trainLoader, testLoader = get_data_loaders()
    
    visualize_samples(trainLoader, 5)
    
    model = NeuralNetwork().to(device)
    
    criterion, optimizer = define_loss_and_optimizer(model)
    
    train_model(model, trainLoader, criterion, optimizer)
    
    test_model(model, testLoader)
























