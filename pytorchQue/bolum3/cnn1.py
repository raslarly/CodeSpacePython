"""
    Problem tanimi: CIFAR10 veriseti siniflandirma problemi
    CNN kullanilacak
"""

# %% import libraries 
import torch # pytorch
import torch.nn as nn # sinir agi katmanlari icin
import torch.optim as optim # optimizsayon algoritmasi icin
import torchvision # goruntu isleme icin
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np


# Load dataset

# batch size her iterasyonda islenecek veri sayisidir
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        # goruntuyu tensore cevir
        transforms.ToTensor(),
        # renkli bir goruntu oldugundan 3 adet 0.5 kullanilmistir.
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # CIFAR10 veri setini indir ve egitim / test veri setini olustur
    train_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    # data Loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


# %% visualize dataset

def imshow(img):
    # verileri normailze etmeden once geri donustur
    img = img/2 + 0.5 # normalizasyon isleminin tersi
    np_img = img.numpy() # tensor da numpy array'a dondur
    plt.imshow(np.transpose(np_img, (1,2,0))) # 3kanalIcin renkleri dogruGoster

# veri kumesinden ornek gorselleri almak icin fonksiyon
def get_sample_images(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

def visualize(gS):
    train_loader, test_loader = get_data_loaders()
    
    # 'gS' tane veri gorsellestirme
    images, labels = get_sample_images(train_loader)
    plt.figure()
    for i in range(gS):
        plt.subplot(1, gS, i+1)
        imshow(images[i]) # gorsellestir
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()

# visualize(3)

# %% build CNN Model

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        # input channel rgb oldugundan 3 output channel 32 filtre sayim
        # kernel size'im 3x3'luk matrix o yuzden 3
        # padding(dolgu) 1 yani giris goruntusunun her kenarina 1 piksellik 0
        # ekliyoruz
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,padding=1)
        self.relu = nn.ReLU() # Aktivasyon fonksiyonu relu genelde hizlidir
        # 2x2 boyutunda pooling katmani yaratiyoruz 
        # goruntu analizi penceresinin kayma hizi = stride
        # kernel_size = goruntu analiz penceresinin boyutu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # input channer = 32 conv1'den gliyor . output channel'e biz seciyoruz
        # 64 filtreli ikinci convolusion layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Noronlarin %20'si sifirlanacak diger noronlar ile baglantisi kesilcek
        self.dropout = nn.Dropout(0.2)# dropout %20 oraninda calisir
        self.fc1 = nn.Linear(64*8*8, 128)#fully connected layer giris = 4096,
        # output = 128
        self.fc2 = nn.Linear(128, 10) # output layer sinif sayisi kadar cikis
        
        # image 3x32x32 -> conv(32) -> relu(32) -> pool (16)
        # conv(16) -> relu(16) -> pool(8) -> image = 8x8
    
    def forward(self, x):
        """
            image 3x32x32 -> conv(32) -> relu(32) -> pool (16)
            conv(16) -> relu(16) -> pool(8) -> image = 8x8
            flatten
            fc1 -> relu -> dropout
            fc2 -> output 
        """
        
        x = self.pool(self.relu(self.conv1(x))) # ilk convolutional blok
        x = self.pool(self.relu(self.conv2(x))) # ikinci convolutional blok
        x = x.view(-1, 64*8*8) # flatten
        x = self.dropout(self.relu(self.fc1(x))) # fully connected layer
        x = self.fc2(x) # output
        return x
    



# define loss function and optimizer 

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # multi class classification problem
    optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # SGD
    # Sthotastic Gradient Descent
    )

# %% Training

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    
    # modeli egitim moduna alalim
    model.train()
    # loss degerlerini saklamak icin bir liste olustur
    train_loses = []
    
    # belirtilen epoch sayisi kadar for dongusu olustur
    for epoch in range(epochs):
        # toplam loss degerini saklamak icin total_loss        
        total_loss = 0
        # for dongusu tum egitim verisetini taramak icin        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)        
            # gradyanlari sifirlamak
            optimizer.zero_grad()
            # forward propogation (preidction)
            outputs = model(images)
            # loss degeri hesapla
            loss = criterion(outputs, labels)
            # geri yayilim (gradyan hesaplama)
            loss.backward()
            # ogrenme = agirlik yani parametre guncelleme
            optimizer.step()
            
            total_loss += loss.item()
            
        # ortalama kayip hesaplama
        avg_loss = total_loss/len(train_loader)
        train_loses.append(avg_loss)
        print(f'Epoch: {epoch+1}/{epochs}, loss: {avg_loss:.5f}')
    # kayip (loss) grafigi
    plt.figure()
    plt.plot(range(1, epochs+1), train_loses, marker='o', linestyle='-',
             label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Trainingh loss')
    plt.legend()
    plt.show()

# %% Test

def test_model(model, test_loader, dataset_type):
    # degerlendirme modu
    model.eval()
    # dogru tahmin sayaci
    correct=0
    # topşlam veri sayaci
    total =0
    
    # gradyan hesaplamasini kapat
    with torch.no_grad():
        # test verisetini kullanarak degerlendirme
        for images,labels in test_loader:
            # verileri cihaza tasi
            images, labels = images.to(device), labels.to(device)
            # predictions
            outputs = model(images)
            # en yuksek olasilikli sinifi sec
            _, predicted = torch.max(outputs, 1)
            # toplam veri sayisini guncelle
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # dogruluk oranini ekrana yazdir
    print(f'{dataset_type} accuracy: %{100*correct/total}')



# %% main program

if __name__ == '__main__':
    
    # on ayarlar
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # veri gorsellestirme
    visualize(3)
    # model tanimlama
    model = CNN().to(device)
    # training
    train_loader, test_loader = get_data_loaders()
    criterion, optimizer =define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    # test
    test_model(model, test_loader, dataset_type='Test') # %62.83
    test_model(model, train_loader, dataset_type='Training') # %65.23












