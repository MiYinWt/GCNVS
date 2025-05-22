import  os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from random import shuffle
from sklearn.metrics import roc_curve,auc,confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from dataset import *
from model import GCNnet
from utils import *

def train(model,device,train_loader,epoch,optimizer):  
    loss_fn = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    total_loss = 0
    for batch_idx,data in enumerate(train_loader):
        data = data.to(device)
        out = model(data)
        # print("out:\n",out)
        loss = loss_fn(out, data.y.to(torch.long).to(device))
        # loss = loss_fn(out, data.y.to(torch.float).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_loss += loss.item()
    print('Train epoch: {:5d} /{:5d} [{:3.2f}%]\t Train tLoss: {:.6f}'.format(epoch,
                                                                    NUM_EPOCHS,
                                                                    100. * (epoch) / NUM_EPOCHS,
                                                                    total_loss / len(train_loader)))
    return total_loss / len(train_loader)

def validate(model, device, val_loader):
    loss_fn = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            out = model(data)
            pred = torch.argmax(out, dim=1)
            loss = loss_fn(out, data.y.to(torch.long).to(device))
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            total_loss += loss.item()
        accuracy = 100. * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%\t Validation Loss: {total_loss/len(val_loader):.6f}')
    return total_loss/len(val_loader)



def test(model, device, test_loader):
    model.eval()  
    model.to(device)

    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():  
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            # print("out:\n",out)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            # print("pred:\n",pred)
            proba = torch.softmax(out,dim=1).cpu().numpy() 
            proba = proba[:,1]
            labels = data.y.cpu().numpy()
            all_probs.append(proba)
            all_labels.append(labels)
            # print("pred:\n",pred,"\ntrue:\n",data.y)  
            correct += (pred == labels).sum()
            total += len(labels)
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.4f}%')
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
       # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


NUM_EPOCHS = 500

train_data,train_label = proccesed_data('data/train.csv')
test_data,test_label = proccesed_data('data/test.csv')
val_data,val_label = proccesed_data('data/val.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=64, shuffle=True)
test_loader = DataLoader(VSDataset(test_data,test_label),batch_size=64, shuffle=False)
val_loader = DataLoader(VSDataset(val_data,val_label),batch_size=64, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = GCNnet()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
train_losses = []
val_losses = []

for i in range(NUM_EPOCHS):
    
    train_loss = train(model, device, train_loader, i+1, optimizer)
    val_loss = validate(model, device, val_loader)
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

test(model, device, test_loader)

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train & Validation Loss Curve')
plt.show()