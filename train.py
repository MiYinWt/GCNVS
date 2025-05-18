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
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():  
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            # print("out:\n",out)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_probs.append(pred)
            all_labels.append(labels)
            # print("pred:\n",pred,"\ntrue:\n",data.y)  
            # correct += (pred == data.y).sum().item()
            # total += data.y.size(0)
    # accuracy = 100. * correct / total
    # print(f'Test Accuracy: {accuracy:.2f}%')
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

    # 计算最佳阈值下的TPR、TNR
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold: {best_threshold:.4f}")

    pred_label = (all_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, pred_label).ravel()
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    print(f"TPR (Recall): {TPR:.4f}")
    print(f"TNR (Specificity): {TNR:.4f}")



NUM_EPOCHS = 500

train_data,train_label = proccesed_data('data/train.csv')
test_data,test_label = proccesed_data('data/test.csv')
val_data,val_label = proccesed_data('data/val.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=64, shuffle=True)
test_loader = DataLoader(VSDataset(test_data,test_label),batch_size=64, shuffle=True)
val_loader = DataLoader(VSDataset(val_data,val_label),batch_size=64, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = GCNnet()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

for i in range(NUM_EPOCHS):
    
    train(model, device, train_loader, i+1, optimizer)
    val_loss = validate(model, device, val_loader)
    scheduler.step(val_loss)

test(model, device, test_loader)