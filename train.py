import  os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from random import shuffle
from sklearn.metrics import f1_score, roc_curve,auc,confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from dataset import *
from model import GCNnet,CNN
from scipy.interpolate import make_interp_spline
from utils import *
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def mask_node_features(x, mask_ratio=0.1):
    x = x.clone() 
    mask = torch.rand(x.size(0)) < mask_ratio  
    x[mask] = 0
    return x

def train(model,device,train_loader,epoch,optimizer):  
    loss_fn = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    total_loss = 0
    for batch_idx,data in enumerate(train_loader):
        data = data.to(device)
        data.x = mask_node_features(data.x, mask_ratio=0.1)  # Mask node features
        data.graph_features = mask_node_features(data.graph_features, mask_ratio=0.1)  # Mask graph features
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



def test(model, device, test_loader,i):
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
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    # print(f'Test Accuracy: {accuracy:.4f}%  F1_score: {f1_score(all_labels, np.round(all_probs)):.4f}  Recacll: {f1_score(all_labels, np.round(all_probs), average="macro"):.4f} recision: {f1_score(all_labels, np.round(all_probs), average="micro"):.4f}')
    print(f'Test Accuracy: {accuracy:.4f}%') 
    if i +1 == NUM_EPOCHS:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.4f}")
        # draw ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        #plt.savefig('roc_curve.png')
        plt.show()
        


NUM_EPOCHS = 300

train_data,train_label = proccesed_data('data/train.csv')
test_data,test_label = proccesed_data('data/test.csv')
val_data,val_label = proccesed_data('data/val.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=64, shuffle=True)
test_loader = DataLoader(VSDataset(test_data,test_label),batch_size=64, shuffle=False)
val_loader = DataLoader(VSDataset(val_data,val_label),batch_size=64, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = CNN()
#model = GCNnet()
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
train_losses = []
val_losses = []

for i in range(NUM_EPOCHS):
    
    train_loss = train(model, device, train_loader, i+1, optimizer)
    val_loss = validate(model, device, val_loader)
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test(model, device, test_loader,i)

plt.figure()
window = 5  
train_losses_smooth = pd.Series(train_losses).rolling(window, min_periods=1, center=True).mean()
val_losses_smooth = pd.Series(val_losses).rolling(window, min_periods=1, center=True).mean()

plt.plot(train_losses_smooth, label='Train Loss')
plt.plot(val_losses_smooth, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train & Validation Loss Curve')
plt.show()