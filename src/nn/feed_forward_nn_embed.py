import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO review https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/

class FFModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) \
                                         for categories,size in embedding_sizes])
        #length of all embeddings combined
        self.n_emb = sum(e.embedding_dim for e in self.embeddings) 
        self.n_cont = n_cont
        
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 1)
        
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        
        self.embedding_dropout = nn.Dropout(0.6)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x_cat, x_cont, *other):

        # Set up processing for categorical columns
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.embedding_dropout(x)
        # Set up processing for continous columns
        x2 = self.bn1(x_cont)        
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.bn3(x)
        x = self.lin3(x)
        
        return x.squeeze()

    
def fit(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for X1, X2, y in train_dl:
        batch = y.shape[0]
        output = model(X1, X2)
        loss = F.mse_loss(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch*(loss.item())
    return sum_loss/total


def calculate_mse(model, valid_loader):
    """
    MSE of loss 
    """
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x1, x2, y in valid_loader:
        current_batch_size = y.shape[0]
        output = model(x1, x2)
        
        loss = F.mse_loss(output, y)
        
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size

    return sum_loss/total


    
    
    
    
    
    
