import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

args = argparse.ArgumentParser(description='args')
args.add_argument('--batch', default=128, type=int)
args.add_argument('--dims', default=768, type=int)
args.add_argument('--lr', default=0.0001, type=float)
args.add_argument('--hidden', default=128, type=int)
args.add_argument('--reg_lr', default=1, type=float)
args.add_argument('--CL', default=0, type=int)
args.add_argument('--aa', default=0, type=int)
args.add_argument('--cl_lr', default=1, type=float)
args.add_argument('--aa_lr', default=1, type=float)
args.add_argument('--early_stop', default=5, type=int)
args.add_argument('--dataset', default='CL', type=str)
args.add_argument('--pivot', default=3, type=int)
args.add_argument('--edge_ratio', default=100, type=int)
args.add_argument('--device', default=0, type=int)
args.add_argument('--rand', default=10, type=int)
args = args.parse_args()

torch.manual_seed(args.rand)
np.random.seed(args.rand)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
device = torch.device('cuda:0')

class ReviewDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        self.data = []
        for user_id, samples in data.items():
            for sample in samples:
                self.data.append((int(user_id), sample[0], sample[1]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_data(train_path, val_path, test_path):
    train_data = ReviewDataset(train_path)
    val_data = ReviewDataset(val_path)
    test_data = ReviewDataset(test_path)
    return train_data, val_data, test_data

def collate_fn(batch):
    user_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
    ratings = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    items = torch.tensor([x[2] for x in batch], dtype=torch.long)
    
    return user_ids, items, ratings

class Model(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim, CL, aa, reviews, ratings):
        super(Model, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_pos_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_neg_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_pos_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_neg_embedding = nn.Embedding(num_items, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding.weight = nn.Parameter(torch.from_numpy(reviews[0]))
        self.item_embedding.weight = nn.Parameter(torch.from_numpy(reviews[1]))
        self.user_pos_embedding.weight = nn.Parameter(torch.from_numpy(reviews[2]))
        self.user_neg_embedding.weight = nn.Parameter(torch.from_numpy(reviews[4]))
        self.item_pos_embedding.weight = nn.Parameter(torch.from_numpy(reviews[3]))
        self.item_neg_embedding.weight = nn.Parameter(torch.from_numpy(reviews[5]))

        self.user_embedding.weight.requires_grad = False
        self.user_pos_embedding.weight.requires_grad = False
        self.user_neg_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False
        self.item_pos_embedding.weight.requires_grad = False
        self.item_neg_embedding.weight.requires_grad = False
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_p = nn.Embedding(num_users, hidden_dim)
        self.item_p = nn.Embedding(num_items, hidden_dim)
        nn.init.xavier_uniform_(self.user_p.weight)
        nn.init.xavier_uniform_(self.item_p.weight)

        self.uFC = nn.Sequential(
                nn.Linear(num_items, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.iFC = nn.Sequential(
                nn.Linear(num_users, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.ruFC = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.riFC = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.rupFC = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.runFC = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.ripFC = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.rinFC = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.nFC = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU()
                )
        self.ugnn1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                )
        self.ignn1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                )
        self.upnn1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                )
        self.ipnn1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                )
        self.unnn1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                )
        self.innn1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                )

        

        self.num_users = num_users
        self.num_items = num_items
        self.CL = CL
        self.aa = aa
        self.ReLU = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.reviews = reviews
        self.ratings = ratings
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.user_rating = torch.from_numpy(ratings[0].astype('float32')).to(device)
        self.item_rating = torch.from_numpy(ratings[1].astype('float32')).to(device)

        self.rr_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_up = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_un = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_ip = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_in = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)

    def graph_edge__(self, user_emb, item_emb, user_ids, item_ids):

        k = args.edge_ratio
        norm_u = torch.nn.functional.normalize(user_emb, p=2, dim=1)
        norm_i = torch.nn.functional.normalize(item_emb, p=2, dim=1)

        num_u = max(1, int(torch.ceil(torch.tensor(k / 100.0 * self.num_users))))
        num_i = max(1, int(torch.ceil(torch.tensor(k / 100.0 * self.num_items))))

        def calculate_mask_in_batches(norm_matrix, num_elements, ids, top_k):
            mask = torch.ones((len(ids), num_elements), dtype=torch.bool, device=norm_matrix.device)
            similarities = torch.sparse.mm(norm_matrix[ids], norm_matrix.t())
            probs = nn.Softmax(dim=1)(similarities)
            selected_indices = torch.multinomial(probs, top_k, replacement=False)
            mask[torch.arange(ids.size(0)).unsqueeze(1), selected_indices] = False

            probs = nn.Softmax(dim=1)((~mask) * similarities)
            return mask, probs

        u_mask, u_sims = calculate_mask_in_batches(norm_u, self.num_users, user_ids, num_u)
        i_mask, i_sims = calculate_mask_in_batches(norm_i, self.num_items, item_ids, num_i)
        del norm_u, norm_i
        del num_u, num_i

        return u_mask, i_mask, u_sims, i_sims

    def mark_unique_elements(self, tensor):

        unique_elements, inverse_indices, counts = torch.unique(tensor, return_inverse=True, return_counts=True)

        boolean_tensor = torch.zeros_like(tensor, dtype=torch.bool, device=device)

        first_occurrence = torch.zeros_like(unique_elements, dtype=torch.bool, device=device)
        boolean_tensor[first_occurrence[inverse_indices].logical_not()] = True
        first_occurrence[inverse_indices] = True
        
        return boolean_tensor    

    def forward(self, user_ids, item_ids, rating, clip):

        u_mask, i_mask, u_sims, i_sims = self.graph_edge__(self.user_rating/5, self.item_rating/5, user_ids, item_ids)

        unique_u_mask = (~u_mask).any(dim=0)
        unique_i_mask = (~i_mask).any(dim=0)

        unique_u_mask[user_ids] = True
        unique_i_mask[item_ids] = True
        t_user_ids = torch.searchsorted(torch.nonzero(unique_u_mask, as_tuple=False).squeeze(), user_ids)
        t_item_ids = torch.searchsorted(torch.nonzero(unique_i_mask, as_tuple=False).squeeze(), item_ids)

        reg_loss = 0
        cl_loss = 0

        user_embeds = self.ruFC(self.user_embedding.weight[unique_u_mask])
        item_embeds = self.riFC(self.item_embedding.weight[unique_i_mask])
        user_pos_embeds = self.rupFC(self.user_pos_embedding.weight[unique_u_mask])
        user_neg_embeds = self.runFC(self.user_neg_embedding.weight[unique_u_mask])
        item_pos_embeds = self.ripFC(self.item_pos_embedding.weight[unique_i_mask])
        item_neg_embeds = self.rinFC(self.item_neg_embedding.weight[unique_i_mask])

        user_r_embeds = torch.mm(u_sims[:,unique_u_mask], user_embeds)
        user_r_embeds = self.ugnn1(user_r_embeds)
        item_r_embeds = torch.mm(i_sims[:,unique_i_mask], item_embeds)
        item_r_embeds = self.ignn1(item_r_embeds)
        user_r_embeds = user_r_embeds + user_embeds[t_user_ids]
        item_r_embeds = item_r_embeds + item_embeds[t_item_ids]

        user_pos_r_embeds = torch.mm(u_sims[:,unique_u_mask], user_pos_embeds)
        user_pos_r_embeds = self.upnn1(user_pos_r_embeds)
        user_pos_r_embeds = user_pos_r_embeds + user_pos_embeds[t_user_ids]

        item_pos_r_embeds = torch.mm(i_sims[:,unique_i_mask], item_pos_embeds)
        item_pos_r_embeds = self.ipnn1(item_pos_r_embeds)
        item_pos_r_embeds = item_pos_r_embeds + item_pos_embeds[t_item_ids]

        user_neg_r_embeds = torch.mm(u_sims[:,unique_u_mask], user_neg_embeds)
        user_neg_r_embeds = self.unnn1(user_neg_r_embeds)
        user_neg_r_embeds = user_neg_r_embeds + user_neg_embeds[t_user_ids]

        item_neg_r_embeds = torch.mm(i_sims[:,unique_i_mask], item_neg_embeds)
        item_neg_r_embeds = self.innn1(item_neg_r_embeds)
        item_neg_r_embeds = item_neg_r_embeds + item_neg_embeds[t_item_ids]


        user_biases = self.user_bias(user_ids).squeeze()
        item_biases = self.item_bias(item_ids).squeeze()
        dot_product = 0
        dot_product = dot_product + (user_r_embeds * item_r_embeds).sum(1)
        dot_product = dot_product + (user_pos_r_embeds * item_r_embeds).sum(1)
        dot_product = dot_product - (user_neg_r_embeds * item_r_embeds).sum(1)

        prediction = dot_product + user_biases + item_biases
        if clip==1:
            prediction = torch.clamp(prediction, 1, 5)

        aa_loss = 0
        return prediction, reg_loss, cl_loss, aa_loss

def sigmoid(z):
    return 1/(1+np.exp(-z))

def ratings_to_array(ratings):
    z = []
    for rating in ratings:
        r_array = [0 for i in range(0, 5)]
        r_array[int(rating.item()-1)] = 1
        z.append(r_array)
    return torch.tensor(z).float()

def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    total_mse = 0
    total_cl = 0
    total_aa = 0
    for user_ids, item_ids, ratings in tqdm(dataloader):
        user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
        
        optimizer.zero_grad()
        outputs, reg_loss, cl_loss, aa_loss = model(user_ids, item_ids, ratings, clip=0)

        loss_ = criterion(outputs, ratings)

        loss = loss_ + args.reg_lr*reg_loss + args.cl_lr*cl_loss + args.aa_lr*aa_loss
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
        total_mse += loss_.item()
        total_cl += args.cl_lr*cl_loss
        total_aa += args.aa_lr*aa_loss
    return total_loss / len(dataloader), total_mse / len(dataloader), total_cl / len(dataloader), total_aa / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            
            outputs, _, _, _ = model(user_ids, item_ids, ratings)
            loss = criterion(outputs, ratings)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_rmse(model, dataloader, device, criterion):
    model.eval()
    predictions = []
    actuals = []
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            
            outputs, _, _, _ = model(user_ids, item_ids, ratings, clip=1)
            mae = (ratings - outputs).abs().mean()
            mse = nn.MSELoss()(outputs, ratings)
            rmse = np.sqrt(mse.item())
            total_mae = total_mae + mae.item()
            total_mse = total_mse + mse.item()
            total_rmse = total_rmse + rmse
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())

    return total_rmse / len(dataloader), total_mse / len(dataloader), total_mae / len(dataloader)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

def load_ratings(dataset, max_reviewer_index, max_asin_index):
    with open(dataset, 'r') as file:
        data = json.load(file)

    reviewer_ratings = np.zeros((max_reviewer_index, max_asin_index))
    asin_ratings = np.zeros((max_asin_index, max_reviewer_index))

    for reviewer_index, ratings in data.items():
        reviewer_index = int(reviewer_index)
        for rating, asin_index in ratings:
            reviewer_ratings[reviewer_index, asin_index] = rating
            asin_ratings[asin_index, reviewer_index] = rating

    return reviewer_ratings.astype('int'), asin_ratings.astype('int')

def main():
    dataset = args.dataset
    train_path = f'./dataset/{dataset}/train.json'
    val_path = f'./dataset/{dataset}/val.json'
    test_path = f'./dataset/{dataset}/test.json'
    
    train_data, val_data, test_data = load_data(train_path, val_path, test_path)

    num_users = max([int(user_id) for user_id, _, _ in train_data.data]) + 1
    num_items = max([asin for _, _, asin in train_data.data]) + 1
    print(f'Users: {num_users}')
    print(f'Items: {num_items}')
    
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    
    u_review = np.load(f'./dataset/{dataset}/encoded_user_review_v2.npy', allow_pickle=True)
    u_p_review = np.load(f'./dataset/{dataset}/encoded_user_review_rating_{args.pivot+1}_or_above_v2.npy', allow_pickle=True)
    u_n_review = np.load(f'./dataset/{dataset}/encoded_user_review_rating_{args.pivot}_or_below_v2.npy', allow_pickle=True)
    i_review = np.load(f'./dataset/{dataset}/encoded_item_review_v2.npy', allow_pickle=True)
    i_p_review = np.load(f'./dataset/{dataset}/encoded_item_review_rating_{args.pivot+1}_or_above_v2.npy', allow_pickle=True)
    i_n_review = np.load(f'./dataset/{dataset}/encoded_item_review_rating_{args.pivot}_or_below_v2.npy', allow_pickle=True)

    u_ratings, i_ratings = load_ratings(train_path, num_users, num_items)

    model = Model(num_users, num_items, embedding_dim=args.dims, hidden_dim=args.hidden, CL=args.CL, aa=args.aa, reviews=[u_review,i_review,u_p_review,i_p_review,u_n_review,i_n_review], ratings=[u_ratings, i_ratings]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model_name = f'LETTER_{args.pivot}'
    num_epochs = 1000

    updated = 0
    best = 100
    for epoch in range(num_epochs):
        train_loss, mse, cl, aa = train(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_mse, val_mae = evaluate_rmse(model, val_loader, device, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, MSE: {mse:.4f}, Val RMSE: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}')
        if best > val_mse:
            best = val_mse
            best_path = f'./saved_model/{dataset}_{model_name}_{epoch}_b{args.batch}_h{args.hidden}_l{args.lr}'
            best_model = copy.deepcopy(model.state_dict())
            updated = 0

        updated = updated + 1
        if updated > args.early_stop and epoch > 0:
            break
    
    save_model(best_model, best_path)
    model.load_state_dict(best_model)

    test_rmse, test_mse, test_mae = evaluate_rmse(model, test_loader, device, criterion)
    print(f'Test RMSE: {test_rmse:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}')


if __name__ == '__main__':
    main()
