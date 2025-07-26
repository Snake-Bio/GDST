import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GATConv
from sklearn.neighbors import NearestNeighbors
import os
from sklearn.decomposition import PCA
import scanpy as sc
from sklearn import metrics
class InfoNCEDiscriminator(nn.Module):
    def __init__(self, temperature=0.2):
        super(InfoNCEDiscriminator, self).__init__()
        self.temperature = temperature
    def forward(self, q, k_pos, k_neg):
        q = F.normalize(q, dim=1)
        k_pos = F.normalize(k_pos, dim=1)
        k_neg = F.normalize(k_neg, dim=1)
        pos_logits = torch.sum(q * k_pos, dim=1, keepdim=True)
        neg_logits = torch.matmul(q, k_neg.T)
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, 1+M]
        logits /= self.temperature
        labels = torch.zeros(q.size(0), dtype=torch.long).to(q.device)  # 正样本在第一个位置
        loss = F.cross_entropy(logits, labels)
        return loss

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)
class Encoder(nn.Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.elu,
                 hidden_dims=[64, 64], heads=1, temperature=0.2):
        super(Encoder, self).__init__()
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.read = AvgReadout()
        self.disc = InfoNCEDiscriminator(temperature=temperature)

        dims = [in_features] + hidden_dims
        self.gnn_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.gnn_layers.append(GATConv(dims[i], dims[i+1], heads=heads, concat=False))
        self.reconstruct = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ELU(),
            nn.Linear(64, in_features),
            nn.ELU()
        )

    def forward(self, feat, feat_a, adj):
        x = F.dropout(feat, self.dropout, self.training)
        for layer in self.gnn_layers:
            x = self.act(layer(x, adj))
        hiden_emb = x
        h = self.reconstruct(hiden_emb)
        h = self.act(h)
        xa = F.dropout(feat_a, self.dropout, self.training)
        for layer in self.gnn_layers:
            xa = self.act(layer(xa, adj))
        emb_a = xa
        emb = hiden_emb
        g = self.read(emb, self.graph_neigh)
        g_a = self.read(emb_a, self.graph_neigh)
        neg_samples = emb[torch.randperm(emb.size(0))]
        ret = self.disc(g, emb, neg_samples)
        neg_samples_a = emb_a[torch.randperm(emb_a.size(0))]
        ret_a = self.disc(g_a, emb_a, neg_samples_a)
        return hiden_emb, h, ret, ret_a

class Encoder_sparse(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, temperature=0.2):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.disc = InfoNCEDiscriminator(temperature=temperature)
        self.read = AvgReadout(
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def graph_diffusion(self, feat, adj, diffusion_steps=3):
        diffused_feat = feat
        for _ in range(diffusion_steps):
            diffused_feat = torch.spmm(adj, diffused_feat)
        return diffused_feat
    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        hiden_emb = z
        z_aug = self.graph_diffusion(z, adj)
        h = torch.mm(z_aug, self.weight2)
        h = torch.spmm(adj, h)
        emb = self.act(z)
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
        g = self.read(emb, self.graph_neigh)
        g_a = self.read(emb_a, self.graph_neigh)
        ret = self.disc(g, emb)
        ret_a = self.disc(g_a, emb_a)
        return hiden_emb, h, ret, ret_a


def filter_with_overlap_gene(adata, adata_sc):

    if 'highly_variable' not in adata.var.keys():
        raise ValueError("'highly_variable' are not existed in adata!")
    else:
        adata = adata[:, adata.var['highly_variable']]

    if 'highly_variable' not in adata_sc.var.keys():
        raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:
        adata_sc = adata_sc[:, adata_sc.var['highly_variable']]

    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes

    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]

    return adata, adata_sc


def construct_interaction(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj


def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm['graph_neigh'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj
    print('Graph constructed!')


def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_feature_with_diffusion(adata, tau=0.5, mix_ratio=0.8):
    import numpy as np
    import pandas as pd
    import scipy.linalg
    if 'highly_variable' in adata.var:
        adata_vars = adata[:, adata.var['highly_variable']]
    else:
        raise ValueError("adata.var 中没有 'highly_variable'，请先运行 preprocess()")
    from scipy.sparse import issparse
    if issparse(adata_vars.X):
        feat = adata_vars.X.toarray()
    else:
        feat = adata_vars.X
    adata.obsm['feat'] = feat
    A = adata.obsm['adj']
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    H = scipy.linalg.expm(-tau * L)
    feat_diffused = H @ feat
    feat_a = mix_ratio * feat_diffused+(1-mix_ratio)*feat
    adata.obsm['feat_a'] = feat_a


class GDST():
    def __init__(self,
                 adata,
                 adata_sc=None,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 learning_rate_sc=0.01,
                 weight_decay=0.00,
                 epochs=600,
                 dim_input=3000,
                 dim_output=64,
                 random_seed=42,
                 alpha=10,
                 beta=1,
                 theta=0.1,
                 lamda1=10,
                 lamda2=1,
                 deconvolution=False,
                 datatype='10X'
                 ):

        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype

        fix_seed(self.random_seed)
        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
            add_contrastive_label(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature_with_diffusion(adata, tau=0.5, mix_ratio=0.8)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(
            self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(
            self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
            print('Building sparse matrix ...')
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)
        if self.deconvolution:
            self.adata_sc = adata_sc.copy()
            if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
                self.feat_sp = adata.X.toarray()[:, ]
            else:
                self.feat_sp = adata.X[:, ]
            if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
                self.feat_sc = self.adata_sc.X.toarray()[:, ]
            else:
                self.feat_sc = self.adata_sc.X[:, ]

            self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
            self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
            self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
            self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
            if self.adata_sc is not None:
                self.dim_input = self.feat_sc.shape[1]
            self.n_cell = adata_sc.n_obs
            self.n_spot = adata.n_obs

    def train(self):
        if isinstance(self.adj, torch.Tensor):
            edge_index, _ = dense_to_sparse(self.adj)
            self.adj_edge_index = edge_index.to(self.device)
        else:
            raise ValueError("当前模型期望 adj 为 Tensor 类型")

        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        print('开始训练ST数据')
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj_edge_index)
            self.loss_sl_1 = ret
            self.loss_sl_2 = ret_a
            self.loss_feat = F.mse_loss(self.features, self.emb)
            loss = self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("结束")

        with torch.no_grad():
            self.model.eval()

            if self.deconvolution:

                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                return self.emb_rec
            else:
                if self.datatype in ['Stereo', 'Slide']:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj_edge_index)[1]
                    self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
                else:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj_edge_index)[1]

                self.adata.obsm['emb'] = self.emb_rec.detach().cpu().numpy()

                return self.adata


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01,
               refinement=False):

    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']

    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def extract_top_value(map_matrix, retain_percent=0.1):
    top_k = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)

    return output


def construct_cell_type_matrix(adata_sc):
    label = 'cell_type'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    return mat
