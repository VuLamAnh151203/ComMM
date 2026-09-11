# coding: utf-8
"""FREEDOM: freezing and denoising graphs for multimodal recommendation.

This is the original FREEDOM architecture adapted to ComMM's runtime.  Its
item-item graph is built once from the raw multimodal features and kept
frozen, while the user-item graph is degree-pruned once per training epoch.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


def _config_value(config, key, default):
    value = config[key]
    return default if value is None else value


class FREEDOM(GeneralRecommender):
    """The original two-graph FREEDOM recommender."""

    def __init__(self, config, dataset):
        super(FREEDOM, self).__init__(config, dataset)

        self.embedding_dim = int(_config_value(config, 'embedding_size', 64))
        self.feat_embed_dim = int(
            _config_value(config, 'feat_embed_dim', self.embedding_dim)
        )
        self.knn_k = int(_config_value(config, 'knn_k', 10))
        self.n_mm_layers = int(_config_value(config, 'n_mm_layers', 1))
        # Keep the name used by the upstream implementation for compatibility.
        self.n_layers = self.n_mm_layers
        self.n_ui_layers = int(_config_value(config, 'n_ui_layers', 2))
        self.reg_weight = float(_config_value(config, 'reg_weight', 0.0))
        self.mm_image_weight = float(
            _config_value(config, 'mm_image_weight', 0.1)
        )
        self.dropout = float(_config_value(config, 'dropout', 0.8))

        if self.embedding_dim <= 0 or self.feat_embed_dim <= 0:
            raise ValueError('Embedding dimensions must be positive.')
        if self.knn_k <= 0:
            raise ValueError('knn_k must be positive.')
        if self.n_mm_layers < 0 or self.n_ui_layers < 0:
            raise ValueError('Graph layer counts cannot be negative.')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError('dropout must be in the interval [0, 1).')
        if not 0.0 <= self.mm_image_weight <= 1.0:
            raise ValueError('mm_image_weight must be in [0, 1].')

        self.n_nodes = self.n_users + self.n_items
        interaction_matrix = dataset.inter_matrix(form='coo').astype(
            np.float32
        )
        interaction_matrix = interaction_matrix.tocsr()
        interaction_matrix.eliminate_zeros()
        interaction_matrix.data.fill(1.0)
        self.interaction_matrix = interaction_matrix.tocoo()
        if self.interaction_matrix.nnz == 0:
            raise ValueError('FREEDOM requires at least one interaction.')

        self._build_ui_graph()

        self.user_embedding = nn.Embedding(
            self.n_users, self.embedding_dim
        )
        self.item_id_embedding = nn.Embedding(
            self.n_items, self.embedding_dim
        )
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(
                self.v_feat, freeze=False
            )
            self.image_trs = nn.Linear(
                self.v_feat.shape[1], self.feat_embed_dim
            )
            self.image_aux_projection = self._make_aux_projection(
                self.embedding_dim
            )
        else:
            self.image_embedding = None
            self.image_trs = None
            self.image_aux_projection = None

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                self.t_feat, freeze=False
            )
            self.text_trs = nn.Linear(
                self.t_feat.shape[1], self.feat_embed_dim
            )
            self.text_aux_projection = self._make_aux_projection(
                self.embedding_dim
            )
        else:
            self.text_embedding = None
            self.text_trs = None
            self.text_aux_projection = None

        self._validate_feature_shapes()
        self._build_or_load_mm_graph(config)
        self.latest_loss_components = {}

    def _make_aux_projection(self, output_dim):
        if self.feat_embed_dim == output_dim:
            return nn.Identity()
        projection = nn.Linear(self.feat_embed_dim, output_dim)
        nn.init.xavier_uniform_(projection.weight)
        nn.init.zeros_(projection.bias)
        return projection

    def _validate_feature_shapes(self):
        for modality, features in (
            ('image', self.v_feat),
            ('text', self.t_feat),
        ):
            if features is not None and features.shape[0] != self.n_items:
                raise ValueError(
                    '{} features have {} rows, expected {} items.'.format(
                        modality.capitalize(), features.shape[0], self.n_items
                    )
                )

    def _build_ui_graph(self):
        users = torch.from_numpy(
            self.interaction_matrix.row.astype(np.int64, copy=False)
        )
        items = torch.from_numpy(
            self.interaction_matrix.col.astype(np.int64, copy=False)
        )
        edge_indices = torch.stack((users, items), dim=0)
        edge_values = self._normalize_bipartite_edges(edge_indices)

        global_items = items + self.n_users
        forward_edges = torch.stack((users, global_items), dim=0)
        reverse_edges = torch.stack((global_items, users), dim=0)
        ui_edge_index = torch.cat((forward_edges, reverse_edges), dim=1)
        full_norm_edge_weights = torch.cat(
            (edge_values, edge_values), dim=0
        )
        norm_adj = torch.sparse_coo_tensor(
            ui_edge_index,
            full_norm_edge_weights,
            (self.n_nodes, self.n_nodes),
        ).coalesce()

        self.num_interactions = self.interaction_matrix.nnz
        self.register_buffer('edge_indices', edge_indices)
        self.register_buffer('edge_values', edge_values)
        self.register_buffer('ui_edge_index', ui_edge_index)
        self.register_buffer(
            'full_norm_edge_weights', full_norm_edge_weights
        )
        self.register_buffer('norm_adj', norm_adj)
        self.register_buffer(
            'freedom_adj', None, persistent=False
        )

    def _normalize_bipartite_edges(self, edge_indices):
        users, items = edge_indices
        values = torch.ones(
            users.numel(), dtype=torch.float32, device=users.device
        )
        user_degree = torch.zeros(
            self.n_users, dtype=torch.float32, device=users.device
        )
        item_degree = torch.zeros(
            self.n_items, dtype=torch.float32, device=items.device
        )
        user_degree.index_add_(0, users, values)
        item_degree.index_add_(0, items, values)
        return (
            user_degree.clamp_min(1e-7).pow(-0.5)[users]
            * item_degree.clamp_min(1e-7).pow(-0.5)[items]
        )

    # Compatibility helpers retained for code that calls the upstream API.
    def get_norm_adj_mat(self):
        return self.norm_adj

    def get_edge_info(self):
        return self.edge_indices, self.edge_values

    def _normalize_adj_m(self, indices, adj_size=None):
        del adj_size
        return self._normalize_bipartite_edges(indices)

    @property
    def freedom_keep_count(self):
        return max(
            1,
            min(
                self.num_interactions,
                int(self.num_interactions * (1.0 - self.dropout)),
            ),
        )

    @torch.no_grad()
    def _resample_freedom_adjacency(self):
        """Resample only FREEDOM's degree-sensitive training branch."""
        if self.dropout <= 0.0:
            self.freedom_adj = self.norm_adj
            return self.freedom_adj

        kept = torch.multinomial(
            self.edge_values,
            self.freedom_keep_count,
            replacement=False,
        )
        kept_local_edges = self.edge_indices[:, kept]
        kept_values = self._normalize_bipartite_edges(kept_local_edges)
        global_items = kept_local_edges[1] + self.n_users
        forward_edges = torch.stack(
            (kept_local_edges[0], global_items), dim=0
        )
        all_indices = torch.cat(
            (forward_edges, torch.flip(forward_edges, dims=(0,))), dim=1
        )
        all_values = torch.cat((kept_values, kept_values), dim=0)
        self.freedom_adj = torch.sparse_coo_tensor(
            all_indices,
            all_values,
            (self.n_nodes, self.n_nodes),
            device=all_values.device,
        ).coalesce()
        return self.freedom_adj

    def pre_epoch_processing(self):
        self._resample_freedom_adjacency()

    @property
    def masked_adj(self):
        """Upstream-compatible name for FREEDOM's sampled adjacency."""
        return self.freedom_adj

    @masked_adj.setter
    def masked_adj(self, adjacency):
        self.freedom_adj = adjacency

    def _freedom_ui_adjacency(self):
        if not self.training:
            return self.norm_adj
        if self.freedom_adj is None:
            return self._resample_freedom_adjacency()
        return self.freedom_adj

    def _build_or_load_mm_graph(self, config):
        dataset_path = os.path.abspath(
            os.path.join(config['data_path'], config['dataset'])
        )
        cache_name = 'mm_adj_freedomdsp_{}_{}.pt'.format(
            self.knn_k, int(10 * self.mm_image_weight)
        )
        cache_path = os.path.join(dataset_path, cache_name)

        if os.path.isfile(cache_path):
            try:
                mm_adj = torch.load(
                    cache_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                mm_adj = torch.load(cache_path, map_location=self.device)
            if tuple(mm_adj.shape) != (self.n_items, self.n_items):
                raise ValueError(
                    'Cached multimodal graph has shape {}, expected {}.'
                    .format(
                        tuple(mm_adj.shape),
                        (self.n_items, self.n_items),
                    )
                )
            if mm_adj.layout != torch.sparse_coo:
                mm_adj = mm_adj.to_sparse_coo()
            mm_adj = mm_adj.coalesce()
        else:
            with torch.no_grad():
                image_adj = None
                text_adj = None
                if self.image_embedding is not None:
                    image_adj = self.get_knn_adj_mat(
                        self.image_embedding.weight.detach()
                    )
                if self.text_embedding is not None:
                    text_adj = self.get_knn_adj_mat(
                        self.text_embedding.weight.detach()
                    )

                if image_adj is not None and text_adj is not None:
                    mm_adj = (
                        self.mm_image_weight * image_adj
                        + (1.0 - self.mm_image_weight) * text_adj
                    ).coalesce()
                elif image_adj is not None:
                    mm_adj = image_adj
                else:
                    mm_adj = text_adj

            os.makedirs(dataset_path, exist_ok=True)
            torch.save(mm_adj.cpu(), cache_path)
            mm_adj = mm_adj.to(self.device)

        self.register_buffer('mm_adj', mm_adj.coalesce())

    def get_knn_adj_mat(self, mm_embeddings):
        if self.n_items <= 0:
            raise ValueError('Cannot build an item graph without items.')
        topk = min(self.knn_k, self.n_items)
        normalized = F.normalize(mm_embeddings, p=2, dim=-1, eps=1e-12)
        similarity = torch.mm(normalized, normalized.transpose(0, 1))
        knn_indices = torch.topk(similarity, topk, dim=-1).indices

        rows = torch.arange(
            self.n_items, device=mm_embeddings.device
        ).unsqueeze(1).expand(-1, topk)
        indices = torch.stack(
            (rows.reshape(-1), knn_indices.reshape(-1)), dim=0
        )
        return self.compute_normalized_laplacian(
            indices, (self.n_items, self.n_items)
        )

    @staticmethod
    def compute_normalized_laplacian(indices, adjacency_size):
        values = torch.ones(
            indices.shape[1], dtype=torch.float32, device=indices.device
        )
        adjacency = torch.sparse_coo_tensor(
            indices, values, adjacency_size, device=indices.device
        ).coalesce()
        row_sum = torch.sparse.sum(adjacency, dim=1).to_dense()
        degree_inv_sqrt = row_sum.clamp_min(1e-7).pow(-0.5)
        row, col = adjacency.indices()
        normalized_values = (
            degree_inv_sqrt[row]
            * adjacency.values()
            * degree_inv_sqrt[col]
        )
        return torch.sparse_coo_tensor(
            adjacency.indices(),
            normalized_values,
            adjacency_size,
            device=indices.device,
        ).coalesce()

    def _propagate_ui_graph(self, adjacency, initial_embeddings):
        embeddings = [initial_embeddings]
        current = initial_embeddings
        for _ in range(self.n_ui_layers):
            current = torch.sparse.mm(adjacency, current)
            embeddings.append(current)
        return torch.stack(embeddings, dim=1).mean(dim=1)

    def _propagate_mm_graph(self, item_embeddings):
        propagated = item_embeddings
        for _ in range(self.n_mm_layers):
            propagated = torch.sparse.mm(self.mm_adj, propagated)
        return propagated

    def forward(self, adjacency=None):
        if adjacency is None:
            adjacency = self._freedom_ui_adjacency()

        initial = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight),
            dim=0,
        )
        ui_embeddings = self._propagate_ui_graph(adjacency, initial)
        user_embeddings, item_ui_embeddings = torch.split(
            ui_embeddings, (self.n_users, self.n_items), dim=0
        )
        item_mm_embeddings = self._propagate_mm_graph(
            self.item_id_embedding.weight
        )
        return user_embeddings, item_ui_embeddings + item_mm_embeddings

    @staticmethod
    def bpr_loss(users, positive_items, negative_items):
        positive_scores = torch.sum(users * positive_items, dim=1)
        negative_scores = torch.sum(users * negative_items, dim=1)
        return -F.logsigmoid(positive_scores - negative_scores).mean()

    def _modality_bpr_loss(
        self, user_embeddings, users, positive_items, negative_items
    ):
        loss = user_embeddings.new_zeros(())
        if self.text_embedding is not None:
            text_features = self.text_aux_projection(
                self.text_trs(self.text_embedding.weight)
            )
            loss = loss + self.bpr_loss(
                user_embeddings[users],
                text_features[positive_items],
                text_features[negative_items],
            )
        if self.image_embedding is not None:
            image_features = self.image_aux_projection(
                self.image_trs(self.image_embedding.weight)
            )
            loss = loss + self.bpr_loss(
                user_embeddings[users],
                image_features[positive_items],
                image_features[negative_items],
            )
        return loss

    def calculate_loss(self, interaction):
        users, positive_items, negative_items = interaction[:3]
        all_users, all_items = self.forward()
        ranking_loss = self.bpr_loss(
            all_users[users],
            all_items[positive_items],
            all_items[negative_items],
        )
        modality_loss = self._modality_bpr_loss(
            all_users, users, positive_items, negative_items
        )
        total_loss = ranking_loss + self.reg_weight * modality_loss
        self.latest_loss_components = {
            'bpr': ranking_loss.detach(),
            'auxiliary': modality_loss.detach(),
        }
        return total_loss

    def full_sort_predict(self, interaction):
        users, items = self.forward(self.norm_adj)
        return torch.matmul(
            users[interaction[0]], items.transpose(0, 1)
        )
