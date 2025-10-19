import os
import re
import gc
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import timm
import matplotlib.pyplot as plt

# --- MODÈLE ---
class ReidModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224_in21k', img_size=(256, 128)):
        super(ReidModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
        self.feature_dim = self.backbone.embed_dim
        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)

    def forward(self, x):
        features = self.backbone(x)
        bn_features = self.bottleneck(features)
        return F.normalize(bn_features)

# --- DATASETS ET SAMPLER ---
class TestReIDDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), img_path

class AdaptationDataset(Dataset):
    def __init__(self, image_paths, pseudo_labels, transform):
        self.image_paths = image_paths
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.pseudo_labels[index]
        img = self.transform(Image.open(img_path).convert('RGB'))
        return img, torch.tensor(label, dtype=torch.long)

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.data_source.pseudo_labels):
            if pid == -1: continue
            self.index_dic[int(pid)].append(index)
        
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        self.length = 0
        if self.num_identities >= self.num_pids_per_batch:
            self.length = (self.num_identities // self.num_pids_per_batch) * self.batch_size

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = np.random.choice(self.index_dic[pid], size=self.num_instances, replace=len(self.index_dic[pid]) < self.num_instances)
            batch_idxs_dict[pid].extend(idxs)
        
        avai_pids = self.pids.copy()
        final_idxs = []
        
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                final_idxs.extend(batch_idxs_dict[pid])
                avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self): return self.length

# --- FONCTIONS D'ENTRAÎNEMENT ET DE LABELLISATION ---
def mixup_contrastive(features, labels, alpha=0.2):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = features.size(0)
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2: return features, labels, None
    mixed_features, mixed_labels, mixup_info = [], [], []
    for i in range(batch_size):
        current_label = labels[i]
        different_labels = unique_labels[unique_labels != current_label]
        if len(different_labels) == 0: continue
        target_label = different_labels[torch.randint(len(different_labels), (1,))]
        target_indices = (labels == target_label).nonzero(as_tuple=True)[0]
        if target_indices.numel() == 0: continue
        target_idx = target_indices[torch.randint(len(target_indices), (1,))] if target_indices.numel() > 1 else target_indices
        mixed_feature = lam * features[i] + (1 - lam) * features[target_idx]
        mixed_features.append(mixed_feature)
        mixed_labels.append(current_label)
        mixup_info.append({'lam': lam, 'target_label': target_label.item()})
    if len(mixed_features) > 0: return torch.stack(mixed_features), torch.tensor(mixed_labels).to(features.device), mixup_info
    else: return features, labels, None

def advanced_triplet_loss_with_temperature(features, labels, margin, temperature, hard_mining_ratio=0.5):
    dist_mat = torch.cdist(features, features, p=2) / temperature
    N = features.size(0)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = ~is_pos
    is_pos.fill_diagonal_(False)
    dist_ap = torch.max(dist_mat * is_pos.float(), dim=1)[0]
    dist_an = torch.min(dist_mat * is_neg.float() + 1e9 * is_pos.float(), dim=1)[0]
    individual_losses = F.relu(dist_ap - dist_an + margin)
    n_hard = int(N * hard_mining_ratio)
    if n_hard > 0 and len(individual_losses) > n_hard:
        hard_indices = torch.topk(individual_losses, n_hard)[1]
        return individual_losses[hard_indices].mean()
    return individual_losses.mean()

def progressive_pseudo_labeling_with_confidence(features, epoch, args):
    print(f"Progressive Pseudo-Labeling - Epoch {epoch+1}/{args.adaptation_epochs}")
    progress = epoch / args.adaptation_epochs
    confidence_threshold = args.confidence_threshold_start - progress * (args.confidence_threshold_start - args.confidence_threshold_end)
    print(f"Seuil de confiance actuel: {confidence_threshold:.3f}")
    nn_calculator = NearestNeighbors(n_neighbors=args.dbscan_min_samples + 1, metric='cosine', n_jobs=-1)
    nn_calculator.fit(features)
    distances, _ = nn_calculator.kneighbors(features)
    eps = np.percentile(distances[:, -1], args.k_distance_percentile)
    initial_labels = DBSCAN(eps=eps, min_samples=args.dbscan_min_samples, metric='cosine', n_jobs=-1).fit_predict(features.cpu().numpy())
    confidences = np.zeros(len(features))
    cluster_ids = np.unique(initial_labels[initial_labels != -1])
    for cluster_id in cluster_ids:
        cluster_mask = (initial_labels == cluster_id)
        if np.sum(cluster_mask) < args.min_cluster_size: continue
        cluster_features = features[cluster_mask]
        cluster_centroid = cluster_features.mean(dim=0, keepdim=True)
        similarities = F.cosine_similarity(cluster_features, cluster_centroid)
        confidences[cluster_mask] = similarities.cpu().numpy()
    confident_mask = confidences >= confidence_threshold
    pseudo_labels = np.full(len(features), -1, dtype=int)
    new_label = 0
    for cluster_id in cluster_ids:
        cluster_mask = (initial_labels == cluster_id) & confident_mask
        if np.sum(cluster_mask) >= args.min_cluster_size:
            pseudo_labels[cluster_mask] = new_label
            new_label += 1
    if new_label > 1:
        pseudo_labels = merge_clusters_confident(features, pseudo_labels, confidences, args.merge_threshold)
    n_labeled = np.sum(pseudo_labels != -1)
    n_clusters = len(np.unique(pseudo_labels[pseudo_labels != -1]))
    print(f"Échantillons étiquetés après filtrage: {n_labeled}/{len(features)} ({n_labeled/len(features)*100:.1f}%)")
    print(f"Nombre de clusters (pré-raffinement): {n_clusters}")
    return pseudo_labels, confidences

def merge_clusters_confident(features, pseudo_labels, confidences, threshold):
    cluster_ids = np.unique(pseudo_labels[pseudo_labels != -1])
    if len(cluster_ids) <= 1: return pseudo_labels
    centroids = []
    for cid in cluster_ids:
        mask = (pseudo_labels == cid)
        w_centroid = (features[mask] * torch.tensor(confidences[mask], device=features.device).unsqueeze(1)).sum(dim=0) / torch.tensor(confidences[mask]).sum()
        centroids.append(w_centroid)
    centroids = torch.stack(centroids)
    sim_matrix = torch.mm(F.normalize(centroids), F.normalize(centroids).t())
    parent = {cid: cid for cid in cluster_ids}
    def find_root(cid):
        root = cid
        while root != parent[root]: root = parent[root]
        while cid != root: next_cid, parent[cid] = parent[cid], root; cid = next_cid
        return root
    def union(cid1, cid2):
        root1, root2 = find_root(cid1), find_root(cid2)
        if root1 != root2: parent[max(root1, root2)] = min(root1, root2)
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if sim_matrix[i, j] > threshold: union(cluster_ids[i], cluster_ids[j])
    new_labels = np.copy(pseudo_labels)
    for i, label in enumerate(pseudo_labels):
        if label != -1: new_labels[i] = find_root(label)
    n_before = len(cluster_ids)
    n_after = len(np.unique(new_labels[new_labels != -1]))
    if n_before > n_after: print(f"Fusion: {n_before} clusters -> {n_after} clusters")
    return new_labels

def refine_labels_with_camera_priors(features, pseudo_labels, camids, k, cross_cam_weight):
    print(f"Raffinage des labels avec les a-priori caméra (k={k}, weight={cross_cam_weight})...")
    labeled_mask = pseudo_labels != -1
    labeled_indices = np.where(labeled_mask)[0]
    if len(labeled_indices) < k + 1:
        print("Pas assez d'échantillons labellisés pour le raffinement. Passage.")
        return pseudo_labels, 0.0
    labeled_features = features[labeled_mask]
    labeled_pseudos = pseudo_labels[labeled_mask]
    labeled_camids = camids[labeled_mask]
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    nn.fit(labeled_features)
    _, indices = nn.kneighbors(labeled_features)
    refined_labels = pseudo_labels.copy()
    num_changed = 0
    for i in range(len(labeled_indices)):
        original_idx_in_full_array = labeled_indices[i]
        current_label = labeled_pseudos[i]
        current_camid = labeled_camids[i]
        neighbor_relative_indices = indices[i, 1:]
        votes = defaultdict(float)
        for neighbor_relative_idx in neighbor_relative_indices:
            neighbor_label = labeled_pseudos[neighbor_relative_idx]
            neighbor_camid = labeled_camids[neighbor_relative_idx]
            weight = cross_cam_weight if neighbor_camid != current_camid else 1.0
            votes[neighbor_label] += weight
        if not votes: continue
        best_label = max(votes, key=votes.get)
        if best_label != current_label:
            refined_labels[original_idx_in_full_array] = best_label
            num_changed += 1
    percent_changed = (num_changed / len(labeled_indices) * 100) if len(labeled_indices) > 0 else 0
    print(f"Raffinage terminé. {num_changed} labels ({percent_changed:.2f}%) ont été modifiés/corrigés.")
    return refined_labels, percent_changed

# --- FONCTIONS UTILITAIRES ET D'ÉVALUATION ---
def _parse_info_from_paths(paths):
    return (np.array([int(re.search(r'([-\d]+)_c', os.path.basename(p)).group(1)) for p in paths]),
            np.array([int(re.search(r'_c(\d+)', os.path.basename(p)).group(1)) for p in paths]))

def extract_features(model, loader, device):
    model.eval()
    features_list, path_list = [], []
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="Extraction des features"):
            features_list.append(model(imgs.to(device)).cpu())
            path_list.extend(paths)
    return torch.cat(features_list), path_list

def evaluate_from_distmat(distmat, q_pids, g_pids, q_camids, g_camids):
    indices = np.argsort(distmat, axis=1)
    cmc, all_ap, num_valid_q = np.zeros(len(g_pids)), [], 0
    for i in range(len(q_pids)):
        q_pid, q_camid = q_pids[i], q_camids[i]
        matches = (g_pids[indices[i]] == q_pid).astype(np.int32)
        remove = (g_pids[indices[i]] == q_pid) & (g_camids[indices[i]] == q_camid)
        orig_cmc = matches[~remove]
        if not np.any(orig_cmc): continue
        num_valid_q += 1
        cmc_idx = np.where(orig_cmc == 1)[0][0]
        cmc[cmc_idx:] += 1
        pos_idx = np.where(orig_cmc == 1)[0]
        ap = sum((k + 1) / (pos + 1) for k, pos in enumerate(pos_idx)) / len(pos_idx)
        all_ap.append(ap)
    if num_valid_q == 0: return 0.0, np.zeros_like(cmc)
    return np.mean(all_ap), cmc / num_valid_q

def load_pretrained_weights(model, path, device):
    try:
        sd = torch.load(path, map_location=device)
        sd = sd.get('model', sd.get('state_dict', sd))
        new_sd = {k.replace('image_encoder.', 'backbone.'): v for k, v in sd.items() if k.startswith('image_encoder.')}
        model.load_state_dict(new_sd, strict=False)
        print(f"Poids pré-entraînés chargés depuis: {path}")
    except Exception as e: print(f"Erreur de chargement: {e}"); raise e

def comprehensive_evaluation(model, query_loader, gallery_loader, q_pids, g_pids, q_camids, g_camids, device, epoch=None):
    model.eval()
    desc = f"Évaluation Epoch {epoch+1}" if epoch is not None else "Évaluation finale"
    print(desc + "...")
    q_feat, _ = extract_features(model, query_loader, device)
    g_feat, _ = extract_features(model, gallery_loader, device)
    distmat_cosine = 1 - torch.mm(q_feat, g_feat.t()).cpu().numpy()
    mAP_cosine, cmc_cosine = evaluate_from_distmat(distmat_cosine, q_pids, g_pids, q_camids, g_camids)
    print(f"  => Cosine - mAP: {mAP_cosine:.2%}, Rank-1: {cmc_cosine[0]:.2%}")
    return {'mAP_cosine': mAP_cosine, 'rank1_cosine': cmc_cosine[0], 'rank5_cosine': cmc_cosine[4], 'rank10_cosine': cmc_cosine[9]}

# --- NOUVEAU : FONCTION DE GÉNÉRATION DES GRAPHIQUES ---
def generate_and_save_plots(history, baseline_results, final_results, output_dir):
    """
    Génère et sauvegarde tous les graphiques d'analyse de l'entraînement.
    """
    print("\n" + "="*50 + "\n### GÉNÉRATION DES GRAPHIQUES D'ANALYSE ###\n" + "="*50)
    
    graphs_path = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_path, exist_ok=True)
    
    epochs = [h['epoch'] for h in history]
    mAP_values = [h['mAP_cosine'] * 100 for h in history]
    rank1_values = [h['rank1_cosine'] * 100 for h in history]
    loss_values = [h['loss'] for h in history]
    labeled_counts = [h['n_labeled'] for h in history]
    refinement_impact = [h['percent_changed'] for h in history]

    best_epoch_idx = np.argmax(mAP_values)
    best_epoch = epochs[best_epoch_idx]

    # --- Graphique 1: Évolution des Performances ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, mAP_values, marker='o', linestyle='-', label='mAP (%)')
    plt.plot(epochs, rank1_values, marker='s', linestyle='--', label='Rank-1 Accuracy (%)')
    plt.axvline(x=best_epoch, color='r', linestyle=':', label=f'Meilleur Modèle (Époque {best_epoch})')
    plt.title('Évolution des Performances du Modèle durant l\'Adaptation', fontsize=16, fontweight='bold')
    plt.xlabel('Époque d\'Adaptation', fontsize=12)
    plt.ylabel('Performance (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(graphs_path, '1_evolution_performances.png'))
    plt.close()
    print("Graphique 1/4 sauvegardé : Évolution des Performances")

    # --- Graphique 2: Comparaison Avant vs. Après ---
    labels = ['mAP', 'Rank-1']
    base_metrics = [baseline_results['mAP_cosine'] * 100, baseline_results['rank1_cosine'] * 100]
    final_metrics = [final_results['mAP_cosine'] * 100, final_results['rank1_cosine'] * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - width/2, base_metrics, width, label='Modèle de Base', color='skyblue')
    rects2 = ax.bar(x + width/2, final_metrics, width, label='Modèle Adapté (Final)', color='royalblue')

    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title('Comparaison : Avant vs. Après Adaptation', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f%%')
    ax.bar_label(rects2, padding=3, fmt='%.2f%%')
    fig.tight_layout()
    plt.savefig(os.path.join(graphs_path, '2_comparaison_avant_apres.png'))
    plt.close()
    print("Graphique 2/4 sauvegardé : Comparaison Avant vs. Après")

    # --- Graphique 3: Dynamique de l'Entraînement ---
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = 'tab:red'
    ax1.set_xlabel('Époque d\'Adaptation', fontsize=12)
    ax1.set_ylabel('Perte Moyenne (Loss)', color=color, fontsize=12)
    ax1.plot(epochs, loss_values, color=color, marker='.', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Nombre d\'Échantillons Étiquetés', color=color, fontsize=12)
    ax2.plot(epochs, labeled_counts, color=color, marker='.', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Dynamique de l\'Entraînement : Perte vs. Pseudo-Labels', fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(graphs_path, '3_dynamique_entrainement.png'))
    plt.close()
    print("Graphique 3/4 sauvegardé : Dynamique de l'Entraînement")

    # --- Graphique 4: Impact du Raffinement ---
    plt.figure(figsize=(12, 7))
    plt.bar(epochs, refinement_impact, color='teal', alpha=0.7)
    plt.title('Impact du Raffinement par Caméra à chaque Époque', fontsize=16, fontweight='bold')
    plt.xlabel('Époque d\'Adaptation', fontsize=12)
    plt.ylabel('Pourcentage de Labels Modifiés (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(graphs_path, '4_impact_raffinement.png'))
    plt.close()
    print("Graphique 4/4 sauvegardé : Impact du Raffinement par Caméra")

# --- FONCTION PRINCIPALE ---
def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    best_model_save_path = os.path.join(args.output_path, "best_model_camera_refined.pth")
    
    image_size = tuple(args.image_size)
    transform_test = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    transform_train = transforms.Compose([
        transforms.Resize(image_size, interpolation=3), transforms.RandomHorizontalFlip(), 
        transforms.Pad(10), transforms.RandomCrop(image_size), 
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3), transforms.RandomErasing(p=0.5)
    ])

    print(f"Chargement des données depuis: {args.duke_data_path}")
    query_dataset = TestReIDDataset(os.path.join(args.duke_data_path, 'query'), transform_test)
    gallery_dataset = TestReIDDataset(os.path.join(args.duke_data_path, 'bounding_box_test'), transform_test)
    train_dataset = TestReIDDataset(os.path.join(args.duke_data_path, 'bounding_box_train'), transform_test)
    
    q_pids, q_camids = _parse_info_from_paths(query_dataset.img_paths)
    g_pids, g_camids = _parse_info_from_paths(gallery_dataset.img_paths)
    train_pids, train_camids = _parse_info_from_paths(train_dataset.img_paths)

    query_loader = DataLoader(query_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    train_feature_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    print("\n" + "="*50 + "\n### ÉVALUATION DU MODÈLE DE BASE (SANS ADAPTATION) ###\n" + "="*50)
    baseline_model = ReidModel(img_size=image_size).to(args.device)
    load_pretrained_weights(baseline_model, args.market_model_path, args.device)
    baseline_results = comprehensive_evaluation(baseline_model, query_loader, gallery_loader, q_pids, g_pids, q_camids, g_camids, args.device)
    del baseline_model; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "="*50 + "\n### DÉBUT DE L'ADAPTATION DE DOMAINE ###\n" + "="*50)
    model = ReidModel(img_size=image_size).to(args.device)
    load_pretrained_weights(model, args.market_model_path, args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.adaptation_lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    patience, best_mAP, avg_loss = 0, 0.0, 0.0
    training_history = []
    
    batch_size = args.p * args.k

    for epoch in range(args.adaptation_epochs):
        print(f"\n--- ADAPTATION EPOCH {epoch+1}/{args.adaptation_epochs} ---")
        
        model.eval()
        all_features, _ = extract_features(model, train_feature_loader, args.device)
        pseudo_labels, _ = progressive_pseudo_labeling_with_confidence(all_features, epoch, args)
        
        refined_pseudo_labels, percent_changed = refine_labels_with_camera_priors(
            all_features.cpu(), pseudo_labels, train_camids,
            k=args.camera_refinement_k,
            cross_cam_weight=args.camera_refinement_weight
        )
        
        adapt_dataset_instance = AdaptationDataset(train_dataset.img_paths, refined_pseudo_labels, transform_train)
        sampler = RandomIdentitySampler(adapt_dataset_instance, batch_size, args.k)
        
        if len(sampler) > 0:
            adapt_loader = DataLoader(adapt_dataset_instance, batch_size=batch_size, sampler=sampler, num_workers=args.num_workers, drop_last=True)
            model.train()
            total_loss, num_batches = 0.0, 0
            
            progress = epoch / (args.adaptation_epochs - 1) if args.adaptation_epochs > 1 else 1
            current_temperature = args.temperature_start - progress * (args.temperature_start - args.temperature_end)
            
            for imgs, labels in tqdm(adapt_loader, desc=f"Training Epoch {epoch+1}"):
                optimizer.zero_grad()
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                features = model(imgs)
                
                loss = advanced_triplet_loss_with_temperature(features, labels, args.triplet_margin, current_temperature, args.hard_mining_ratio)
                
                if loss is not None and loss.item() > 0:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Loss moyenne: {avg_loss:.4f}")
            lr_scheduler.step()
        else:
            print("Aucun batch d'entraînement généré, passage de l'époque.")
            avg_loss = 0.0

        validation_results = comprehensive_evaluation(model, query_loader, gallery_loader, q_pids, g_pids, q_camids, g_camids, args.device, epoch)
        current_mAP = validation_results['mAP_cosine']
        history_entry = {
            'epoch': epoch + 1, 'loss': avg_loss, 
            'n_labeled': np.sum(refined_pseudo_labels != -1),
            'percent_changed': percent_changed, 
            **validation_results
        }
        training_history.append(history_entry)
        
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            patience = 0
            print(f"🏆 Nouveau meilleur mAP: {best_mAP:.2%}! Sauvegarde du modèle...")
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'best_mAP': best_mAP}, best_model_save_path)
        else:
            patience += 1
            print(f"Patience: {patience}/{args.early_stopping_patience}")
        
        if patience >= args.early_stopping_patience:
            print(f"🛑 Early stopping. Meilleur mAP obtenu: {best_mAP:.2%}")
            break
    
    print("\n" + "="*50 + "\n### ÉVALUATION FINALE DU MEILLEUR MODÈLE ###\n" + "="*50)
    final_results = {}
    if os.path.exists(best_model_save_path):
        print("Chargement du meilleur modèle sauvegardé pour l'évaluation finale...")
        checkpoint = torch.load(best_model_save_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modèle de l'époque {checkpoint['epoch']} (mAP: {checkpoint['best_mAP']:.2%}) chargé.")
        
        final_results = comprehensive_evaluation(model, query_loader, gallery_loader, q_pids, g_pids, q_camids, g_camids, args.device)
        
        print("\n" + "="*60 + "\n### RÉSULTATS FINAUX ###\n" + "="*60)
        print(f"BASELINE (SANS ADAPTATION):")
        print(f"  - mAP: {baseline_results['mAP_cosine']:.2%}, Rank-1: {baseline_results['rank1_cosine']:.2%}")
        
        print(f"\nMODÈLE ADAPTÉ (AVEC RAFFINEMENT PAR CAMÉRA):")
        print(f"  - mAP: {final_results['mAP_cosine']:.2%} (Amélioration: {final_results['mAP_cosine']/baseline_results['mAP_cosine']-1:+.1%})")
        print(f"  - Rank-1: {final_results['rank1_cosine']:.2%} (Amélioration: {final_results['rank1_cosine']/baseline_results['rank1_cosine']-1:+.1%})")
    else:
        print("Aucun modèle n'a été sauvegardé. Utilisation du dernier modèle pour l'évaluation.")
        final_results = comprehensive_evaluation(model, query_loader, gallery_loader, q_pids, g_pids, q_camids, g_camids, args.device)

    # Génération des graphiques si l'entraînement a eu lieu
    if training_history:
        generate_and_save_plots(training_history, baseline_results, final_results, args.output_path)

    print("\n🎉 ADAPTATION TERMINÉE! 🎉")

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'adaptation de domaine pour le Person Re-Identification")
    parser.add_argument('--duke_data_path', type=str, default='./data/DukeMTMC-reID', help="Chemin vers le dossier du dataset DukeMTMC-reID")
    parser.add_argument('--market_model_path', type=str, default='./pretrained_models/Market1501_clipreid_12x12sie_ViT-B-16_60.pth', help="Chemin vers le modèle pré-entraîné")
    parser.add_argument('--output_path', type=str, default='./output', help="Dossier où sauvegarder le meilleur modèle et les graphiques")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Appareil à utiliser ('cuda' ou 'cpu')")
    parser.add_argument('--num_workers', type=int, default=2, help="Nombre de workers pour les DataLoaders")
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 128], help="Taille de l'image (hauteur largeur)")
    parser.add_argument('--adaptation_epochs', type=int, default=40, help="Nombre d'époques pour l'adaptation")
    parser.add_argument('--adaptation_lr', type=float, default=3.5e-5, help="Taux d'apprentissage")
    parser.add_argument('--p', type=int, default=16, help="Nombre d'identités par batch")
    parser.add_argument('--k', type=int, default=4, help="Nombre d'instances par identité")
    parser.add_argument('--eval_batch_size', type=int, default=128, help="Taille du batch pour l'évaluation")
    parser.add_argument('--early_stopping_patience', type=int, default=8, help="Patience pour l'early stopping")
    parser.add_argument('--dbscan_min_samples', type=int, default=4)
    parser.add_argument('--k_distance_percentile', type=int, default=30)
    parser.add_argument('--merge_threshold', type=float, default=0.9)
    parser.add_argument('--confidence_threshold_start', type=float, default=0.8)
    parser.add_argument('--confidence_threshold_end', type=float, default=0.6)
    parser.add_argument('--min_cluster_size', type=int, default=4)
    parser.add_argument('--temperature_start', type=float, default=0.1)
    parser.add_argument('--temperature_end', type=float, default=0.05)
    parser.add_argument('--hard_mining_ratio', type=float, default=0.5)
    parser.add_argument('--triplet_margin', type=float, default=0.3)
    parser.add_argument('--camera_refinement_k', type=int, default=10)
    parser.add_argument('--camera_refinement_weight', type=float, default=2.0)
    
    args = parser.parse_args()
    main(args)