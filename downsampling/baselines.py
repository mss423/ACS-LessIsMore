# -*- coding: utf-8 -*-
"""
Modified script incorporating CoresetSelection methods
including mislabel_mask as optional pre-filtering step (Cleaned Version)
"""
import random, math
import torch
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# --- Assume imports/mocks/helpers defined as before ---
class MockKCGSampler:
    def __init__(self, X, y=None, seed=0):
        self.X = X
        self.n_samples = X.shape[0] if X is not None else 0
        self.indices = np.arange(self.n_samples)
        if X is None: 
            pass

    def select_batch_(self, already_selected, N):
        available_indices = np.setdiff1d(self.indices, already_selected if already_selected else [])
        count = min(N,len(available_indices))
        if count <= 0: 
            return []

        selected = np.random.choice(available_indices, count, replace=False)
        return selected.tolist()

kCenterGreedy = MockKCGSampler

def get_median(features, targets):
    if isinstance(features, torch.Tensor): 
        features=features.numpy()
    if isinstance(targets, torch.Tensor): 
        targets=targets.numpy()
    elif isinstance(targets, list): 
        targets=np.array(targets)
    if features.shape[0] == 0:
        unique_targets = np.unique(targets) if targets.size>0 else []
        num_classes = len(unique_targets) if len(unique_targets) > 0 else 0
        feature_dim = features.shape[-1] if len(features.shape)>1 else 0
        return np.zeros((num_classes,feature_dim), dtype=features.dtype)

    unique_targets = np.unique(targets)
    num_classes = len(unique_targets)
    prot = np.zeros((num_classes,features.shape[-1]), dtype=features.dtype)
    target_map = {label: i for i,label in enumerate(unique_targets)}

    for label in unique_targets:
        class_indices = np.where(targets==label)[0]
        if len(class_indices) > 0:
             class_features = features[class_indices,:]
             if class_features.ndim == 1: 
                class_features=class_features[np.newaxis,:]

             prot[target_map[label]] = np.median(class_features,axis=0,keepdims=False)
        else: 
            prot[target_map[label]] = np.zeros(features.shape[-1],dtype=features.dtype)
    return prot

def get_distance(features, labels):
    if isinstance(features, torch.Tensor): 
        features = features.numpy()
    if isinstance(labels, torch.Tensor): 
        labels = labels.numpy()
    elif isinstance(labels, list): 
        labels = np.array(labels)
    if features.shape[0]==0: 
        return np.array([], dtype=np.float64)
    prots = get_median(features,labels)

    if prots.shape[0] == 0: 
        return np.zeros(features.shape[0])

    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]), dtype=features.dtype)
    unique_labels =n p.unique(labels)
    target_map = {label: i for i,label in enumerate(unique_labels)}
    for label in unique_labels:
        if label in target_map:
            class_indices = np.where(labels==label)[0]
            median_index = target_map[label]
            if median_index<prots.shape[0]: 
                prots_for_each_example[class_indices,:] = prots[median_index]
            else: 
                prots_for_each_example[class_indices,:] = np.zeros(prots.shape[-1],dtype=features.dtype)
        else: 
            class_indices = np.where(labels==label)[0]
            prots_for_each_example[class_indices,:] = np.zeros(prots.shape[-1], dtype=features.dtype)

    distance = np.linalg.norm(features-prots_for_each_example,axis=1)
    return distance
# Assume bin_allocate defined if needed

# START OF CORESET SCRIPT CLASS
class CoresetSelection(object):
    # --- Keep previous static methods: ---
    # random_selection, moderate_selection, stratified_kcenter, score_monotonic_selection, mislabel_mask
    @staticmethod
    def random_selection(total_num, num):
        if isinstance(total_num, torch.Tensor): 
            total_num = total_num.item()
        elif not isinstance(total_num, int): 
            total_num = int(total_num)
        if isinstance(num, torch.Tensor): 
            num = num.item()
        elif not isinstance(num, int): 
            num = int(num)

        actual_num = min(num,total_num)
        if total_num == 0 or actual_num <= 0: 
            return torch.tensor([], dtype=torch.long)
        score_random_index = torch.randperm(total_num)
        return score_random_index[:actual_num]

    @staticmethod
    def moderate_selection(data_score, ratio, features):
        if isinstance(features, torch.Tensor): 
            features = features.numpy()

        targets_list = data_score['targets']
        if isinstance(targets_list, torch.Tensor): 
            targets_list = targets_list.numpy()
        elif isinstance(targets_list, list): 
            targets_list = np.array(targets_list)
        if features.shape[0] == 0: 
            return np.array([], dtype=int)

        def get_prune_idx(keep_ratio_param, distance):
            n_samples = distance.shape[0]
            prune_rate = max(0.0, min(1.0, 1.0 - keep_ratio_param))
            low_p = (1.0-prune_rate)/2.0
            high_p = 1.0-low_p
            if not isinstance(distance,np.ndarray): 
                distance=np.array(distance)
            if n_samples==0 or distance.size==0: 
                return np.array([], dtype=int)

            sorted_idx = distance.argsort()
            low_idx = max(0, round(n_samples*low_p))
            high_idx = min(n_samples, round(n_samples*high_p))
            prune_low = sorted_idx[:low_idx]
            prune_high = sorted_idx[high_idx:]

            return np.concatenate((prune_low,prune_high))

        distance=get_distance(features,targets_list)
        prune_ids=get_prune_idx(ratio,distance)
        return prune_ids

    @staticmethod
    def stratified_kcenter(embed_data, data_labels, K, n_stratas=10):
        total_num = embed_data.shape[0]
        if K >= total_num: 
            return list(range(total_num))
        if K <= 0 or total_num==0: return []

        scores = torch.rand(total_num)
        min_score, max_score = torch.min(scores), torch.max(scores)*1.0001
        step = (max_score-min_score)/n_stratas
        strata_indices = [[] for _ in range(n_stratas)]; strata_counts=torch.zeros(n_stratas,dtype=torch.long)

        for i in range(n_stratas):
            start, end = min_score+i*step, min_score+(i+1)*step
            mask = torch.logical_and(scores>=start, scores<end)
            indices_in_strata = torch.where(mask)[0].tolist()
            strata_indices[i] = indices_in_strata
            strata_counts[i] = len(indices_in_strata)

        budgets = torch.zeros(n_stratas,dtype=torch.long)
        remaining_k = K
        active_strata_indices = [i for i,count in enumerate(strata_counts) if count>0]
        remaining_bins = len(active_strata_indices)
        sorted_active_indices = sorted(active_strata_indices, key=lambda i: strata_counts[i])
        temp_budgets = {i:0 for i in active_strata_indices}

        for i in sorted_active_indices:
            current_remaining_bins = max(1,len([idx for idx in sorted_active_indices if temp_budgets.get(idx, 0)==0 and strata_counts[idx]>0]))
            avg_budget = remaining_k//current_remaining_bins
            current_budget = max(0, min(strata_counts[i].item(), avg_budget))
            temp_budgets[i] = current_budget
            remaining_k = max(0, remaining_k - current_budget)
        if remaining_k > 0:
            for i in reversed(sorted_active_indices):
                if remaining_k==0: break
                if temp_budgets[i] < strata_counts[i]: 
                    temp_budgets[i]+=1
                    remaining_k-=1

        for i in range(n_stratas): 
            budgets[i] = temp_budgets.get(i,0)

        selected_indices_final=[]
        for i in range(n_stratas):
            pool=strata_indices[i]
            budget=budgets[i].item()

            if not pool or budget==0: continue
            if budget >= len(pool): 
                selected_in_strata = pool
            else:
                 if isinstance(embed_data,pd.DataFrame): 
                    strata_embeds = embed_data.iloc[pool].values
                 elif isinstance(embed_data,np.ndarray): 
                    strata_embeds = embed_data[pool]
                 elif isinstance(embed_data,torch.Tensor): 
                    strata_embeds = embed_data[torch.tensor(pool,dtype=torch.long)]
                 else: 
                    raise TypeError("Unsupported embed_data type")
                 if strata_embeds.shape[0]>0:
                    sampler = kCenterGreedy(X=strata_embeds, seed=0)
                    relative_indices = sampler.select_batch_(None,budget)
                    selected_in_strata = [pool[j] for j in relative_indices] if relative_indices else []
                 else: 
                    selected_in_strata=[]

            selected_indices_final.extend(selected_in_strata)
        current_selected_count = len(selected_indices_final)

        if current_selected_count < K:
            n_needed = K-current_selected_count
            pool_available = list(set(range(total_num)) - set(selected_indices_final))
            n_to_add = min(n_needed,len(pool_available))
            selected_indices_final.extend(random.sample(pool_available,k=n_to_add))
        random.shuffle(selected_indices_final)
        return selected_indices_final[:K]

    @staticmethod
    def score_monotonic_selection(data_score, key, ratio, descending, class_balanced):
        score = data_score[key];
        if not isinstance(score, torch.Tensor): 
            score = torch.tensor(score)
        targets_list = data_score['targets']
        if not isinstance(targets_list, torch.Tensor): 
            targets_list = torch.tensor(targets_list)
        n_total_samples = score.shape[0]

        if n_total_samples==0: 
            return torch.tensor([],dtype=torch.long)
        score_sorted_index = score.argsort(descending=descending)
        target_coreset_num = max(0, min(n_total_samples, int(round(ratio*n_total_samples))))

        if class_balanced:
            all_original_indices = torch.arange(n_total_samples)
            selected_indices_list = []
            targets_sorted_by_score = targets_list[score_sorted_index]
            targets_unique = torch.unique(targets_sorted_by_score)

            if targets_unique.numel() == 0: 
                return score_sorted_index[:target_coreset_num]
            for target_label in targets_unique:
                target_mask_in_sorted = (targets_sorted_by_score==target_label)
                indices_in_sorted_list = torch.where(target_mask_in_sorted)[0]
                n_in_class = len(indices_in_sorted_list)
                n_to_select_for_class = max(0, min(n_in_class, int(round(n_in_class*ratio))))

                if n_to_select_for_class > 0:
                    selected_indices_for_class_relative = indices_in_sorted_list[:n_to_select_for_class]
                    original_indices_for_class = score_sorted_index[selected_indices_for_class_relative]
                    selected_indices_list.extend(original_indices_for_class.tolist())

            selected_index_tensor = torch.tensor(selected_indices_list,dtype=torch.long) if selected_indices_list else torch.tensor([],dtype=torch.long)
            return selected_index_tensor
        else: 
            return score_sorted_index[:target_coreset_num]

    @staticmethod
    def mislabel_mask(data_score, mis_key, mis_num, mis_descending, coreset_key=None):
        """ Returns indices to KEEP after pruning 'mis_num' based on 'mis_key'. """
        mis_score = data_score[mis_key]
        if not isinstance(mis_score, torch.Tensor): 
            mis_score = torch.tensor(mis_score)
        n_total = mis_score.shape[0]
        mis_num = max(0, min(n_total, int(mis_num)))

        if n_total == 0 or mis_num == 0: 
            return np.arange(n_total)
        mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
        easy_index = mis_score_sorted_index[mis_num:] # Indices kept
        return easy_index.numpy() # Return numpy array of kept indices

# END OF CORESET SCRIPT CLASS

# --- Wrapper Functions ---
def random_sample_wrapper(embed_data_shape_info, data_labels, K):
    total_num = embed_data_shape_info[0]
    selected_indices_tensor = CoresetSelection.random_selection(total_num, K)
    return selected_indices_tensor.tolist()

def stratified_kcenter_wrapper(embed_data, data_labels, K, **kwargs):
    if isinstance(embed_data, pd.DataFrame): embed_data_np = embed_data.values
    else: embed_data_np = embed_data
    if isinstance(data_labels, pd.Series): data_labels_list = data_labels.tolist()
    else: data_labels_list = data_labels
    selected_indices = CoresetSelection.stratified_kcenter(embed_data=embed_data_np, data_labels=data_labels_list, K=K, n_stratas=kwargs.get('n_stratas', 10))
    return selected_indices

def moderate_selection_wrapper(embed_data, data_labels, K):
    total_num = embed_data.shape[0]
    if total_num == 0: return []
    K = int(K)
    if K <= 0: 
        return []
    if K >= total_num: 
        return list(range(total_num))

    keep_ratio = K/total_num
    prune_ratio_each_end = (1.0-keep_ratio)/2.0
    low_keep_p = prune_ratio_each_end
    high_keep_p = 1.0-low_keep_p

    if isinstance(embed_data,pd.DataFrame): 
        features_np = embed_data.values
    elif isinstance(embed_data,torch.Tensor): 
        features_np = embed_data.numpy()
    else: 
        features_np = embed_data

    if isinstance(data_labels,pd.Series): 
        labels_np = data_labels.values
    elif isinstance(data_labels,list): 
        labels_np = np.array(data_labels)
    elif isinstance(data_labels,torch.Tensor): 
        labels_np = data_labels.numpy()
    else: 
        labels_np = data_labels

    distance = get_distance(features_np,labels_np)
    if distance.size == 0: 
        return []
    sorted_idx = distance.argsort()
    low_idx = max(0,round(total_num*low_keep_p))
    high_idx = min(total_num,round(total_num*high_keep_p))
    keep_indices_np = sorted_idx[low_idx:high_idx]
    final_list = sorted(keep_indices_np.tolist())

    if len(final_list) != K and K < total_num: # Adjust size if needed
        if len(final_list) > K:
            trim_count = len(final_list)-K
            trim_start = trim_count//2
            trim_end = trim_count-trim_start
            final_list = final_list[trim_start:-trim_end] if trim_end>0 else final_list[trim_start:]
    return final_list

def score_monotonic_wrapper(score_array, data_labels, K, descending, class_balanced):
    """
    Wrapper for score_monotonic_selection (Cleaned - Explicit keep_ratio assignment).
    """
    total_num = len(score_array)
    if total_num == 0:
        return []

    K = int(K)
    # Check original K for non-positive selection request
    if K <= 0:
        return []

    K_actual = min(K, total_num) # Determine actual number to select

    # Explicitly assign keep_ratio
    if total_num > 0:
        keep_ratio = K_actual / total_num
    else:
        keep_ratio = 0.0 # Assign 0.0 if total_num is 0

    # Prepare data_score dictionary
    if isinstance(data_labels, pd.Series): data_labels = data_labels.tolist()
    if isinstance(score_array, pd.Series): score_array = score_array.values # Ensure not pandas Series
    data_score_dict = {'targets': data_labels, 'the_score': score_array }

    # Call the original function using the assigned keep_ratio
    selected_indices_tensor = CoresetSelection.score_monotonic_selection(
        data_score=data_score_dict,
        key='the_score',
        ratio=keep_ratio, # Use the calculated keep_ratio
        descending=descending,
        class_balanced=class_balanced
    )

    final_list = selected_indices_tensor.tolist()

    # Optional adjustment if exact K is needed, esp. after class balancing
    if len(final_list) != K_actual and K_actual < total_num:
         if len(final_list) > K_actual:
              final_list = final_list[:K_actual] # Simple truncation
         # else: less than K_actual selected, return as is.

    return final_list


# --- Main Sampling Function (Reverted to Pre-Filtering Mislabel Mask) ---
def coreset_sample(data_df, Ks, method='random',
                   embed_col='embeddings', label_col='label',
                   score_col=None, descending=True, class_balanced=False,
                   use_mislabel_mask=False, # Flag to enable mask
                   mislabel_score_col=None, # Score column for mask
                   mislabel_prune_num=0,    # Number to prune via mask
                   mislabel_descending=True,# Sort order for mask score
                   **kwargs):
    """
    Main function with mislabel_mask as optional pre-filtering step.

    Args:
        ... (Args same as before, method does NOT include 'mislabel_mask') ...

    Returns:
        dict: {K: list_of_selected_original_indices}.
    """
    print(f"\n=== Running Coreset Selection ===")
    print(f"Method: {method}, K values: {Ks}")
    if use_mislabel_mask:
        print(f"Applying Mislabel Mask: score='{mislabel_score_col}', prune_num={mislabel_prune_num}, descending={mislabel_descending}")

    total_num_samples_original = len(data_df)
    active_indices = np.arange(total_num_samples_original) # Start with all indices

    # --- Apply Mislabel Mask (Pre-filtering) ---
    if use_mislabel_mask and mislabel_score_col and mislabel_prune_num > 0:
        # Check if the specified column exists in the DataFrame
        if mislabel_score_col not in data_df.columns:
             raise ValueError(f"Mislabel mask requires column '{mislabel_score_col}', which was not found in the DataFrame columns: {data_df.columns.tolist()}")

        # If check passes, extract the score array
        mislabel_score_array = data_df[mislabel_score_col].values
        temp_data_score = {'mislabel_score': mislabel_score_array}

        # Get indices to KEEP after pruning
        active_indices = CoresetSelection.mislabel_mask(
            data_score=temp_data_score,
            mis_key='mislabel_score',
            mis_num=mislabel_prune_num,
            mis_descending=mislabel_descending
        )
        print(f"After Mislabel Mask, using {len(active_indices)} active samples.")
        if len(active_indices) == 0:
            print("Warning: Mislabel mask removed all samples. Returning empty results.")
            return {int(k): [] for k in Ks} # Return empty dict if all pruned

    elif use_mislabel_mask:
         print("Warning: 'use_mislabel_mask' is True, but 'mislabel_score_col' not provided or 'mislabel_prune_num' is 0. Skipping mask.")

    # --- Prepare Data for Selected Method (using active_indices) ---
    active_data_df = data_df.iloc[active_indices] # Use .iloc for integer-based indexing
    active_total_num = len(active_indices) # Number of samples remaining

    # Ensure active_indices is numpy array for potential advanced indexing later
    if not isinstance(active_indices, np.ndarray):
        active_indices = np.array(active_indices)


    embed_data_active = None; data_labels_active = None; score_array_active = None

    # Extract labels from the active subset
    if label_col in active_data_df.columns:
        data_labels_active = active_data_df[label_col].values # Get as numpy array
    else:
        data_labels_active = np.zeros(active_total_num, dtype=int) # Dummy labels for active set

    # Extract other data required by the *specific* method from the active subset
    if method in ['stratified_kcenter', 'moderate']:
        if embed_col and embed_col in active_data_df.columns:
             # Ensure stacking works even if only one row remains
             embed_values = active_data_df[embed_col].values
             if active_total_num == 1: embed_data_active = embed_values[0][np.newaxis, :] # Reshape if single embedding
             elif active_total_num > 1: embed_data_active = np.stack(embed_values)
             else: embed_data_active = np.array([]).reshape(0,0) # Empty case
        else: raise ValueError(f"Method '{method}' requires embed_col '{embed_col}'.")
    if method == 'score_monotonic':
        if score_col and score_col in active_data_df.columns:
            score_array_active = active_data_df[score_col].values
        else: raise ValueError(f"Method '{method}' requires score_col '{score_col}'.")

    # --- Main Loop over Ks ---
    selected_samples = {}
    Ks = [int(k) for k in Ks]

    for K in tqdm(Ks, desc=f"Processing Ks ({method})"):
        if K < 0: selected_samples[K] = []; continue

        # Adjust K relative to the number of *active* samples
        K_adjusted = min(K, active_total_num)
        if K_adjusted != K and active_total_num > 0 : # Avoid printing if no samples left
             print(f"Note: Requested K={K}, but only {active_total_num} samples active after filtering. Selecting max {K_adjusted}.")
        elif K_adjusted != K and active_total_num == 0:
             print(f"Note: Requested K={K}, but 0 samples active after filtering.")


        if K_adjusted == 0 or active_total_num == 0: # If adjusted K is 0 or no active samples, select nothing
             selected_indices_relative = []
        elif method == 'random':
             # Pass shape info for the *active* set
             selected_indices_relative = random_sample_wrapper((active_total_num, 0), data_labels_active, K_adjusted)
        elif method == 'stratified_kcenter':
            if embed_data_active is None: raise ValueError("Embeddings required.")
            selected_indices_relative = stratified_kcenter_wrapper(embed_data_active, data_labels_active, K_adjusted, **kwargs)
        elif method == 'moderate':
            if embed_data_active is None: raise ValueError("Embeddings required.")
            selected_indices_relative = moderate_selection_wrapper(embed_data_active, data_labels_active, K_adjusted)
        elif method == 'score_monotonic':
            if score_array_active is None: raise ValueError("Scores required.")
            selected_indices_relative = score_monotonic_wrapper(score_array_active, data_labels_active, K_adjusted, descending, class_balanced)
        # Note: 'mislabel_mask' is not a method here
        else:
            raise ValueError(f"Unsupported coreset selection method: {method}")

        # --- Map relative indices back to original indices ---
        # selected_indices_relative contains indices relative to the 'active_indices' array
        if selected_indices_relative: # Check if list is not empty
            original_indices = active_indices[selected_indices_relative]
            selected_samples[K] = original_indices.tolist() # Store as list
        else:
            selected_samples[K] = [] # Store empty list if nothing selected

    return selected_samples


# --- Example Usage (Reverted Mislabel Mask Handling) ---
if __name__ == '__main__':
    num_samples = 1000; embed_dim = 64; num_classes = 5
    dummy_data = {
        'embeddings': [np.random.rand(embed_dim) for _ in range(num_samples)],
        'label': [random.randint(0, num_classes-1) for _ in range(num_samples)],
        'dummy_score': np.random.rand(num_samples)*10,
        'hardness_score': np.concatenate([np.linspace(0,1,num_samples//2), np.linspace(1,0,num_samples-num_samples//2)])
    }
    df = pd.DataFrame(dummy_data)
    Ks_to_select = [50, 100, 200]

    # --- Example 1: Random selection after pruning hardest 100 ---
    print("\n--- Testing Random Selection with Mislabel Mask Pre-Filter ---")
    masked_random_results = coreset_sample(
        df, Ks_to_select,
        method='random', # Use random selection...
        label_col='label',
        use_mislabel_mask=True, # ...after applying the mask
        mislabel_score_col='hardness_score',
        mislabel_prune_num=100,
        mislabel_descending=True
    )
    print("Random results after mask:")
    for k, indices in masked_random_results.items():
        print(f" K={k}: {len(indices)} indices selected from original dataset.")

    # --- Example 2: Score monotonic on remaining samples ---
    print("\n--- Testing Score Monotonic (Desc) with Mislabel Mask Pre-Filter ---")
    masked_score_results = coreset_sample(
        df, Ks_to_select,
        method='score_monotonic', # Use score monotonic...
        label_col='label', score_col='dummy_score', descending=True, class_balanced=False,
        use_mislabel_mask=True, # ...after applying the mask
        mislabel_score_col='hardness_score',
        mislabel_prune_num=100,
        mislabel_descending=True
    )
    print("Score monotonic results after mask:")
    for k, indices in masked_score_results.items():
        print(f" K={k}: {len(indices)} indices selected from original dataset.")

    # --- Example 3: Standard moderate selection (no mask) ---
    print("\n--- Testing Moderate Selection (No Mask) ---")
    moderate_results = coreset_sample(df, Ks_to_select, method='moderate',
                                      embed_col='embeddings', label_col='label')
    print("Moderate selection results:")
    for k, indices in moderate_results.items(): print(f" K={k}: {len(indices)} indices")



