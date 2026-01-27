import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import orjson
import random
import os
import rl_env
import time
import math


is_auto_upload = False
fs = None

#import util_fs
#is_auto_upload = os.environ.get('AUTO_UPLOAD','True').lower() in ('true', '1', 't', 'yes')
#fs = util_fs.connect_cf()


num2time = lambda t=None: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t if t else time.time()))
path_this = os.path.dirname(os.path.abspath(__file__))
name_this = 'grpo'
VERSION = 'v3'

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

class ReferencePolicy_HEFT(PolicyNetwork):
    def __init__(self, E:int,D:int,F:int, hidden_dim: int = 128):
        super().__init__()
        self.i = F-1
        self.j = F-2
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        not_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            not_batch = True
        B, E, D, F = x.shape
        mask_flat = mask.view(B,E*D)
        upward_ranks = x[:,:,0,0]
        EFTs = x[:,:,:,self.i]+x[:,:,:,self.j]
        logits = ((EFTs.max()-EFTs)*upward_ranks.unsqueeze(-1)).view(B,E*D)
        logits[~mask_flat] = float('-inf')
        if not_batch: # (1, E*D) -> (E*D)
            logits = logits.squeeze(0)
        return torch.distributions.Categorical(logits=logits)


class PolicyNetwork_v1(PolicyNetwork):
    """
    MLP
    Input: (E, D, F) or (B, E, D, F)
        B: batch size
        E: number of ready-tasks
        D: number of computing-units
        F: number of features per (task,unit) pair
    Output: (E*D) or (B, E*D)
    """
    def __init__(self, E:int,D:int,F:int, hidden_dim: int = 128):
        super().__init__()
        self.feature_scorer = nn.Sequential(
            nn.Linear(F, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        not_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            not_batch = True
        B, E, D, F = x.shape
        mask_flat = mask.view(B,E*D)
        x_flat = x.view(B,E*D,F)
        logits = self.feature_scorer(x_flat).squeeze(-1) # shape: (B,E*D)
        logits[~mask_flat] = float('-inf')
        if not_batch: # (1, E*D) -> (E*D)
            logits = logits.squeeze(0)
        return torch.distributions.Categorical(logits=logits)

class PolicyNetwork_v2(PolicyNetwork):
    """
    self-attention
    Input: (E, D, F) or (B, E, D, F)
        B: batch size
        E: number of ready-tasks
        D: number of computing-units
        F: number of features per (task,unit) pair
    Output: (E*D) or (B, E*D)
    """
    def __init__(self, E:int,D:int,F:int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_extractor = nn.Sequential(
            nn.Linear(F, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (E,D,F) or (B,E,D,F)
        # mask: (E,D) or (B, E, D)
        not_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            not_batch = True
        B, E, D, F = x.shape
        mask_flat = mask.view(B,E*D)
        # (B, E, D, F) -> (B, E*D, F)
        x_flat = x.view(B,E*D,F)
        # (B, E*D, F) -> (B, E*D, H)
        h_flat = self.feature_extractor(x_flat)
        # (B, E*D, H) -> (B, E*D, H)
        att_out, _ = self.attention(h_flat, h_flat, h_flat,key_padding_mask=~mask_flat)
        # (B, E*D, H) -> (B, E*D)
        logits = self.scorer(att_out).squeeze(-1)
        logits[~mask_flat] = float('-inf')
        if not_batch: # (1, E*D) -> (E*D)
            logits = logits.squeeze(0)
        return torch.distributions.Categorical(logits=logits)


# ref: "Attention is All You Need"
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)



class PolicyNetwork_v3(PolicyNetwork):
    """
    Transformer
    Input: (E, D, F) or (B, E, D, F)
        B: batch size
        E: number of ready-tasks
        D: number of computing-units
        F: number of features per (task,unit) pair
    Output: (E*D) or (B, E*D)
    """
    def __init__(self, E: int, D: int, F: int, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.E = E
        self.D = D
        self.hidden_dim = hidden_dim
        self.seq_len = E * D

        self.feature_extractor = nn.Sequential(
            nn.Linear(F, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (E,D,F) or (B,E,D,F)
        # mask: (E,D) or (B, E, D)
        not_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            not_batch = True
        
        B, E, D, F = x.shape
        mask_flat = mask.view(B,E*D)
        x_flat = x.view(B, E * D, F)
        h_flat = self.feature_extractor(x_flat)  # (B, E*D, H)
        h_flat_with_pos = self.pos_encoder(h_flat) # (B, E*D, H)
        att_out = self.transformer_encoder(
            h_flat_with_pos, 
            src_key_padding_mask=~mask_flat
        ) # (B, E*D, H)
        logits = self.scorer(att_out).squeeze(-1) # (B, E*D)
        logits[~mask_flat] = float('-inf')
        if not_batch:
            logits = logits.squeeze(0)
        return torch.distributions.Categorical(logits=logits)
    

def save_checkpoint(model_key, save_name, actor, optimizer, iteration, best_reward):
    checkpoint = {
        "iteration": iteration,
        "model": actor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_reward": best_reward,
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    key = f"{VERSION}/{model_key}/{save_name}.pth"
    path_save = f"{path_this}/results/{model_key}/{save_name}.pth"
    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    torch.save(checkpoint, path_save)
    if fs and is_auto_upload:
        fs.upload(key,path_save)

def load_checkpoint(model_key, save_name, actor, optimizer=None):
    key = f"{VERSION}/{model_key}/{save_name}.pth"
    path_save = f"{path_this}/results/{model_key}/{save_name}.pth"
    if fs and not os.path.exists(path_save):
        fs.download(key,path_save)
    try:
        with torch.serialization.safe_globals([
            np._core.multiarray._reconstruct,
            np._core.multiarray.scalar,
            np.dtype,
            np.dtypes.Float32DType,
            np.dtypes.UInt32DType,
            np.ndarray,
        ]):
            ckpt = torch.load(path_save, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint from {path_save}")
        print(e)
        return 0, -1e9
    actor.load_state_dict(ckpt["model"])
    if not optimizer is None:
        optimizer.load_state_dict(ckpt["optimizer"])
    torch.random.set_rng_state(ckpt["torch_rng"])
    if torch.cuda.is_available() and ckpt["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    random.setstate(ckpt["python_rng"])
    return ckpt["iteration"], ckpt["best_reward"]


def save_training_log(model_key, save_name, data):
    key = f"{VERSION}/{model_key}/{save_name}.json"
    path_save = f"{path_this}/results/{model_key}/{save_name}.json"
    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    with open(path_save, "wb") as f:
        f.write(orjson.dumps(data))
    if fs and is_auto_upload:
        fs.upload(key,path_save)
def load_training_log(model_key, save_name, default):
    key = f"{VERSION}/{model_key}/{save_name}.json"
    path_save = f"{path_this}/results/{model_key}/{save_name}.json"
    if fs and not os.path.exists(path_save):
        fs.download(key,path_save)
    try:
        with open(path_save, "rb") as f:
            return orjson.loads(f.read())
    except:
        return default

def get_best_model(model_key):
    if fs:
        files = fs.ls(f"{VERSION}/{model_key}/")
    else:
        files = os.listdir(f"{path_this}/results/{model_key}")
    pth_files = [filename for filename in files if filename.endswith(".pth")]
    if pth_files:
        import re
        pth_files.sort(key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)])
        return pth_files[-1][:-4]
    return 'auto_save'
    

#-------------------------------------------------------------#
# ref: https://github.com/emparu/PPO-vs-GRPO
#-------------------------------------------------------------#
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def update_grpo(
    actor: PolicyNetwork,
    actor_ref: PolicyNetwork, # Reference policy model (frozen)
    actor_optimizer: optim.Optimizer,
    group_states_list: List[torch.Tensor],
    group_masks_list: List[torch.Tensor],
    group_actions_list: List[torch.Tensor],
    group_log_probs_old_list: List[torch.Tensor],
    group_advantages_list: List[torch.Tensor], # These are the NORMALIZED advantages
    grpo_epochs: int,
    grpo_clip_epsilon: float,
    grpo_kl_beta: float,       # Coefficient for KL divergence penalty
    entropy_coeff: float,      # Coefficient for entropy bonus (optional but often kept)
    grpo_version:str
) -> Tuple[float, float, float]: # Return avg policy objective, avg KL div, avg entropy
    """
    Performs the GRPO update for multiple epochs over the collected group data.

    Parameters:
    - actor (PolicyNetwork): The policy network being trained.
    - actor_ref (PolicyNetwork): The frozen reference policy network for KL calculation.
    - actor_optimizer (optim.Optimizer): The optimizer for the actor network.
    - group_states_list (List[torch.Tensor]): List of state tensors for each rollout.
    - group_masks_list (List[torch.Tensor]): List of mask tensors for each rollout.
    - group_actions_list (List[torch.Tensor]): List of action tensors for each rollout.
    - group_log_probs_old_list (List[torch.Tensor]): List of log prob tensors (from actor_ref) for each rollout.
    - group_advantages_list (List[torch.Tensor]): List of NORMALIZED advantage tensors for each rollout.
    - grpo_epochs (int): Number of optimization epochs.
    - grpo_clip_epsilon (float): Clipping parameter epsilon for the surrogate objective.
    - grpo_kl_beta (float): Coefficient for the KL divergence penalty term.
    - entropy_coeff (float): Coefficient for the entropy bonus.

    Returns:
    - Tuple[float, float, float]: Average surrogate objective value (before KL/entropy),
                                   average KL divergence estimate, average entropy over the epochs.
    """
    total_policy_objective = 0.0 # Tracks the clipped surrogate objective part
    total_kl_div = 0.0
    total_entropy = 0.0

    # --- 1. Concatenate Group Data into Batches ---
    # Ensure lists are not empty before concatenation
    if not group_states_list or not group_actions_list or not group_log_probs_old_list or not group_advantages_list:
         print("Warning: Empty lists passed to update_grpo. Skipping update.")
         return 0.0, 0.0, 0.0 # Or handle as appropriate

    try:
        states = torch.cat(group_states_list, dim=0).to(device)
        masks = torch.cat(group_masks_list, dim=0).to(device)
        actions = torch.cat(group_actions_list, dim=0).to(device)
        log_probs_old = torch.cat(group_log_probs_old_list, dim=0).to(device)
        advantages = torch.cat(group_advantages_list, dim=0).to(device) # \hat{A}_{i,t}
    except RuntimeError as e:
        print(f"Error during concatenation in update_grpo: {e}")
        # Print shapes for debugging
        print("Shapes:")
        for i in range(len(group_states_list)):
            print(f"  Rollout {i}: States={group_states_list[i].shape}, Actions={group_actions_list[i].shape}, LogProbs={group_log_probs_old_list[i].shape}, Advs={group_advantages_list[i].shape}")
        raise e # Re-raise the error after printing info


    # Detach advantages and old log probs - they are treated as constants during the update
    advantages = advantages.detach()
    log_probs_old = log_probs_old.detach()

    # Ensure reference actor is in eval mode and doesn't track gradients
    actor_ref.eval()

    for epoch in range(grpo_epochs):
        # --- Actor (Policy) Update ---
        # Evaluate current policy
        policy_dist = actor(states,masks)
        log_probs_new = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy().mean() # Entropy for exploration bonus

        # Calculate ratio r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
        # prob = pi_theta = exp(log_prob) exp(A)/exp(B)=exp(A-B)
        # log_probs_old comes from the policy *at the time of sampling*, which acts as pi_theta_old
        ratio = torch.exp(log_probs_new - log_probs_old)

        # Calculate surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - grpo_clip_epsilon, 1.0 + grpo_clip_epsilon) * advantages

        # Clipped Surrogate Objective part of the loss
        if grpo_version=='GRPO':
            clipped_surrogate_objective = torch.min(surr1, surr2).mean()
        elif grpo_version=='DrGRPO':
            clipped_surrogate_objective = torch.min(surr1, surr2).sum()/len(group_advantages_list)
        # --- Calculate KL Divergence Penalty ---
        kl_div_estimate_mean = 0.0
        with torch.no_grad(): # Ensure no gradients computed for reference policy
            policy_dist_ref = actor_ref(states,masks)
            log_probs_ref = policy_dist_ref.log_prob(actions)

        # Calculate KL divergence using the unbiased estimator
        # D_KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
        #      = exp(log_probs_ref - log_probs_new) - (log_probs_ref - log_probs_new) - 1
        # Detach log_probs_new here to prevent grads flowing through KL term backprop path incorrectly?
        # The gradient should only come from the direct dependence of the main objective on pi_theta.
        # Let's compute KL term separately and add its gradient scaled by beta.
        # Re-evaluate log_probs_new *without* detaching for the main objective gradient.
        log_ratio_ref_curr = log_probs_ref - log_probs_new.detach() # Use detached version for KL calculation
        kl_div_estimate = torch.exp(log_ratio_ref_curr) - log_ratio_ref_curr - 1
        kl_div_estimate_mean = kl_div_estimate.mean()

        # Ensure KL estimate is non-negative (it should be theoretically)
        kl_div_estimate_mean = torch.relu(kl_div_estimate_mean)


        # --- Combine Loss Terms ---
        # We want to MAXIMIZE the surrogate objective and MINIMIZE KL and entropy penalty
        # Optimizer minimizes loss, so: Loss = -Surrogate + KL_Penalty + Entropy_Penalty
        # Loss = -clipped_surrogate_objective + grpo_kl_beta * kl_div_estimate_mean - entropy_coeff * entropy
        # Note: The paper's formula (Eq 3) puts beta outside the KL term. Let's follow that.
        # Maximize: min(...) - beta * D_KL(...) => Minimize: -min(...) + beta * D_KL(...)
        # Adding entropy bonus: Minimize: -min(...) + beta * D_KL(...) - entropy_coeff * entropy
        
        policy_loss = -clipped_surrogate_objective + grpo_kl_beta * kl_div_estimate_mean - entropy_coeff * entropy


        # Optimize the actor
        actor_optimizer.zero_grad()
        policy_loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        actor_optimizer.step()

        # Accumulate metrics for logging (use .item() to detach)
        total_policy_objective += clipped_surrogate_objective.item() # Store the positive objective value
        total_kl_div += kl_div_estimate_mean.item()
        total_entropy += entropy.item()

    # Calculate average metrics over the epochs
    avg_policy_objective = total_policy_objective / grpo_epochs
    avg_kl_div = total_kl_div / grpo_epochs
    avg_entropy = total_entropy / grpo_epochs

    return avg_policy_objective, avg_kl_div, avg_entropy

def get_model_key(*args,**kwargs):
    defaults = {
        "semi_save_key": "G0H0C0W0I5N10F3",
        "rl_env_key": "O1S1R1T1A1E1M3",
        "policy_version": "v1",
        "ref_version": "self",
        "grpo_version": "GRPO", # GRPO or DrGRPO
        "GAMMA_GRPO": 0.997,
        "GRPO_KL_BETA": 0.01,
        "ACTOR_LR_GRPO": 3e-4,
        "GRPO_EPOCHS": 10,
        "GRPO_CLIP_EPSILON": 0.2,
        "ENTROPY_COEFF_GRPO": 0.01,
        "NUM_ITERATIONS_GRPO": 300,
        "GROUP_SIZE": 16,
        "MAX_STEPS_PER_EPISODE_GRPO": 10000,
        "EPSILON_STD": 1e-8,
        "INTERV_PRINT": 10,
        "INTERV_AUTOSAVE": 10
    }
    params = {
        **{k:x for k,x in zip(defaults.keys(),args)},
        **kwargs
    }
    arr = [name_this]
    for k,v in defaults.items():
        if k in params and params[k]!=v:
            arr.append(f"{k}={params[k]}")
    return ' '.join(arr)


def train(
        # --- Scheduler Environment ---
        semi_save_key = "G0H0C0W0I50N10F3", 
        rl_env_key="O1S1R1T1A1E1M3",
        policy_version = "v1",
        ref_version = "self",
        grpo_version = "GRPO", # GRPO or DrGRPO
        # --- GRPO Hyperparameters ---
        GAMMA_GRPO = 0.997,              # Discount factor
        GRPO_KL_BETA = 0.01,            # KL penalty factor
        ACTOR_LR_GRPO = 3e-4,           # Learning rate for the actor
        GRPO_EPOCHS = 10,               # Number of optimization epochs per iteration
        GRPO_CLIP_EPSILON = 0.2,        # GRPO clipping epsilon (used inside update_grpo)
        ENTROPY_COEFF_GRPO = 0.01,      # Coefficient for entropy bonus (used inside update_grpo)
        NUM_ITERATIONS_GRPO = 300,      # Number of GRPO iterations (policy updates)
        GROUP_SIZE = 16,                # Number of rollouts (episodes) per group (G)
        MAX_STEPS_PER_EPISODE_GRPO = 10000, # Max steps per episode
        EPSILON_STD = 1e-8,             # Small value to prevent division by zero
        INTERV_PRINT = 10,              # Print losses every INTERV_PRINT steps
        INTERV_AUTOSAVE = 10,          # Save model every INTERV_AUTOSAVE steps
    ):
        model_key = get_model_key(**locals())
        # --- 获取被覆盖的参数 ---
        #__cfg_frame = inspect.currentframe()
        #__cfg_func_name = __cfg_frame.f_code.co_name              # train
        #__cfg_func_obj = __cfg_frame.f_globals[__cfg_func_name]  # 函数对象
        #__cfg_sig = inspect.signature(__cfg_func_obj)
        #__cfg_defaults = {k: v.default for k, v in __cfg_sig.parameters.items()}
        #__cfg_args_actual = {k: v for k, v in locals().items() if k in __cfg_defaults}
        #__cfg_overridden = {k: v for k, v in __cfg_args_actual.items() if v != __cfg_defaults[k]}
        #model_key = ' '.join([name_this]+[f"{k}={v}" for k,v in __cfg_overridden.items()])
        # --- Initialization ---
        env_grpo = rl_env.SchedulerEnv(semi_save_key, rl_env_key)
        E = env_grpo.obs_max_E
        D = env_grpo.n_devices
        F = env_grpo.n_features
        actor_grpo: PolicyNetwork = globals()[f"PolicyNetwork_{policy_version}"](E,D,F).to(device)

        # Initialize Optimizer for Actor
        actor_optimizer_grpo: optim.Adam = optim.Adam(actor_grpo.parameters(), lr=ACTOR_LR_GRPO)
        
        last_iteration, best_reward = load_checkpoint(model_key, "auto_save", actor_grpo, actor_optimizer_grpo)
        last_log = load_training_log(model_key, "auto_save", {})
        # Lists for plotting/logging
        grpo_iteration_rewards = last_log.get("grpo_iteration_rewards", [])
        grpo_iteration_avg_ep_lens = last_log.get("grpo_iteration_avg_ep_lens", [])
        grpo_iteration_policy_losses = last_log.get("grpo_iteration_policy_losses", [])
        grpo_iteration_entropies = last_log.get("grpo_iteration_entropies", [])
        grpo_iteration_kl_divs = last_log.get("grpo_iteration_kl_divs", []) 

        print(f"Starting GRPO Training on rl_env {model_key}...")
        if ref_version=='self':
            actor_ref = globals()[f"PolicyNetwork_{policy_version}"](E,D,F).to(device)
        else:
            actor_ref = globals()[f"ReferencePolicy_{ref_version}"](E,D,F).to(device)
        start_time = time.time()
        # --- GRPO Training Loop ---
        for iteration in range(last_iteration,NUM_ITERATIONS_GRPO):
            # --- 1. Collect Group of Trajectories (Rollout Phase) ---
            group_states_list: List[torch.Tensor] = []
            group_masks_list: List[torch.Tensor] = []
            group_actions_list: List[torch.Tensor] = []
            group_log_probs_old_list: List[torch.Tensor] = [] # Log probs at time of sampling
            group_rewards_list: List[List[float]] = []      # Store raw rewards per rollout

            episode_rewards_in_iter = []
            episode_lengths_in_iter = []

            actor_grpo.eval() # Set actor to evaluation mode for rollout
            for rollout_idx in range(GROUP_SIZE):
                rollout_states: List[torch.Tensor] = []
                rollout_masks: List[torch.Tensor] = []
                rollout_actions: List[torch.Tensor] = []
                rollout_log_probs: List[torch.Tensor] = []
                rollout_rewards: List[float] = []

                
                state,mask = env_grpo.reset(iteration*GROUP_SIZE+rollout_idx+10000)


                episode_reward = 0.0
                episode_steps = 0
                done = False

                for t in range(MAX_STEPS_PER_EPISODE_GRPO):
                    # Sample action and get log_prob
                    with torch.no_grad():
                        state_tensor = state.to(device)
                        mask_tensor = mask.to(device)
                        policy_dist = actor_grpo(state_tensor,mask_tensor)
                        action_tensor = policy_dist.sample()
                        log_prob = policy_dist.log_prob(action_tensor)


                    # Interact with environment
                    action_item = action_tensor.item()
                    next_state,next_mask, reward, done = env_grpo.step(action_item)

                    # Store data for this step in the current rollout
                    rollout_states.append(state_tensor)
                    rollout_masks.append(mask_tensor)
                    rollout_actions.append(action_tensor)
                    rollout_log_probs.append(log_prob)
                    rollout_rewards.append(reward)

                    state = next_state
                    mask = next_mask
                    episode_reward += reward
                    episode_steps += 1

                    if done:
                        break
                    
                # Store the completed rollout data as tensors
                if rollout_states:
                     group_states_list.append(torch.stack(rollout_states))
                     group_masks_list.append(torch.stack(rollout_masks))
                     action_dim = rollout_actions[0].dim() if rollout_actions else 0
                     log_prob_dim = rollout_log_probs[0].dim() if rollout_log_probs else 0
                     # Squeeze added dimensions if they exist from stacking scalar tensors
                     group_actions_list.append(torch.stack(rollout_actions).squeeze() if action_dim > 0 else torch.tensor([a.item() for a in rollout_actions], device=device))
                     group_log_probs_old_list.append(torch.stack(rollout_log_probs).squeeze() if log_prob_dim > 0 else torch.tensor([lp.item() for lp in rollout_log_probs], device=device))
                     group_rewards_list.append(rollout_rewards)
                else:
                     # Keep warning for empty rollouts as it might indicate issues
                     print(f"Warning: Rollout {rollout_idx+1} in iteration {iteration+1} was empty.") # Keep this print commented unless debugging
                     group_states_list.append(torch.empty((0, D, F), device=device))
                     group_masks_list.append(torch.empty((0, D), device=device))
                     group_actions_list.append(torch.empty((0,), dtype=torch.long, device=device))
                     group_log_probs_old_list.append(torch.empty((0,), device=device))
                     group_rewards_list.append([])

                episode_rewards_in_iter.append(episode_reward)
                episode_lengths_in_iter.append(episode_steps)

            actor_grpo.train() # Set actor back to training mode for update

            # --- 2. Calculate Group Relative Advantages (Discounted Returns Method) ---
            # 这里用的是Process Supervision，根据最终奖励计算单步奖励。
            group_advantages_list: List[torch.Tensor] = [] # Will store FINAL normalized advantages
            all_raw_advantages_in_group: List[float] = [] # Temp storage for mean/std calc (advantages are now discounted returns)
            temp_raw_advantages_tensors: List[torch.Tensor] = [] # Store raw tensors before normalization

            # --- First Pass: Calculate RAW discounted returns-to-go and collect them ---
            for i, rollout_rewards in enumerate(group_rewards_list):
                rollout_len = len(rollout_rewards)
                # Renamed to reflect it's discounted returns, but keeping variable name for consistency
                rollout_raw_advantages = torch.zeros(rollout_len, dtype=torch.float32, device=device) # [r_i^{index(t)}]

                if rollout_len > 0:
                    # Calculate raw discounted returns (G_t = r_t + gamma*G_{t+1})
                    discounted_return = 0.0
                    for t in reversed(range(rollout_len)):
                        discounted_return = rollout_rewards[t] + GAMMA_GRPO * discounted_return # Apply discount factor
                        rollout_raw_advantages[t] = discounted_return # r_i^{index(t)}

                    # Store raw advantages (discounted returns) for later normalization
                    temp_raw_advantages_tensors.append(rollout_raw_advantages) # r_i^{index(t)} (g,t)
                    all_raw_advantages_in_group.extend(rollout_raw_advantages.cpu().numpy()) # Collect as list of floats
                else:
                    temp_raw_advantages_tensors.append(torch.empty((0,), device=device)) # Placeholder for empty

            # --- Calculate Mean/Std of ALL RAW discounted returns ---
            if len(all_raw_advantages_in_group) > 1:
                group_mean_advantage = np.mean(all_raw_advantages_in_group) # mean(\mathrm{R})
                group_std_advantage = np.std(all_raw_advantages_in_group)   # std(\mathrm{R})
            elif len(all_raw_advantages_in_group) == 1:
                 group_mean_advantage = all_raw_advantages_in_group[0]
                 group_std_advantage = 0.0
            else:
                group_mean_advantage = 0.0
                group_std_advantage = 0.0
                # Keep this warning as it's important if no advantages are calculated
                if iteration == 0: # Only print once if it persists
                     print("Warning: No advantages calculated in group (all rollouts empty?).")


            # --- Second Pass: Normalize raw discounted returns ---
            if grpo_version=='GRPO':
                for i, raw_advantages_tensor in enumerate(temp_raw_advantages_tensors):
                    if raw_advantages_tensor.nelement() > 0:
                        # Normalize using the group's mean/std of discounted returns
                        normalized_advantages = (raw_advantages_tensor - group_mean_advantage) / (group_std_advantage + EPSILON_STD) # \hat{A}_{i,t}=\tilde{r}^{index(t)}
                    else:
                        normalized_advantages = raw_advantages_tensor

                    group_advantages_list.append(normalized_advantages) # [\hat{A}_{i,t}] (g,t)
            elif grpo_version=='DrGRPO':
                for i, raw_advantages_tensor in enumerate(temp_raw_advantages_tensors):
                    if raw_advantages_tensor.nelement() > 0:
                        # Normalize using the group's mean/std of discounted returns
                        normalized_advantages = (raw_advantages_tensor - group_mean_advantage)
                    else:
                        normalized_advantages = raw_advantages_tensor
                    group_advantages_list.append(normalized_advantages) # [\hat{A}_{i,t}] (g,t)
            else:
                print(f"unknown grpo_verion: {grpo_version}")
                return
            # --- Create Reference Actor (Copy current actor state) ---
            if ref_version=='self':
                actor_ref.load_state_dict(actor_grpo.state_dict())
            actor_ref.eval() # Set to evaluation mode

            # --- 3. Perform GRPO Update ---
            avg_policy_obj, avg_kl, avg_entropy = update_grpo(
                actor=actor_grpo,                 # The actor being trained
                actor_ref=actor_ref,              # The frozen reference actor
                actor_optimizer=actor_optimizer_grpo,
                group_states_list=group_states_list,
                group_masks_list=group_masks_list,
                group_actions_list=group_actions_list,
                group_log_probs_old_list=group_log_probs_old_list, # Log probs from sampling time
                group_advantages_list=group_advantages_list,      # Final NORMALIZED advantages
                grpo_epochs=GRPO_EPOCHS,
                grpo_clip_epsilon=GRPO_CLIP_EPSILON,
                grpo_kl_beta=GRPO_KL_BETA,
                entropy_coeff=ENTROPY_COEFF_GRPO,
                grpo_version = grpo_version
            )
            
            # Store losses for logging/plotting
            # Note: avg_policy_obj is the surrogate value, not the final combined loss
            grpo_iteration_policy_losses.append(avg_policy_obj) # Or store the combined loss if preferred
            grpo_iteration_entropies.append(avg_entropy)
            grpo_iteration_kl_divs.append(avg_kl)
            # --- Logging ---
            avg_reward_iter = np.mean(episode_rewards_in_iter) if episode_rewards_in_iter else np.nan
            avg_len_iter = np.mean(episode_lengths_in_iter) if episode_lengths_in_iter else np.nan

            grpo_iteration_rewards.append(float(avg_reward_iter))
            grpo_iteration_avg_ep_lens.append(float(avg_len_iter))

            # Print summary log every N iterations (e.g., 10)
            iteration = iteration + 1
            if iteration % INTERV_PRINT == 0 or iteration == NUM_ITERATIONS_GRPO:
                now = time.time()
                remaining = (now-start_time)/(iteration-last_iteration) * (NUM_ITERATIONS_GRPO-iteration)
                progress_info = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))}(+{int(remaining//3600)}h{int(remaining%3600//60):02d}m)'
                print(f"{progress_info} | Iter {iteration}/{NUM_ITERATIONS_GRPO} | Avg Reward (Group): {avg_reward_iter:.2f} | P_loss: {avg_policy_obj:.4f} | Entropy: {avg_entropy:.4f}")
                if avg_reward_iter > best_reward:
                    best_reward = avg_reward_iter
                    save_checkpoint(model_key, f"best_model_{iteration}",actor_grpo, actor_optimizer_grpo,iteration,best_reward)
                    save_training_log(model_key,  f"best_model_{iteration}", {
                        'grpo_iteration_rewards':grpo_iteration_rewards, 
                        'grpo_iteration_avg_ep_lens':grpo_iteration_avg_ep_lens, 
                        'grpo_iteration_policy_losses':grpo_iteration_policy_losses, 
                        'grpo_iteration_entropies':grpo_iteration_entropies, 
                        'grpo_iteration_kl_divs':grpo_iteration_kl_divs
                    })
            if iteration % INTERV_AUTOSAVE == 0 or iteration == NUM_ITERATIONS_GRPO:
                save_checkpoint(model_key, "auto_save",actor_grpo, actor_optimizer_grpo,iteration,best_reward)
                save_training_log(model_key, "auto_save", {
                    'grpo_iteration_rewards':grpo_iteration_rewards, 
                    'grpo_iteration_avg_ep_lens':grpo_iteration_avg_ep_lens, 
                    'grpo_iteration_policy_losses':grpo_iteration_policy_losses, 
                    'grpo_iteration_entropies':grpo_iteration_entropies, 
                    'grpo_iteration_kl_divs':grpo_iteration_kl_divs
                })
        print(f"\nGRPO Training Loop Finished ({model_key}).")


def post_test_show(senv,model_key,save_name):
    results = senv.score()
    print(senv.stringify_score(results))
    return results


def test(
        # --- Scheduler Environment ---
        semi_save_key = "G0H0C0W0I5N10F3", 
        rl_env_key="O1S1R1T1A1E1M3",
        policy_version = "v1",
        ref_version = "self",
        grpo_version = "GRPO", # GRPO or DrGRPO
        # --- GRPO Hyperparameters ---
        GAMMA_GRPO = 0.997,              # Discount factor
        GRPO_KL_BETA = 0.01,            # KL penalty factor
        ACTOR_LR_GRPO = 3e-4,           # Learning rate for the actor
        GRPO_EPOCHS = 10,               # Number of optimization epochs per iteration
        GRPO_CLIP_EPSILON = 0.2,        # GRPO clipping epsilon (used inside update_grpo)
        ENTROPY_COEFF_GRPO = 0.01,      # Coefficient for entropy bonus (used inside update_grpo)
        NUM_ITERATIONS_GRPO = 300,      # Number of GRPO iterations (policy updates)
        GROUP_SIZE = 16,                # Number of rollouts (episodes) per group (G)
        MAX_STEPS_PER_EPISODE_GRPO = 10000, # Max steps per episode
        EPSILON_STD = 1e-8,             # Small value to prevent division by zero
        # --- run parameters ---
        save_name = 'best_model',
        seed_from = 0,
        seed_to = 3,
        alter_semi_save_key = None,
        overwrite = False
    ):
        model_key = get_model_key(**locals())
        if save_name=='best_model':
            save_name = get_best_model(model_key)
        # --- Initialization ---
        alter_semi_save_key = alter_semi_save_key or semi_save_key
        key = f"{VERSION}/{model_key}/{save_name} {alter_semi_save_key} {seed_from} {seed_to}.json"
        path_save = f"{path_this}/results/{model_key}/{save_name} {alter_semi_save_key} {seed_from} {seed_to}.json"
        if not overwrite:
            if fs and not os.path.exists(path_save):
                fs.download(key, path_save)
            try:
                with open(path_save,'rb') as f:
                    data = orjson.loads(f.read())
                return data
            except Exception as e:
                ...
        senv = rl_env.SchedulerEnv(alter_semi_save_key, rl_env_key, skip_baseline=True)
        E = senv.obs_max_E
        D = senv.n_devices
        F = senv.n_features
        actor_grpo: PolicyNetwork = globals()[f"PolicyNetwork_{policy_version}"](E,D,F).to(device)
        actor_grpo.eval()
        last_iteration, best_reward = load_checkpoint(model_key, save_name, actor_grpo)
        mat_results = []
        clock_time = 0
        for seed in range(seed_from, seed_to):
            t0=time.time()
            state,mask = senv.reset(seed)
            done = False
            while not done:
                # Sample action and get log_prob
                with torch.no_grad():
                    state_tensor = state.to(device)
                    mask_tensor = mask.to(device)
                    policy_dist = actor_grpo(state_tensor,mask_tensor)
                    action_tensor = policy_dist.sample()

                # Interact with environment
                action_item = action_tensor.item()
                next_state,next_mask, reward, done = senv.step(action_item)

                state = next_state
                mask = next_mask
            t1=time.time()
            clock_time += t1-t0
            mat_results.append(senv.score())
        ret = {
            'model_key':model_key,
            'save_name':save_name,
            'alter_semi_save_key':alter_semi_save_key,
            'seed_from':seed_from,
            'seed_to':seed_to,
            'mat':mat_results,
            'time':clock_time
        }
        os.makedirs(os.path.dirname(path_save),exist_ok=True)
        with open(path_save,'wb') as f:
            f.write(orjson.dumps(ret))
        if fs and is_auto_upload:
            fs.upload(key,path_save)
        return ret


if __name__ == "__main__":
    import sys,ast
    if len(sys.argv) > 1:
        f_name = sys.argv[1]
        args = []
        kwargs = {}
        for x in sys.argv[2:]:
            if '=' in x:
                k,v = x.split('=',1)
                try: v = ast.literal_eval(v)
                except: pass
                kwargs[k] = v
            else:
                try: x = ast.literal_eval(x)
                except: pass
                args.append(x)
        f = globals().get(f_name)
        if f and callable(f):
            print(f"Invoke {f_name}{args}{kwargs}")
            f(*args,**kwargs)
        else:
            print(f"'{f_name}' not found, nothing happened.")
    else:
        print(f"Usage: python {os.path.split(__file__)[1]} <func> [<params>, ..]")

#python rl_grpo.py env_run G0H0C0W2I500N10F3 O0S1R1T1A1E1M3 v1 HEFT ACTOR_LR_GRPO=0.001 save_key=G0H0C0W1I500N10F3R0