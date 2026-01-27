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
name_this = 'ppo'
VERSION = 'v3'

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

class ValueNetwork(nn.Module):
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

# Define the Value Network (Critic)
class ValueNetwork_v1(ValueNetwork):
    """ MLP Critic network for PPO """
    def __init__(self, E:int, D:int, F:int, hidden_dim: int = 128):
        super().__init__()
        self.pair_feature_extractor = nn.Sequential(
            nn.Linear(F, hidden_dim),
            nn.ReLU(),
        )
        self.overall_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        pair_features_flat  = self.pair_feature_extractor(x_flat) # shape: (B,E*D,H)
        masked_pair_features_flat = pair_features_flat * mask_flat.unsqueeze(-1)# (B,E*D,H)
        global_feature = masked_pair_features_flat.sum(dim=1) # shape: (B,H)
        state_value = self.overall_scorer(global_feature) # shape: (B,1)
        if not_batch: # (1, 1) -> (1)
            state_value = state_value.squeeze(0)
        return state_value



def save_checkpoint(model_key, save_name, actor, critic, actor_optimizer, critic_optimizer, iteration, best_reward):
    checkpoint = {
        "iteration": iteration,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
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

def load_checkpoint(model_key, save_name, actor, critic=None, actor_optimizer=None, critic_optimizer=None):
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
    actor.load_state_dict(ckpt["actor"])
    if not critic is None:
        critic.load_state_dict(ckpt["critic"])
    if not actor_optimizer is None:
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    if not critic_optimizer is None:
        critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
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
    except Exception as e:
        print(f"Failed to load training log from {path_save}, {e}")
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

# Define the compute_gae function
def compute_gae(rewards: torch.Tensor, 
               values: torch.Tensor, 
               next_values: torch.Tensor, 
               dones: torch.Tensor, 
               gamma: float, 
               lambda_gae: float, 
               standardize: bool,
               epsilon_std: float) -> torch.Tensor:
    """
    Computes Generalized Advantage Estimation (GAE).
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        advantages[t] = delta + gamma * lambda_gae * last_advantage * mask
        last_advantage = advantages[t]

    if standardize:
        mean_adv = torch.mean(advantages)
        std_adv = torch.std(advantages) + epsilon_std
        advantages = (advantages - mean_adv) / std_adv
        
    return advantages


# Modified update_ppo function
def update_ppo_with_kl(actor: PolicyNetwork,
                       critic: ValueNetwork,
                       actor_ref: PolicyNetwork, # Add reference actor
                       actor_optimizer: optim.Optimizer,
                       critic_optimizer: optim.Optimizer,
                       states: torch.Tensor,
                       masks: torch.Tensor,
                       actions: torch.Tensor,
                       log_probs_old: torch.Tensor,
                       advantages: torch.Tensor,
                       returns_to_go: torch.Tensor,
                       ppo_epochs: int,
                       ppo_clip_epsilon: float,
                       ppo_kl_beta: float,      # Add KL beta coefficient
                       value_loss_coeff: float,
                       entropy_coeff: float) -> Tuple[float, float, float, float]: # Return avg policy obj, value loss, KL div, entropy
    """
    Performs the PPO update for multiple epochs over the collected batch,
    INCLUDING a KL divergence penalty term in the actor loss.

    Parameters:
    - actor, critic: The networks being trained.
    - actor_ref: The frozen reference policy network for KL calculation.
    - actor_optimizer, critic_optimizer: The optimizers.
    - states, masks, actions, log_probs_old, advantages, returns_to_go: Batch data tensors.
    - ppo_epochs (int): Number of optimization epochs.
    - ppo_clip_epsilon (float): Clipping parameter epsilon.
    - ppo_kl_beta (float): Coefficient for the KL divergence penalty.
    - value_loss_coeff (float): Coefficient for the value loss.
    - entropy_coeff (float): Coefficient for the entropy bonus.

    Returns:
    - Tuple[float, float, float, float]: Average policy objective (clipped surrogate),
                                         average value loss, average KL divergence,
                                         and average entropy over the epochs.
    """
    total_policy_objective = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl_div = 0.0

    # Detach advantages and old log probs - they are treated as constants during the update
    advantages = advantages.detach()
    log_probs_old = log_probs_old.detach()
    returns_to_go = returns_to_go.detach()

    # Ensure reference actor is in eval mode and doesn't track gradients
    actor_ref.eval()

    for _ in range(ppo_epochs):
        # --- Actor (Policy) Update ---
        # Evaluate current policy
        policy_dist = actor(states,masks)
        log_probs_new = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy().mean() # Entropy for exploration bonus

        # Calculate ratio r_t(theta)
        ratio = torch.exp(log_probs_new - log_probs_old)

        # Calculate surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages

        # Clipped Surrogate Objective part
        clipped_surrogate_objective = torch.min(surr1, surr2).mean()

        # --- Calculate KL Divergence Penalty ---
        kl_div_estimate_mean = 0.0
        with torch.no_grad(): # No gradients for reference policy
             policy_dist_ref = actor_ref(states,masks)
             log_probs_ref = policy_dist_ref.log_prob(actions)

        # Calculate KL divergence using the unbiased estimator: exp(log P_ref - log P) - (log P_ref - log P) - 1
        log_ratio_ref_curr = log_probs_ref - log_probs_new.detach() # Use detached log_probs_new for KL calc
        kl_div_estimate = torch.exp(log_ratio_ref_curr) - log_ratio_ref_curr - 1
        kl_div_estimate_mean = torch.relu(kl_div_estimate.mean()) # Ensure non-negative

        # --- Combine Actor Loss Terms ---
        # Loss = -ClippedSurrogate + KL_Penalty - EntropyBonus
        policy_loss = -clipped_surrogate_objective + ppo_kl_beta * kl_div_estimate_mean - entropy_coeff * entropy

        # Optimize the actor
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        # --- Critic (Value) Update ---
        # Predict values - ensure critic input is correctly handled
        values_pred = critic(states,masks).squeeze() # Squeeze potential extra dimension

        # Value loss (MSE)
        value_loss = F.mse_loss(values_pred, returns_to_go)

        # Optimize the critic
        critic_optimizer.zero_grad()
        # Scale value loss before backward pass
        (value_loss_coeff * value_loss).backward()
        critic_optimizer.step()

        # Accumulate metrics for logging
        total_policy_objective += clipped_surrogate_objective.item() # Log positive objective
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        total_kl_div += kl_div_estimate_mean.item()

    # Calculate average metrics over the epochs
    avg_policy_objective = total_policy_objective / ppo_epochs
    avg_value_loss = total_value_loss / ppo_epochs
    avg_entropy = total_entropy / ppo_epochs
    avg_kl_div = total_kl_div / ppo_epochs

    return avg_policy_objective, avg_value_loss, avg_kl_div, avg_entropy

def get_model_key(*args,**kwargs):
    defaults = {
        "semi_save_key": "G0H0C0W0I5N10F3",
        "rl_env_key": "O1S1R1T1A1E1M3",
        "policy_version": "v1",
        "ref_version": "self",
        "value_version": "v1",
        "GAMMA_PPO": 0.997,
        "GAE_LAMBDA_PPO": 0.997,
        "PPO_KL_BETA": 0.01,
        "ACTOR_LR": 3e-4,
        "CRITIC_LR_PPO": 0.001,
        "PPO_EPOCHS": 10,
        "PPO_CLIP_EPSILON": 0.2,
        "VALUE_LOSS_COEFF": 0.5,
        "ENTROPY_COEFF": 0.01,
        "STANDARDIZE_ADV_PPO": True,
        "NUM_ITERATIONS_PPO": 300,
        "GROUP_SIZE": 16,
        "MAX_STEPS_PER_EPISODE_PPO": 10000,
        "EPSILON_STD": 1e-8,
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
        ref_version = "HEFT", # HEFT, self
        value_version = "v1",
        # --- PPO Hyperparameters ---
        GAMMA_PPO = 0.997,             # Discount factor
        GAE_LAMBDA_PPO = 0.997,        # GAE lambda parameter
        PPO_KL_BETA = 0.01,           # KL penalty factor
        ACTOR_LR = 3e-4,              # Learning rate for the actor
        CRITIC_LR_PPO = 1e-3,         # Learning rate for the critic
        PPO_EPOCHS = 10,              # Number of optimization epochs per iteration
        PPO_CLIP_EPSILON = 0.2,       # PPO clipping epsilon
        VALUE_LOSS_COEFF = 0.5,       # Coefficient for value loss
        ENTROPY_COEFF = 0.01,         # Coefficient for entropy bonus
        STANDARDIZE_ADV_PPO = True,   # Whether to standardize advantages
        NUM_ITERATIONS_PPO = 300,     # Number of PPO iterations (policy updates)
        GROUP_SIZE = 16,                # Number of rollouts (episodes) per group (G)
        MAX_STEPS_PER_EPISODE_PPO = 10000, # Max steps per episode
        EPSILON_STD = 1e-8,             # Small value to prevent division by zero
        INTERV_PRINT = 10,              # Print losses every INTERV_PRINT steps
        INTERV_AUTOSAVE = 10,          # Save model every INTERV_AUTOSAVE steps
    ):
        model_key = get_model_key(**locals())
        # --- Initialization ---
        env_ppo = rl_env.SchedulerEnv(semi_save_key, rl_env_key)
        E = env_ppo.obs_max_E
        D = env_ppo.n_devices
        F = env_ppo.n_features
        # Initialize Actor and Critic
        actor_ppo: PolicyNetwork = globals()[f"PolicyNetwork_{policy_version}"](E,D,F).to(device)
        critic_ppo: ValueNetwork = globals()[f"ValueNetwork_{value_version}"](E,D,F).to(device)

        # Initialize Optimizers
        actor_optimizer_ppo: optim.Adam = optim.Adam(actor_ppo.parameters(), lr=ACTOR_LR)
        critic_optimizer_ppo: optim.Adam = optim.Adam(critic_ppo.parameters(), lr=CRITIC_LR_PPO)

        if ref_version=='self':
            actor_ref = globals()[f"PolicyNetwork_{policy_version}"](E,D,F).to(device)
        else:
            actor_ref = globals()[f"ReferencePolicy_{ref_version}"](E,D,F).to(device)
        
        # Load previous checkpoint
        last_iteration, best_reward = load_checkpoint(model_key, "auto_save", actor_ppo, critic_ppo, actor_optimizer_ppo, critic_optimizer_ppo)
        last_log = load_training_log(model_key, "auto_save", {})
        
        # Lists for plotting
        ppo_iteration_rewards       = last_log.get("ppo_iteration_rewards", [])
        ppo_iteration_avg_ep_lens   = last_log.get("ppo_iteration_avg_ep_lens", [])
        ppo_iteration_policy_losses = last_log.get("ppo_iteration_policy_losses", [])
        ppo_iteration_value_losses  = last_log.get("ppo_iteration_value_losses", [])
        ppo_iteration_entropies     = last_log.get("ppo_iteration_entropies", [])
        ppo_iteration_kl_divs       = last_log.get("ppo_iteration_kl_divs", [])

        print(f"Starting PPO Training on rl_env {model_key}...")

        start_time = time.time()
        # --- PPO Training Loop ---
        for iteration in range(last_iteration,NUM_ITERATIONS_PPO):
            # --- 1. Collect Trajectories (Rollout Phase) --- 
            # Store data in lists temporarily
            batch_states_list = []
            batch_masks_list = []
            batch_actions_list = []
            batch_log_probs_old_list = []
            batch_rewards_list = []
            batch_values_list = []
            batch_dones_list = []

            episode_rewards_in_iter = []
            episode_lengths_in_iter = []

            for rollout_idx in range(GROUP_SIZE):
                state,mask = env_ppo.reset(iteration*GROUP_SIZE+rollout_idx+10000)
                episode_reward = 0.0
                episode_steps = 0
                done = False

                for t in range(MAX_STEPS_PER_EPISODE_PPO):

                    # Sample action and get value estimate
                    with torch.no_grad():
                        state_tensor = state.to(device)
                        mask_tensor = mask.to(device)
                        policy_dist = actor_ppo(state_tensor,mask_tensor)
                        value = critic_ppo(state_tensor,mask_tensor).squeeze()

                        action_tensor = policy_dist.sample()
                        action = action_tensor.item()
                        log_prob = policy_dist.log_prob(action_tensor)

                    # Interact with environment
                    next_state,next_mask, reward, done = env_ppo.step(action)

                    # Store data
                    batch_states_list.append(state)
                    batch_masks_list.append(mask)
                    batch_actions_list.append(action)
                    batch_log_probs_old_list.append(log_prob)
                    batch_values_list.append(value)
                    batch_rewards_list.append(reward)
                    batch_dones_list.append(float(done))

                    state = next_state
                    mask = next_mask
                    episode_reward += reward
                    episode_steps += 1

                    if done or t==MAX_STEPS_PER_EPISODE_PPO-1:
                        episode_rewards_in_iter.append(episode_reward)
                        episode_lengths_in_iter.append(episode_steps)
                        break

            # --- End Rollout --- 

            # Calculate next_values for GAE
            # For non-terminal states, next_value is the value of the next state
            # For terminal states, next_value is 0
            next_values = []
            with torch.no_grad():
                for i in range(len(batch_states_list)):
                    if batch_dones_list[i] > 0.5:  # If done
                        next_values.append(torch.tensor(0.0).to(device))
                    elif i == len(batch_states_list) - 1:  # Last state in batch
                        next_state,next_mask, reward, done = env_ppo.step(batch_actions_list[i])[0]  # Get next state
                        next_values.append(critic_ppo(next_state.to(device),next_mask.to(device)).squeeze())
                    else:  # Not done, use value of next state in batch
                        next_values.append(batch_values_list[i+1])

            # Convert lists to tensors
            states_tensor = torch.stack(batch_states_list).to(device)
            masks_tensor = torch.stack(batch_masks_list).to(device)
            actions_tensor = torch.tensor(batch_actions_list, dtype=torch.long, device=device)
            log_probs_old_tensor = torch.stack(batch_log_probs_old_list).squeeze().to(device)
            rewards_tensor = torch.tensor(batch_rewards_list, dtype=torch.float32, device=device)
            values_tensor = torch.stack(batch_values_list).to(device)
            next_values_tensor = torch.stack(next_values).to(device)
            dones_tensor = torch.tensor(batch_dones_list, dtype=torch.float32, device=device)

            # --- 2. Estimate Advantages & Returns-to-go --- 
            advantages_tensor = compute_gae(
                rewards_tensor, values_tensor, next_values_tensor, dones_tensor, 
                GAMMA_PPO, GAE_LAMBDA_PPO, standardize=STANDARDIZE_ADV_PPO,epsilon_std=EPSILON_STD
            )
            returns_to_go_tensor = advantages_tensor + values_tensor

            # --- Create Reference Actor (Snapshot before update) ---
            if ref_version=='self':
                actor_ref.load_state_dict(actor_ppo.state_dict())
            actor_ref.eval() # Set to evaluation mode

            # --- 3. Perform PPO Update --- 
            avg_policy_loss, avg_value_loss, avg_kl, avg_entropy = update_ppo_with_kl(
                actor=actor_ppo,
                critic=critic_ppo,
                actor_ref=actor_ref,             # <<< Pass reference actor
                actor_optimizer=actor_optimizer_ppo,
                critic_optimizer=critic_optimizer_ppo,
                states=states_tensor,
                masks=masks_tensor,
                actions=actions_tensor,
                log_probs_old=log_probs_old_tensor.squeeze(), # Squeeze log probs
                advantages=advantages_tensor,
                returns_to_go=returns_to_go_tensor,
                ppo_epochs=PPO_EPOCHS,
                ppo_clip_epsilon=PPO_CLIP_EPSILON,
                ppo_kl_beta=PPO_KL_BETA,         # <<< Pass KL beta
                value_loss_coeff=VALUE_LOSS_COEFF,
                entropy_coeff=ENTROPY_COEFF
            )

            # --- Logging --- 
            avg_reward_iter = np.mean(episode_rewards_in_iter) if episode_rewards_in_iter else np.nan
            avg_len_iter = np.mean(episode_lengths_in_iter) if episode_lengths_in_iter else np.nan

            ppo_iteration_rewards.append(float(avg_reward_iter))
            ppo_iteration_avg_ep_lens.append(float(avg_len_iter))
            ppo_iteration_policy_losses.append(avg_policy_loss)
            ppo_iteration_value_losses.append(avg_value_loss)
            ppo_iteration_entropies.append(avg_entropy)
            ppo_iteration_kl_divs.append(avg_kl)

            # Print summary log every N iterations (e.g., 10)
            iteration = iteration + 1
            if iteration % INTERV_PRINT == 0 or iteration == NUM_ITERATIONS_PPO:
                now = time.time()
                remaining = (now-start_time)/(iteration-last_iteration) * (NUM_ITERATIONS_PPO-iteration)
                progress_info = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))}(+{int(remaining//3600)}h{int(remaining%3600//60):02d}m)'
                print(f"{progress_info} | Iter {iteration}/{NUM_ITERATIONS_PPO} | Avg Reward: {avg_reward_iter:.2f} | P_Loss: {avg_policy_loss:.4f} | V_Loss: {avg_value_loss:.4f} | Entropy: {avg_entropy:.4f}")
                if avg_reward_iter > best_reward:
                    best_reward = avg_reward_iter
                    save_checkpoint(model_key, f"best_model_{iteration}",actor_ppo,critic_ppo,actor_optimizer_ppo,critic_optimizer_ppo,iteration,best_reward)
                    save_training_log(model_key,  f"best_model_{iteration}", {
                        'ppo_iteration_rewards':ppo_iteration_rewards, 
                        'ppo_iteration_avg_ep_lens':ppo_iteration_avg_ep_lens, 
                        'ppo_iteration_policy_losses':ppo_iteration_policy_losses, 
                        'ppo_iteration_value_losses':ppo_iteration_value_losses, 
                        'ppo_iteration_entropies':ppo_iteration_entropies,
                        'ppo_iteration_kl_divs':ppo_iteration_kl_divs
                    })
            if iteration % INTERV_AUTOSAVE == 0 or iteration == NUM_ITERATIONS_PPO:
                save_checkpoint(model_key, "auto_save",actor_ppo,critic_ppo,actor_optimizer_ppo,critic_optimizer_ppo,iteration,best_reward)
                save_training_log(model_key, "auto_save", {
                        'ppo_iteration_rewards':ppo_iteration_rewards, 
                        'ppo_iteration_avg_ep_lens':ppo_iteration_avg_ep_lens, 
                        'ppo_iteration_policy_losses':ppo_iteration_policy_losses, 
                        'ppo_iteration_value_losses':ppo_iteration_value_losses, 
                        'ppo_iteration_entropies':ppo_iteration_entropies,
                        'ppo_iteration_kl_divs':ppo_iteration_kl_divs
                })
        print(f"\nPPO Training Loop Finished ({model_key}).")

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
        value_version = "v1",
        # --- PPO Hyperparameters ---
        GAMMA_PPO = 0.997,             # Discount factor
        GAE_LAMBDA_PPO = 0.997,        # GAE lambda parameter
        PPO_KL_BETA = 0.01,           # KL penalty factor
        ACTOR_LR = 3e-4,              # Learning rate for the actor
        CRITIC_LR_PPO = 1e-3,         # Learning rate for the critic
        PPO_EPOCHS = 10,              # Number of optimization epochs per iteration
        PPO_CLIP_EPSILON = 0.2,       # PPO clipping epsilon
        VALUE_LOSS_COEFF = 0.5,       # Coefficient for value loss
        ENTROPY_COEFF = 0.01,         # Coefficient for entropy bonus
        STANDARDIZE_ADV_PPO = True,   # Whether to standardize advantages
        NUM_ITERATIONS_PPO = 300,     # Number of PPO iterations (policy updates)
        GROUP_SIZE = 16,                # Number of rollouts (episodes) per group (G)
        MAX_STEPS_PER_EPISODE_PPO = 10000, # Max steps per episode
        EPSILON_STD = 1e-8,             # Small value to prevent division by zero
        INTERV_PRINT = 10,              # Print losses every INTERV_PRINT steps
        INTERV_AUTOSAVE = 10,          # Save model every INTERV_AUTOSAVE steps
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
        policy_network: PolicyNetwork = globals()[f"PolicyNetwork_{policy_version}"](E,D,F).to(device)
        policy_network.eval()
        last_iteration, best_reward = load_checkpoint(model_key, save_name, policy_network)
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
                    policy_dist = policy_network(state_tensor,mask_tensor)
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

