import numpy as np
import random
import environment
import os
import re
import torch
path_this = os.path.dirname(os.path.abspath(__file__))
name_this = os.path.basename(__file__).rsplit('.',1)[0]
INF = 0x7f7f7f7f

def get_dim_action_space(semi_save_key):
    env = environment.new_env(f"{semi_save_key}R0")
    return env['n_devices']


dim_observation_v0 = 11
def env_observation_v0(env,state):
    obs = []
    device_masks = []
    tick = state['tick']
    E = state['max_E']
    D = state['D']
    F = state['F']
    for event_id,event in enumerate(state['events'][:E]):
        request_id = event['request_id']
        app_id = event['app_id']
        task_id = event['task_id']
        device_mask = env['mask'][app_id][task_id]
        device_masks.append(device_mask)
        upward_rank = env['upward_rank'][app_id][task_id]
        downward_rank = env['downward_rank'][app_id][task_id]
        forward_layer = env['forward_layer'][app_id][task_id]
        backward_layer = env['backward_layer'][app_id][task_id]
        is_critical_path = env['is_critical_path'][app_id][task_id]
        n_pred = len(env['pred'][app_id][task_id])
        n_succ = len(env['succ'][app_id][task_id])
        n_succ_2th = sum(len(env['succ'][app_id][succ_task_id]) for succ_task_id in env['succ'][app_id][task_id])
        arrival_time = env['requests'][request_id]['arrival_time']
        obs_task = [
            upward_rank,                                   # o[0]
            downward_rank,                                 # o[1]
            forward_layer,                                 # o[2]
            backward_layer,                                # o[3]
            is_critical_path,                              # o[4]  0|1 is-critical-path 
            -1+2*n_pred/max(1,n_pred+n_succ),              # o[5] -1~1 out-degree-ratio
            -1+2*n_succ/max(1,n_succ+n_succ_2th),          # o[6] -1~1 out-degree-ratio of successors
            tick-arrival_time,                             # o[7] -1~1 elapsed time from request arrival
        ]
        obs_task_devices = []
        for device_id,is_available in enumerate(device_mask):
            ITree = env['ITree'][device_id]
            obs_devices = [
                tick-(ITree[-1] if ITree else 0),          # o[8] -1~1 idle time
            ]
            if is_available:
                AST,AFT = environment.get_AST_AFT(env,request_id,task_id,device_id,fuzz=False)
                obs_devices += [
                    AST-tick,        # o[9] -1~1 waiting time
                    AFT-AST          # o[10] -1~1 execution time
                ]
            else:
                obs_devices += [None,None] # o[9-10] Non optional devices, queue time and execution time do not participate in normalization
            obs_task_devices.append(obs_task+obs_devices)
        obs.append(obs_task_devices)
    torch_obs_padded = torch.zeros(E,D,F)
    torch_masks_padded = torch.zeros(E,D,dtype=torch.bool) # default: False
    if not obs:
        return torch_obs_padded, torch_masks_padded
    np_obs = np.array(obs,dtype=np.float32)
    for start,end in [(0,1),(2,3),(7,10)]:
        features_to_normalize = np_obs[:, :, start:end+1]
        global_min = np.nanmin(features_to_normalize)
        global_max = np.nanmax(features_to_normalize)
        if global_min == global_max:
            global_max+=1.0  # Prevent dividing by zero
        normalized_features = np.nan_to_num(
            (2 * (features_to_normalize - global_min) / (global_max - global_min)) - 1,
            nan=1.0,
            posinf=1.0,
            neginf=1.0
        )
        np_obs[:, :, start:end+1] = normalized_features
    #with np.printoptions(linewidth=120):
    #    print(np_obs)
    np_device_masks = np.stack(device_masks)
    torch_masks_padded[:len(np_device_masks)] = torch.from_numpy(np_device_masks)
    torch_obs_padded[:len(np_obs)] = torch.from_numpy(np_obs)
    return torch_obs_padded, torch_masks_padded


dim_observation_v1 = 9
def env_observation_v1(env,state):
    obs = []
    device_masks = []
    tick = state['tick']
    E = state['max_E']
    D = state['D']
    F = state['F']
    for event_id,event in enumerate(state['events'][:E]):
        request_id = event['request_id']
        app_id = event['app_id']
        task_id = event['task_id']
        device_mask = env['mask'][app_id][task_id]
        device_masks.append(device_mask)
        upward_rank = env['upward_rank'][app_id][task_id]
        downward_rank = env['downward_rank'][app_id][task_id]
        forward_layer = env['forward_layer'][app_id][task_id]
        backward_layer = env['backward_layer'][app_id][task_id]
        is_critical_path = env['is_critical_path'][app_id][task_id]
        n_pred = len(env['pred'][app_id][task_id])
        n_succ = len(env['succ'][app_id][task_id])
        n_succ_2th = sum(len(env['succ'][app_id][succ_task_id]) for succ_task_id in env['succ'][app_id][task_id])
        arrival_time = env['requests'][request_id]['arrival_time']
        obs_task = [
            upward_rank/(upward_rank+downward_rank),       # o[0]  0~1 weighted position 
            forward_layer/(forward_layer+backward_layer),  # o[1]  0~1 unweighted position
            is_critical_path,                              # o[2]  0|1 is-critical-path 
            -1+2*n_pred/max(1,n_pred+n_succ),              # o[3] -1~1 out-degree-ratio
            -1+2*n_succ/max(1,n_succ+n_succ_2th),          # o[4] -1~1 out-degree-ratio of successors
            tick-arrival_time,                             # o[5] -1~1 elapsed time from request arrival
        ]
        obs_task_devices = []
        for device_id,is_available in enumerate(device_mask):
            ITree = env['ITree'][device_id]
            obs_devices = [
                tick-(ITree[-1] if ITree else 0),          # o[6] -1~1 idle time
            ]
            if is_available:
                AST,AFT = environment.get_AST_AFT(env,request_id,task_id,device_id,fuzz=False)
                obs_devices += [
                    AST-tick,        # o[7] -1~1 waiting time
                    AFT-AST          # o[8] -1~1 execution time
                ]
            else:
                obs_devices += [None,None] # o[7-8] Non optional devices, queue time and execution time do not participate in normalization
            obs_task_devices.append(obs_task+obs_devices)
        obs.append(obs_task_devices)
    torch_obs_padded = torch.zeros(E,D,F)
    torch_masks_padded = torch.zeros(E,D,dtype=torch.bool) # default: False
    if not obs:
        return torch_obs_padded, torch_masks_padded
    np_obs = np.array(obs,dtype=np.float32)
    for start,end in [(5,8)]:
        features_to_normalize = np_obs[:, :, start:end+1]
        global_min = np.nanmin(features_to_normalize)
        global_max = np.nanmax(features_to_normalize)
        if global_min == global_max:
            global_max+=1.0  # Prevent dividing by zero
        normalized_features = np.nan_to_num(
            (2 * (features_to_normalize - global_min) / (global_max - global_min)) - 1,
            nan=1.0,
            posinf=1.0,
            neginf=1.0
        )
        np_obs[:, :, start:end+1] = normalized_features
    #with np.printoptions(linewidth=120):
    #    print(np_obs)
    np_device_masks = np.stack(device_masks)
    torch_masks_padded[:len(np_device_masks)] = torch.from_numpy(np_device_masks)
    torch_obs_padded[:len(np_obs)] = torch.from_numpy(np_obs)
    return torch_obs_padded, torch_masks_padded

def env_step_v1(env,state,action):
    # Action is tuple (event_id,device_id)
    event_id, device_id = action
    event = state['events'].pop(event_id)
    env['set_task_device'](event['request_id'],event['app_id'],event['task_id'],device_id)
    done = False
    if not state['events']:
        state['tick'],state['events'] = environment.env_step(env)
        if not state['events']:
            done = True
    return done


def reward_v1(env,state,done,action):
    reward = 0.0
    if done:
        scores = environment.env_score(env)
        for score,score_baseline,factor in zip(scores,state['baseline_scores'],state['factors']):
                improve = (score_baseline - score) / score_baseline
                reward += improve * factor
    return reward
def env_state_init(save_key,skip_baseline = False):
    env = environment.new_env(save_key)
    if skip_baseline:
        tick, events = environment.env_step(env)
        return env, {'tick':tick,'events':events}
    else:
        environment.env_run_best_effort_HEFT(env,show=False)
        baseline_scores = environment.env_score(env)
        environment.env_reset(env)
        tick, events = environment.env_step(env)
        return env, {'tick':tick,'events':events, 'baseline_scores': baseline_scores}



class SchedulerEnv:
    def __init__(self, semi_save_key = "G0H0C0W1I5N10F3", rl_env_key="O1S1R1T1A1E1M5",skip_baseline=False):
        self.semi_save_key = semi_save_key
        self.rl_env_key = rl_env_key
        print(rl_env_key)
        ret = re.match(r"O([0-9]+)S([0-9]+)R([0-9]+)T([0-9]+)A([0-9]+)E([0-9]+)M([0-9]+)",rl_env_key)
        global_funcs = globals()
        self.observation = global_funcs[f"env_observation_v{ret.group(1)}"]
        self.n_features = global_funcs[f"dim_observation_v{ret.group(1)}"]
        self.n_devices = get_dim_action_space(semi_save_key)
        self.step_func = global_funcs[f"env_step_v{ret.group(2)}"]
        self.reward_func = global_funcs[f"reward_v{ret.group(3)}"]
        self.reward_weights = [
            float(ret.group(4)),
            float(ret.group(5)),
            float(ret.group(6)),
        ]
        self.obs_max_E = int(ret.group(7))
        self.skip_baseline = skip_baseline
    def reset(self,seed=None):
        if seed == None:
            seed = random.randint(1000,99999)
        self.env, self.state = env_state_init(f"{self.semi_save_key}R{seed}",self.skip_baseline)
        self.state['factors'] = self.reward_weights
        self.state['max_E'] = self.obs_max_E
        self.state['D'] = self.n_devices
        self.state['F'] = self.n_features
        obs, mask = self.observation(self.env,self.state)
        return obs, mask

    def step(self, action_flat):
        action = divmod(action_flat, self.n_devices)
        done = self.step_func(self.env,self.state,action)
        if self.skip_baseline:
            reward = 0
        else:
            reward_item = self.reward_func(self.env,self.state,done,action)
            reward = torch.tensor([reward_item],dtype=torch.float32)
        obs, mask = self.observation(self.env,self.state)
        return obs, mask, reward, done

    def score(self):
        scores = environment.env_score(self.env)
        results = scores
        if self.skip_baseline:
            return results
        else:
            for score,score_baseline in zip(scores,self.state['baseline_scores']):
                improve = (score_baseline - score) / score_baseline
                results.append(improve)
            return results
    def stringify_score(self,results):
        if self.skip_baseline:
            output = [
                f"Total Makespan: {results[0]}",
                f"Avg Makespan: {results[1]}",
                f"Total Evergy: {results[2]}",
            ]
        else:
            output = [
                f"Total Makespan: {results[0]} ({results[3]:+.2%})".replace('+','↑').replace('-','↓'),
                f"Avg Makespan: {results[1]} ({results[4]:+.2%})".replace('+','↑').replace('-','↓'),
                f"Total Evergy: {results[2]} ({results[5]:+.2%})".replace('+','↑').replace('-','↓'),
            ]
        return ' | '.join(output)