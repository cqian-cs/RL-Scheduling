import tqdm
import os
import time

num2time = lambda: time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
path_this = os.path.dirname(os.path.abspath(__file__))
name_this = 'RQ1'
VERSION = 'v3'

is_auto_upload = False
fs = None

#import util_fs
#is_auto_upload = os.environ.get('AUTO_UPLOAD','True').lower() in ('true', '1', 't', 'yes')
#fs = util_fs.connect_cf()


def get_model_key(mode,args,kwargs):
    if mode == 'GRPO':
        import rl_grpo
        kwargs['grpo_version']='GRPO'
        return rl_grpo.get_model_key(*args,**kwargs)
    elif mode == 'DrGRPO':
        import rl_grpo
        kwargs['grpo_version']='DrGRPO'
        return rl_grpo.get_model_key(*args,**kwargs)
    elif mode == 'PPO':
        import rl_ppo
        return rl_ppo.get_model_key(*args,**kwargs)
    

def train(mode,args,kwargs):
    if mode == 'GRPO':
        import rl_grpo
        kwargs['grpo_version']='GRPO'
        rl_grpo.train(*args,**kwargs)
    elif mode == 'DrGRPO':
        import rl_grpo
        kwargs['grpo_version']='DrGRPO'
        rl_grpo.train(*args,**kwargs)
    elif mode == 'PPO':
        import rl_ppo
        rl_ppo.train(*args,**kwargs)


def batch_train():
    all_kwargs = []
    for semi_save_key in [
        'G0H0C0W2I50N10F3',
        'G0H0C0W1I50N10F3',
    ]:
        for version in ['v1','v2','v3']:
            for ref in ['HEFT','self']:
                all_kwargs+=[
                    {"mode":"GRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.01}},
                    {"mode":"GRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.001}},
                    {"mode":"GRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.0001}},
                    {"mode":"GRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.00001}},
                    {"mode":"GRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.000001}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.01}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.001}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.0001}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.00001}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.000001}},
                    {"mode":"PPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.01}},
                    {"mode":"PPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.001}},
                    {"mode":"PPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.0001}},
                    {"mode":"PPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.00001}},
                    {"mode":"PPO",'args':[semi_save_key,"O1S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.000001}},
                    
                    {"mode":"GRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.01}},
                    {"mode":"GRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.001}},
                    {"mode":"GRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.0001}},
                    {"mode":"GRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.00001}},
                    {"mode":"GRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.000001}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.01}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.001}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.0001}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.00001}},
                    {"mode":"DrGRPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.000001}},
                    {"mode":"PPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.01}},
                    {"mode":"PPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.001}},
                    {"mode":"PPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.0001}},
                    {"mode":"PPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.00001}},
                    {"mode":"PPO",'args':[semi_save_key,"O0S1R1T1A1E1M3",version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.000001}},
                ]
    for kwargs in tqdm.tqdm(all_kwargs):
        train(**kwargs)
def test(mode,args,kwargs,seed_from,seed_to,alter_semi_save_key,overwrite=False):
    if mode == 'GRPO':
        import rl_grpo
        kwargs['grpo_version']='GRPO'
        return rl_grpo.test(*args,seed_from=seed_from,seed_to=seed_to,alter_semi_save_key=alter_semi_save_key,overwrite=overwrite,**kwargs)
    elif mode == 'DrGRPO':
        import rl_grpo
        kwargs['grpo_version']='DrGRPO'
        return rl_grpo.test(*args,seed_from=seed_from,seed_to=seed_to,alter_semi_save_key=alter_semi_save_key,overwrite=overwrite,**kwargs)
    elif mode == 'PPO':
        import rl_ppo
        return rl_ppo.test(*args,seed_from=seed_from,seed_to=seed_to,alter_semi_save_key=alter_semi_save_key,overwrite=overwrite,**kwargs)
    else:
        import environment, orjson, time
        model_key = ' '.join(map(str,[mode] + args))
        key = f"{VERSION}/{model_key}/{alter_semi_save_key} {seed_from} {seed_to}.json"
        path_save = f"{path_this}/results/{model_key}/{alter_semi_save_key} {seed_from} {seed_to}.json"
        if fs and not os.path.exists(path_save):
            fs.download(key, path_save)
        if not overwrite and os.path.exists(path_save):
            return orjson.loads(open(path_save,'rb').read())
        if mode == 'BE':
            env_run = environment.env_run_best_effort_HEFT
        elif mode == 'NSGA':
            env_run = environment.env_run_consolidating_NSGA
        elif mode == 'DP':
            env_run = environment.env_run_dynamic_planning_HEFT
        else:
            return
        mat_results = []
        clock_time = 0
        for seed in range(seed_from,seed_to):
            t0 = time.time()
            env = environment.new_env(f"{alter_semi_save_key}R{seed}")
            env_run(env,*args)
            t1 = time.time()
            clock_time += t1-t0
            mat_results.append(environment.env_score(env))
        ret = {
            'model_key':model_key,
            'save_name':'',
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
    

def draw_curves_once(cfg={
    'title': '_'.join(["G0H0C0W2I500N10F3","O1S1R1T1A1E1M3","v1","HEFT"]),
    'x_label': 'Episode',
    'y_label': 'Reward',
    'field': 'grpo_iteration_rewards',
    'save_name': 'auto_save',
    'curves' : {
        'lr=0.01':    {'mode':'GRPO','args':["G0H0C0W2I500N10F3","O1S1R1T1A1E1M3","v1","HEFT"],"kwargs":{'ACTOR_LR_GRPO':0.01}},
        'lr=0.001':   {'mode':'GRPO','args':["G0H0C0W2I500N10F3","O1S1R1T1A1E1M3","v1","HEFT"],"kwargs":{'ACTOR_LR_GRPO':0.001}},
        'lr=0.0001':  {'mode':'GRPO','args':["G0H0C0W2I500N10F3","O1S1R1T1A1E1M3","v1","HEFT"],"kwargs":{'ACTOR_LR_GRPO':0.0001}},
        'lr=0.00001': {'mode':'GRPO','args':["G0H0C0W2I500N10F3","O1S1R1T1A1E1M3","v1","HEFT"],"kwargs":{'ACTOR_LR_GRPO':0.00001}},
        'lr=0.000001':{'mode':'GRPO','args':["G0H0C0W2I500N10F3","O1S1R1T1A1E1M3","v1","HEFT"],"kwargs":{'ACTOR_LR_GRPO':0.000001}},
    }
}):
    import rl_grpo
    import rl_ppo
    data = []
    for name,kwargs in cfg['curves'].items():
        if kwargs['mode']=='GRPO':
            model_key = rl_grpo.get_model_key(*kwargs['args'],**kwargs['kwargs'],grpo_version='GRPO')
            training_log = rl_grpo.load_training_log(model_key, cfg['save_name'], default={})
        elif kwargs['mode']=='DrGRPO':
            model_key = rl_grpo.get_model_key(*kwargs['args'],**kwargs['kwargs'],grpo_version='DrGRPO')
            training_log = rl_grpo.load_training_log(model_key, cfg['save_name'], default={})
        elif kwargs['mode']=='PPO':
            model_key = rl_ppo.get_model_key(*kwargs['args'],**kwargs['kwargs'])
            training_log = rl_ppo.load_training_log(model_key, cfg['save_name'], default={})
        if not training_log:
            print(f"{num2time()} Error {model_key}\n{kwargs}")
        if cfg['field'] not in training_log:
            print(f"{model_key} not found field {cfg['field']} (only have {list(training_log.keys())})")
        data.append({
            'name':name,
            'y':training_log[cfg['field']]
        })
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    import os
    path_this = os.path.dirname(os.path.abspath(__file__))
    fig,ax = plt.subplots(figsize=(5,3))
    for d in data:
        ax.plot(d['y'],label=d['name'],linewidth=1)
    ax.legend()
    ax.set_xlabel(cfg['x_label'])
    ax.set_ylabel(cfg['y_label'])
    ax.set_title(cfg['title'])
    os.makedirs(f"{path_this}/img",exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{path_this}/img/{cfg['title']}.svg")
    plt.close()

def draw_curves():
    all_kwargs = []
    for semi_save_key in [
        'G0H0C0W2I50N10F3',
        'G0H0C0W1I50N10F3',
    ]:
        for rl_env_key in ['O1S1R1T1A1E1M3','O0S1R1T1A1E1M3']:
            for version in ['v1','v2','v3']:
                for ref in ['HEFT','self']:
                    all_kwargs.append({
                        'title': '_'.join(['GRPO',semi_save_key,rl_env_key,version,ref]),
                        'x_label': 'Episode',
                        'y_label': 'Reward',
                        'field': 'grpo_iteration_rewards',
                        'save_name': 'auto_save',
                        'curves' : {
                            'lr=0.01':    {'mode':'GRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.01}},
                            'lr=0.001':   {'mode':'GRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.001}},
                            'lr=0.0001':  {'mode':'GRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.0001}},
                            'lr=0.00001': {'mode':'GRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.00001}},
                            'lr=0.000001':{'mode':'GRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.000001}},
                        }
                    })
                    all_kwargs.append({
                        'title': '_'.join(['DrGRPO',semi_save_key,rl_env_key,version,ref]),
                        'x_label': 'Episode',
                        'y_label': 'Reward',
                        'field': 'grpo_iteration_rewards',
                        'save_name': 'auto_save',
                        'curves' : {
                            'lr=0.01':    {'mode':'DrGRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.01}},
                            'lr=0.001':   {'mode':'DrGRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.001}},
                            'lr=0.0001':  {'mode':'DrGRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.0001}},
                            'lr=0.00001': {'mode':'DrGRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.00001}},
                            'lr=0.000001':{'mode':'DrGRPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF_GRPO":0.0,'ACTOR_LR_GRPO':0.000001}},
                        }
                    })
                    all_kwargs.append({
                        'title': '_'.join(['PPO',semi_save_key,rl_env_key,version,ref]),
                        'x_label': 'Episode',
                        'y_label': 'Reward',
                        'field': 'ppo_iteration_rewards',
                        'save_name': 'auto_save',
                        'curves' : {
                            'lr=0.01':    {'mode':'PPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.01}},
                            'lr=0.001':   {'mode':'PPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.001}},
                            'lr=0.0001':  {'mode':'PPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.0001}},
                            'lr=0.00001': {'mode':'PPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.00001}},
                            'lr=0.000001':{'mode':'PPO','args':[semi_save_key,rl_env_key,version,ref],"kwargs":{"ENTROPY_COEFF":0.0,'ACTOR_LR':0.000001}},
                        }
                    })
    for kwargs in tqdm.tqdm(all_kwargs):
        draw_curves_once(kwargs)

def batch_test(
    seed_from = 100,
    seed_to = 200,
    overwrite = False
):
    import orjson
    import numpy as np
    rl_kwargs = [
        {'name': 'DrGRPO_v1_HEFT_W1', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.01, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_250'}}, 
        {'name': 'DrGRPO_v1_self_W1', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.01, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_250'}}, 
        {'name': 'DrGRPO_v2_HEFT_W1', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_160'}}, 
        {'name': 'DrGRPO_v2_self_W1', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_120'}}, 
        {'name': 'DrGRPO_v3_HEFT_W1', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'DrGRPO_v3_self_W1', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_40'}}, 
        {'name': 'DrGRPO_v1_HEFT_W2', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.01, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_200'}}, 
        {'name': 'DrGRPO_v1_self_W2', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.01, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'DrGRPO_v2_HEFT_W2', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_170'}}, 
        {'name': 'DrGRPO_v2_self_W2', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_90'}}, 
        {'name': 'DrGRPO_v3_HEFT_W2', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_70'}}, 
        {'name': 'DrGRPO_v3_self_W2', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'GRPO_v1_HEFT_W1', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'GRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'GRPO_v1_self_W1', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.01, 'grpo_version': 'GRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'GRPO_v2_HEFT_W1', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_300'}}, 
        {'name': 'GRPO_v2_self_W1', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_250'}}, 
        {'name': 'GRPO_v3_HEFT_W1', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_250'}}, 
        {'name': 'GRPO_v3_self_W1', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_300'}}, 
        {'name': 'GRPO_v1_HEFT_W2', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'GRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'GRPO_v1_self_W2', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'GRPO', 'save_name': 'best_model_210'}}, 
        {'name': 'GRPO_v2_HEFT_W2', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_290'}}, 
        {'name': 'GRPO_v2_self_W2', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_290'}}, 
        {'name': 'GRPO_v3_HEFT_W2', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_290'}}, 
        {'name': 'GRPO_v3_self_W2', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_100'}}, 
        {'name': 'PPO_v1_HEFT_W1', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.01, 'save_name': 'best_model_120'}}, 
        {'name': 'PPO_v1_self_W1', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.01, 'save_name': 'best_model_270'}}, 
        {'name': 'PPO_v2_HEFT_W1', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_290'}}, 
        {'name': 'PPO_v2_self_W1', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_290'}}, 
        {'name': 'PPO_v3_HEFT_W1', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_290'}}, 
        {'name': 'PPO_v3_self_W1', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_270'}}, 
        {'name': 'PPO_v1_HEFT_W2', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.01, 'save_name': 'best_model_270'}}, 
        {'name': 'PPO_v1_self_W2', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.01, 'save_name': 'best_model_160'}}, 
        {'name': 'PPO_v2_HEFT_W2', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_200'}}, 
        {'name': 'PPO_v2_self_W2', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_250'}}, 
        {'name': 'PPO_v3_HEFT_W2', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_250'}}, 
        {'name': 'PPO_v3_self_W2', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O1S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_290'}}, 
        {'name': 'DrGRPO_v1_HEFT_W1_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_290'}}, 
        {'name': 'DrGRPO_v1_self_W1_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'DrGRPO_v2_HEFT_W1_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'DrGRPO_v2_self_W1_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_300'}}, 
        {'name': 'DrGRPO_v3_HEFT_W1_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_40'}}, 
        {'name': 'DrGRPO_v3_self_W1_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_160'}}, 
        {'name': 'DrGRPO_v1_HEFT_W2_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'DrGRPO_v1_self_W2_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_270'}}, 
        {'name': 'DrGRPO_v2_HEFT_W2_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_250'}}, 
        {'name': 'DrGRPO_v2_self_W2_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_250'}}, 
        {'name': 'DrGRPO_v3_HEFT_W2_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_70'}}, 
        {'name': 'DrGRPO_v3_self_W2_O0', 'mode': 'DrGRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO', 'save_name': 'best_model_230'}}, 
        {'name': 'GRPO_v1_HEFT_W1_O0', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'GRPO', 'save_name': 'best_model_250'}}, 
        {'name': 'GRPO_v1_self_W1_O0', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.01, 'grpo_version': 'GRPO', 'save_name': 'best_model_290'}}, 
        {'name': 'GRPO_v2_HEFT_W1_O0', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_290'}}, 
        {'name': 'GRPO_v2_self_W1_O0', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_300'}}, 
        {'name': 'GRPO_v3_HEFT_W1_O0', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_110'}}, 
        {'name': 'GRPO_v3_self_W1_O0', 'mode': 'GRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_160'}}, 
        {'name': 'GRPO_v1_HEFT_W2_O0', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'GRPO', 'save_name': 'best_model_290'}}, 
        {'name': 'GRPO_v1_self_W2_O0', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'GRPO', 'save_name': 'best_model_180'}}, 
        {'name': 'GRPO_v2_HEFT_W2_O0', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_290'}}, 
        {'name': 'GRPO_v2_self_W2_O0', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'GRPO', 'save_name': 'best_model_250'}}, 
        {'name': 'GRPO_v3_HEFT_W2_O0', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'GRPO', 'save_name': 'best_model_240'}}, 
        {'name': 'GRPO_v3_self_W2_O0', 'mode': 'GRPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'GRPO', 'save_name': 'best_model_60'}}, 
        {'name': 'PPO_v1_HEFT_W1_O0', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.01, 'save_name': 'best_model_250'}}, 
        {'name': 'PPO_v1_self_W1_O0', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.01, 'save_name': 'best_model_300'}}, 
        {'name': 'PPO_v2_HEFT_W1_O0', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_290'}}, 
        {'name': 'PPO_v2_self_W1_O0', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_250'}}, 
        {'name': 'PPO_v3_HEFT_W1_O0', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_270'}}, 
        {'name': 'PPO_v3_self_W1_O0', 'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_250'}}, 
        {'name': 'PPO_v1_HEFT_W2_O0', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.01, 'save_name': 'best_model_250'}}, 
        {'name': 'PPO_v1_self_W2_O0', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.01, 'save_name': 'best_model_270'}}, 
        {'name': 'PPO_v2_HEFT_W2_O0', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_250'}}, 
        {'name': 'PPO_v2_self_W2_O0', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v2', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_250'}}, 
        {'name': 'PPO_v3_HEFT_W2_O0', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'HEFT'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_290'}}, 
        {'name': 'PPO_v3_self_W2_O0', 'mode': 'PPO', 'args': ['G0H0C0W2I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0, 'ACTOR_LR': 0.0001, 'save_name': 'best_model_260'}}
    ]
    heuristic_kwargs = [
        {"name":"DP_HEFT_EFT","mode":"DP","args":['upward_sort','EFT'],"kwargs":{}},
        {"name":"DP_HEFT_EFT_r","mode":"DP","args":['downward_sort','EFT'],"kwargs":{}},
        {"name":"DP_PEFT_EFT","mode":"DP","args":['oct_sort','EFT'],"kwargs":{}},
        {"name":"DP_PPTS_EFT","mode":"DP","args":['pcm_sort','EFT'],"kwargs":{}},
        {"name":"DP_IPPTS_EFT","mode":"DP","args":['p_sort','EFT'],"kwargs":{}},
        {"name":"DP_HEFT_EST","mode":"DP","args":['upward_sort','EST'],"kwargs":{}},
        {"name":"DP_HEFT_EST_r","mode":"DP","args":['downward_sort','EST'],"kwargs":{}},
        {"name":"DP_PEFT_EST","mode":"DP","args":['oct_sort','EST'],"kwargs":{}},
        {"name":"DP_PPTS_EST","mode":"DP","args":['pcm_sort','EST'],"kwargs":{}},
        {"name":"DP_IPPTS_EST","mode":"DP","args":['p_sort','EST'],"kwargs":{}},
        {"name":"NSGA_Small","mode":"NSGA","args":[50,20],"kwargs":{}},
        {"name":"NSGA_Big","mode":"NSGA","args":[500,100],"kwargs":{}},
        {"name":"BE_FIFO_EFT","mode":"BE","args":[None,False,'EFT'],"kwargs":{}},
        {"name":"BE_PEFT_EFT","mode":"BE","args":['oct_rank',False,'EFT'],"kwargs":{}},
        {"name":"BE_PPTS_EFT","mode":"BE","args":['pcm_rank',False,'EFT'],"kwargs":{}},
        {"name":"BE_IPPTS_EFT","mode":"BE","args":['p_rank',False,'EFT'],"kwargs":{}},
        {"name":"BE_FIFO_EFT_r","mode":"BE","args":[None,True,'EFT'],"kwargs":{}},
        {"name":"BE_PEFT_EFT_r","mode":"BE","args":['oct_rank',True,'EFT'],"kwargs":{}},
        {"name":"BE_PPTS_EFT_r","mode":"BE","args":['pcm_rank',True,'EFT'],"kwargs":{}},
        {"name":"BE_IPPTS_EFT_r","mode":"BE","args":['p_rank',True,'EFT'],"kwargs":{}},
        {"name":"BE_FIFO_EST","mode":"BE","args":[None,False,'EST'],"kwargs":{}},
        {"name":"BE_PEFT_EST","mode":"BE","args":['oct_rank',False,'EST'],"kwargs":{}},
        {"name":"BE_PPTS_EST","mode":"BE","args":['pcm_rank',False,'EST'],"kwargs":{}},
        {"name":"BE_IPPTS_EST","mode":"BE","args":['p_rank',False,'EST'],"kwargs":{}},
        {"name":"BE_FIFO_EST_r","mode":"BE","args":[None,True,'EST'],"kwargs":{}},
        {"name":"BE_PEFT_EST_r","mode":"BE","args":['oct_rank',True,'EST'],"kwargs":{}},
        {"name":"BE_PPTS_EST_r","mode":"BE","args":['pcm_rank',True,'EST'],"kwargs":{}},
        {"name":"BE_IPPTS_EST_r","mode":"BE","args":['p_rank',True,'EST'],"kwargs":{}},
    ]
    
    all_test_kwargs = []
    for alter_semi_save_key in [
        'G0H0C0W2I50N10F3',
        'G0H0C0W1I50N10F3',
    ]:
        for kwargs in rl_kwargs + heuristic_kwargs:
            all_test_kwargs.append({
                'seed_from':seed_from,
                'seed_to':seed_to,
                'alter_semi_save_key':alter_semi_save_key,
                'overwrite':overwrite,
                **kwargs
            })
    mat = [['alter_semi_save_key','model_key','algorithm','schedule_time','makespan','latency','energy','cv_makespan','cv_latency','cv_energy','L2_makespan','L2_latency','L2_energy','L2_cv_makespan','L2_cv_latency','L2_cv_energy','perf_similarity','cv_similarity','overall_similarity']]
    matrix = []
    for kwargs in tqdm.tqdm(all_test_kwargs):
        name = kwargs.pop('name')
        ret = test(**kwargs)
        data = np.array(ret['mat'])
        mean = data.mean(axis=0)
        cv = data.std(axis=0)/mean
        arr = [
            ret['alter_semi_save_key'],
            ret['model_key'],
            name,
            ret['time'],
        ]
        mat.append(arr)
        matrix.append(np.concat([mean,cv]))
    matrix = np.array(matrix)
    L2_matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True) # L2 normalize
    similarities = []
    for points in [L2_matrix[:,0:3],L2_matrix[:,3:6],L2_matrix[:,:]]:
        # overall_sort
        best_point = np.min(points,axis=0)
        worst_point = np.max(points,axis=0)
        distances_to_best = np.linalg.norm(points-best_point,axis=1)
        distances_to_worst = np.linalg.norm(points-worst_point,axis=1)
        similarities.append(distances_to_worst/(distances_to_worst+distances_to_best+1e-6))
    # performance_sort
    for i,arr in enumerate(np.column_stack((matrix,points, *similarities)).tolist(),start=1):
        mat[i]+=arr
    for _ in range(3):
        try:
            with open(f"{name_this}_test_raw.csv",'w') as f:
                for arr in mat:
                    f.write(','.join(map(str,arr))+'\n')
        except Exception as e:
            input(f"save failed:{e}\n close “{name_this}_test_raw.csv” and press Enter")
    with open(f"{name_this}_test.json",'wb') as f:
        f.write(orjson.dumps(mat))

def show_mat(mat,sortby=None,reversesort=False):
    import prettytable
    table = prettytable.PrettyTable()
    table.field_names = mat[0]
    for arr in mat[1:]:
        table.add_row(arr)
    text = table.get_string(sortby=sortby,reversesort=reversesort)
    return text

def show_table(mode="top_10"):
    """overall perf"""
    import orjson
    with open(f"{name_this}_test.json",'rb') as f:
        mat = orjson.loads(f.read())
    objs = [{k:v for k,v in zip(mat[0],arr)} for arr in mat[1:]]
    objs_in_test_set = [obj for obj in objs if obj['alter_semi_save_key']=='G0H0C0W2I50N10F3']
    objs_in_test_set.sort(key=lambda obj:(-obj['overall_similarity'],obj['schedule_time']))
    for i,obj in enumerate(objs_in_test_set,start=1):
        obj['rank'] = i
        obj['energy'] = obj['energy'] *0.0000001
    db = {obj['algorithm']:obj for obj in objs_in_test_set}
    inc = lambda n:f"{n:+.2f}%".replace('+','↑').replace('-','↓')
    
    def to_str(that_obj,base_obj,field,is_big_better):
        """that value (my improvement over that)"""
        if that_obj==base_obj:
            if that_obj[field]<100:
                return f"{obj[field]:.3f}"
            else:
                return f"{obj[field]:.2f}"
        if is_big_better:
            inc_value = (base_obj[field] - that_obj[field])/that_obj[field]
        else:
            inc_value = (that_obj[field] - base_obj[field])/that_obj[field]
        if that_obj[field] < 100:
            return f"{that_obj[field]:.3f}({inc_value*100:+.2f}%)"
        else:
            return f"{that_obj[field]:.2f}({inc_value*100:+.2f}%)"

    if mode.startswith("top"):
        import re
        ret = re.search(r"(\d+)",mode)
        top_n = None
        if ret:
            top_n = int(ret.group(1))
        base_name = 'DrGRPO_v1_HEFT_W1'
        base_obj = db[base_name]
        mat = [['rank','method','mean_makespan(ms)','cv_makespan','mean_latency(ms)','cv_latency','mean_energy(J)','cv_energy','CP','SS','ST(s)']]
        for obj in objs_in_test_set[:top_n]:
            mat.append([
                obj['rank'],
                obj['algorithm'],
                to_str(obj,base_obj,'makespan',False),
                to_str(obj,base_obj,'cv_makespan',False),
                to_str(obj,base_obj,'latency',False),
                to_str(obj,base_obj,'cv_latency',False),
                to_str(obj,base_obj,'energy',False),
                to_str(obj,base_obj,'cv_energy',False),
                to_str(obj,base_obj,'perf_similarity',True),
                to_str(obj,base_obj,'cv_similarity',True),
                to_str(obj,base_obj,'schedule_time',False),
            ])
        print(show_mat(mat,sortby='rank'))
    elif mode=="main_result_1":
        mat = [['method','Comprehensive Performance','Service Stability','Schedule Time']]
        base_obj = db['DrGRPO_v1_HEFT_W1']
        for name,algorithm in [
            ['DrGRPO-MLP-OMWS','DrGRPO_v1_HEFT_W1'],
            ['DrGRPO-MLP','DrGRPO_v1_self_W1_O0'],
            ['PPO-MLP','PPO_v1_self_W1_O0'],
            ['PPO-Transformer','PPO_v3_self_W1_O0'],
            ['BE_FCFS_EFT','BE_FIFO_EFT'],
            ['DP_PEFT','DP_PEFT_EFT'],
            ['CO_NSGA','NSGA_Small'],
        ]:
            obj = db[algorithm]
            mat.append([
                name,
                to_str(obj,base_obj,'perf_similarity',True),
                to_str(obj,base_obj,'cv_similarity',True),
                to_str(obj,base_obj,'schedule_time',True),
            ])
        print(show_mat(mat))
    
    elif mode=="main_result_2":
        mat = [['method','mean_makespan(ms)','cv_makespan','mean_latency(ms)','cv_latency','mean_energy(J)','cv_energy']]
        base_obj = db['DrGRPO_v1_HEFT_W1']
        for name,algorithm in [
            ['DrGRPO-MLP-OMWS','DrGRPO_v1_HEFT_W1'],
            ['DrGRPO-MLP','DrGRPO_v1_self_W1_O0'],
            ['PPO-MLP','PPO_v1_self_W1_O0'],
            ['PPO-Transformer','PPO_v3_self_W1_O0'],
            ['BE_FCFS_EFT','BE_FIFO_EFT'],
            ['DP_PEFT','DP_PEFT_EFT'],
            ['CO_NSGA','NSGA_Small'],
        ]:
            obj = db[algorithm]
            mat.append([
                name,
                f"{obj['makespan'       ]:.2f}",
                f"{obj['cv_makespan'    ]:.3f}",
                f"{obj['latency'        ]:.2f}",
                f"{obj['cv_latency'     ]:.3f}",
                f"{obj['energy'         ]:.3f}",
                f"{obj['cv_energy'      ]:.3f}",
            ])
        print(show_mat(mat))
    elif mode=="cmp_training_set":
        mat = [['method','CP_Balanced','SS_Balanced','CP_Long-tailed','SS_Long-tailed','CP_Improve','SS_Improve']]
        for name,algorithm_a,algorithm_b in [
            ['DrGRPO-MLP-OMWS'     ,'DrGRPO_v1_HEFT_W1'   ,'DrGRPO_v1_HEFT_W2'],
            ['DrGRPO-MLP'          ,'DrGRPO_v1_self_W1_O0','DrGRPO_v1_self_W2_O0'],
            ['PPO-MLP'             ,'PPO_v1_self_W1_O0'   ,'PPO_v1_self_W2_O0'],
            ['PPO-Transformer'     ,'PPO_v3_self_W1_O0'   ,'PPO_v3_self_W2_O0'],
        ]:
            obj_a = db[algorithm_a]
            obj_b = db[algorithm_b]
            mat.append([
                name,
                f"{obj_a['perf_similarity']:.3f}",
                f"{obj_a['cv_similarity'  ]:.3f}",
                f"{obj_b['perf_similarity']:.3f}",
                f"{obj_b['cv_similarity'  ]:.3f}",
                f"{100*(obj_a['perf_similarity']-obj_b['perf_similarity'])/obj_b['perf_similarity']:+.2f}%",
                f"{100*(obj_a['cv_similarity']-obj_b['cv_similarity'])/obj_b['cv_similarity']:+.2f}%",
            ])
        print(show_mat(mat))
    elif mode=="cmp_observation":
        mat = [['method','CP_9-dim-obs','SS_9-dim-obs','CP_11-dim-obs','SS_11-dim-obs','CP_Improve','SS_Improve']]
        for name,algorithm_a,algorithm_b in [
            ['DrGRPO-MLP-OMWS'     ,'DrGRPO_v1_HEFT_W1','DrGRPO_v1_HEFT_W1_O0'],
            ['DrGRPO-MLP'  ,'DrGRPO_v1_self_W1','DrGRPO_v1_self_W1_O0'],
            ['PPO-MLP'             ,'PPO_v1_self_W1'   ,'PPO_v1_self_W1_O0'],
            ['PPO-Transformer'     ,'PPO_v3_self_W1'   ,'PPO_v3_self_W1_O0'],
        ]:
            obj_a = db[algorithm_a]
            obj_b = db[algorithm_b]
            mat.append([
                name,
                f"{obj_a['perf_similarity']:.3f}",
                f"{obj_a['cv_similarity'  ]:.3f}",
                f"{obj_b['perf_similarity']:.3f}",
                f"{obj_b['cv_similarity'  ]:.3f}",
                f"{100*(obj_a['perf_similarity']-obj_b['perf_similarity'])/obj_b['perf_similarity']:+.2f}%",
                f"{100*(obj_a['cv_similarity']-obj_b['cv_similarity'])/obj_b['cv_similarity']:+.2f}%",
            ])
        print(show_mat(mat))
    elif mode=="cmp_reference":
        mat = [['method','CP_HEFT','SS_HEFT','CP_Self','SS_Self','CP_Improve','SS_Improve']]
        for name,algorithm_a,algorithm_b in [
            ['DrGRPO-MLP-OMWS'     ,'DrGRPO_v1_HEFT_W1'   ,'DrGRPO_v1_self_W1'],
            ['DrGRPO-MLP'          ,'DrGRPO_v1_HEFT_W1_O0'   ,'DrGRPO_v1_self_W1_O0'],
            ['PPO-MLP'             ,'PPO_v1_HEFT_W1_O0'      ,'PPO_v1_self_W1_O0'],
            ['PPO-Transformer'     ,'PPO_v3_HEFT_W1_O0'      ,'PPO_v3_self_W1_O0'],
        ]:
            obj_a = db[algorithm_a]
            obj_b = db[algorithm_b]
            mat.append([
                name,
                f"{obj_a['perf_similarity']:.3f}",
                f"{obj_a['cv_similarity'  ]:.3f}",
                f"{obj_b['perf_similarity']:.3f}",
                f"{obj_b['cv_similarity'  ]:.3f}",
                f"{100*(obj_a['perf_similarity']-obj_b['perf_similarity'])/obj_b['perf_similarity']:+.2f}%",
                f"{100*(obj_a['cv_similarity']-obj_b['cv_similarity'])/obj_b['cv_similarity']:+.2f}%",
            ])
        print(show_mat(mat))
    

def draw_fig_curve():
    import orjson
    data = []
    for method, kwargs_list in [
        ['DrGRPO-MLP-OMWS', [
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.01, 'grpo_version': 'DrGRPO'}},
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO'}},
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'DrGRPO'}},
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.00001, 'grpo_version': 'DrGRPO'}},
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O1S1R1T1A1E1M3', 'v1', 'HEFT'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.000001, 'grpo_version': 'DrGRPO'}},
        ]],
        ['DrGRPO-MLP', [
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.01, 'grpo_version': 'DrGRPO'}},
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.001, 'grpo_version': 'DrGRPO'}},
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.0001, 'grpo_version': 'DrGRPO'}},
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.00001, 'grpo_version': 'DrGRPO'}},
            {'mode': 'DrGRPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF_GRPO': 0.0, 'ACTOR_LR_GRPO': 0.000001, 'grpo_version': 'DrGRPO'}},
        ]],
        ['PPO-MLP', [
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.01}},
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.001}},
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.0001}},
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.00001}},
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v1', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.000001}},
        ]],
        ['PPO-Transformer', [
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.01}},
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.001}},
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.0001}},
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.00001}},
            {'mode': 'PPO', 'args': ['G0H0C0W1I50N10F3', 'O0S1R1T1A1E1M3', 'v3', 'self'], 'kwargs': {'ENTROPY_COEFF': 0.0,'ACTOR_LR': 0.000001}},
        ]],
    ]:
        ys = []
        for kwargs in kwargs_list:
            model_key = get_model_key(**kwargs)
            with open(f"{path_this}/results/{model_key}/auto_save.json", "rb") as f:
                auto_save_data = orjson.loads(f.read())
            if kwargs['mode'] == 'PPO':
                y = auto_save_data['ppo_iteration_rewards']
            else:
                y = auto_save_data['grpo_iteration_rewards']
            ys.append(y)
        data.append([method, ys])

    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14

    fig, axes = plt.subplots(1, 4, figsize=(8, 3), dpi=100, sharey=True)

    x = list(range(1, 1+len(data[0][1][0])))

    # First subplot
    axes[0].set_title(data[0][0],fontsize=14)
    line0, = axes[0].plot(x, data[0][1][0], lw=1.0, label='lr = 0.01')
    line1, = axes[0].plot(x, data[0][1][1], lw=1.0, label='lr = 0.001')
    line2, = axes[0].plot(x, data[0][1][2], lw=1.0, label='lr = 0.0001')
    line3, = axes[0].plot(x, data[0][1][3], lw=1.0, label='lr = 0.00001')
    line4, = axes[0].plot(x, data[0][1][4], lw=1.0, label='lr = 0.000001')

    # Second subplot
    axes[1].set_title(data[1][0],fontsize=14)
    axes[1].plot(x, data[1][1][0], lw=1.0)
    axes[1].plot(x, data[1][1][1], lw=1.0)
    axes[1].plot(x, data[1][1][2], lw=1.0)
    axes[1].plot(x, data[1][1][3], lw=1.0)
    axes[1].plot(x, data[1][1][4], lw=1.0)

    # Third subplot
    axes[2].set_title(data[2][0],fontsize=14)
    axes[2].plot(x, data[2][1][0], lw=1.0)
    axes[2].plot(x, data[2][1][1], lw=1.0)
    axes[2].plot(x, data[2][1][2], lw=1.0)
    axes[2].plot(x, data[2][1][3], lw=1.0)
    axes[2].plot(x, data[2][1][4], lw=1.0)

    axes[3].set_title(data[3][0],fontsize=14)
    axes[3].plot(x, data[3][1][0], lw=1.0)
    axes[3].plot(x, data[3][1][1], lw=1.0)
    axes[3].plot(x, data[3][1][2], lw=1.0)
    axes[3].plot(x, data[3][1][3], lw=1.0)
    axes[3].plot(x, data[3][1][4], lw=1.0)

    axes[0].set_ylabel('Reward')
    fig.supxlabel('Iterations')
    fig.legend(
        [line0, line1, line2, line3, line4], 
        ['lr = 0.01', 'lr = 0.001', 'lr = 0.0001', 'lr = 0.00001', 'lr = 0.000001'], 
        ncol=5,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.85),
        columnspacing=1,
    )
    plt.subplots_adjust(
        top=0.75,
        bottom=0.2,
        wspace=0.1
    )
    plt.show()






def plot_TFN():
    def new_TFN(t,rnd):
        if t==0:
            return 0
        ξ=t
        ξl = 0.75*ξ + rnd[0]*0.25*ξ
        ξr = ξ + rnd[1]*min(0.2*ξ,ξ-ξl)
        fc = (ξ - ξl) / (ξr - ξl)
        u = rnd[2]
        if u <= fc:
            c = ξl + (u * (ξr - ξl) * (ξ - ξl)) ** 0.5
        else:
            c = ξr - ((1 - u) * (ξr - ξl) * (ξr - ξ)) ** 0.5
        return c
    import matplotlib.pyplot as plt
    import random
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    fig,axs = plt.subplots(1,2,figsize=(5.0, 2.0),gridspec_kw={'width_ratios': [0.85,1.15]})
    ξ=1000
    ξl = 0.75*ξ + 0.5*0.25*ξ
    ξr = ξ + 0.5*min(0.2*ξ,ξ-ξl)
    axs[0].fill_between([ξl,ξ,ξr],[0,1,0],[0,0,0])
    axs[0].plot([ξl,ξ,ξr],[0,1,0],color='k',linewidth=1)
    
    axs[0].set_xticks([ξl,ξ,ξr])
    axs[0].set_xticklabels(["$\\xi^l$","$\\xi^m$","$\\xi^u$"])
    #axs[0].set_ylabel("$\\mu_{\\tilde{\\xi}}(x)$")
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    xmin,xmax = ξl-50,ξr+50
    ymin,ymax = 0,1.2
    axs[0].set_xlim([xmin,xmax])
    axs[0].set_ylim([ymin,ymax])
    axs[0].set_yticks([0,1])
    axs[0].set_yticklabels([0,1])
    axs[0].tick_params(length=0)
    axs[0].plot([ξ,ξ],[0,1],color='k',linestyle='--',linewidth=1)
    axs[0].plot([xmin,xmax],[1,1],color='k',linestyle='--',linewidth=1)
    axs[0].annotate('', xy=(xmax, ymin), xytext=(xmin-10, ymin),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
    axs[0].annotate('', xy=(xmin, ymax), xytext=(xmin, -0.1),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
    axs[0].set_xlabel("(a) The membership function $\\mu_{\\tilde{\\xi}}(x)$")
    
    rng = random.Random()
    samples = [new_TFN(1000,(rng.random(),rng.random(),rng.random())) for _ in range(30000)]
    axs[1].hist(samples, bins=100, density=True)
    axs[1].set_xticks([ξl,ξ,ξr])
    axs[1].set_xticklabels(["$\\xi^l$","$\\xi^m$","$\\xi^u$"])
    axs[1].set_xlabel("(b) The triangular fuzzy variable $\\tilde{\\xi}$")
    axs[1].set_ylabel("Probability Density")
    plt.tight_layout()
    plt.savefig('triangular_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved as './triangular_distribution.pdf'")


def show_tables():
    print("=== Table 2 ===")
    show_table('main_result_1')
    print("=== Table 3 ===")
    show_table('main_result_2')
    print("=== Table 4 ===")
    show_table('cmp_training_set')
    print("=== Table 5 ===")
    show_table('cmp_observation')
    print("=== Table 6 ===")
    show_table('cmp_reference')

if __name__=='__main__':
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


    #batch_train()
    #draw_curves()
    #batch_test(100,200)
    #show_table('main_result_1')
    #show_table('main_result_2')
    #show_table('cmp_training_set')
    #show_table('cmp_observation')
    #show_table('cmp_reference')
    #batch_test_merge()
    #draw_fig_curve()