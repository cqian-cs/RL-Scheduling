import numpy as np
import random
import heapq
import re
import os
import json
path_this = os.path.dirname(os.path.abspath(__file__))
INF=0x7f7f7f7f

"""
Heterogeneous Edge Computing Settings
    ref“GPU-Disaggregated Serving for Deep Learning Recommendation Models at Scale”2025
    GPU server：128 cpu cores + 1 GPU card(9TF)
    CPU server：128 cpu cores + 0 GPU card
    GPU 1TF idle power: 40/9 = 4.44W = 4.44J/s = 4.44mJ/ms = 4.44 mJ/tick (1tick = 1ms)
    GPU 1TF busy power: 60/9 = 6.66W = 6.66 mJ/tick
    48 CPU core = 1TF
    CPU 1TF idle power = 66.00 W
    CPU 1TF busy power = 95.64 W
    
    NVIDIA A2: 40~60W, 9TF
    https://www.nvidia.com/content/dam/en-zz/solutions/data-center/a2/pdf/a2-datasheet.pdf
    Intel(R) Xeon(R) 6756E: TDP=225W, 128 cpu cores
    https://www.intel.cn/content/www/cn/zh/products/sku/240361/intel-xeon-6756e-processor-96m-cache-1-80-ghz/specifications.html
    
"""
# p5: CPU partition on the GPU server (1=16 cores=1/3TF, 2=32 cores=2/3TF, 3=48 cores=1TF)
p5=[[2, 3], [1, 1, 3], [1, 2, 2], [1, 1, 1, 2], [1, 1, 1, 1, 1]]
# p8: CPU partition on the CPU server (1=16 cores=1/3TF, 2=32 cores=2/3TF, 3=48 cores=1TF)
p8=[[2, 3, 3], [1, 1, 3, 3], [1, 2, 2, 3], [2, 2, 2, 2], [1, 1, 1, 2, 3], [1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 3], [1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1]]
# p9: GPU partition on the GPU server (1=1TF, 2=2TF)
p9=[[1, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1]]
def new_env(save_key='G0H0C0W1I25N10F0T12E95R0'):
    """create new environment (a heterogeneous GPU server and a homogeneous CPU server.)
    
    Args:
        save_key (str): Specification of new environment. Defaults to 'G0H0C0W000001I25N10R0'.
            G: 0~5 partitioning config of GPU units in the Heterogeneous Server
              [[1, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1]]
            H: 0~5 partitioning config of CPU units in the Heterogeneous Server
              [[2, 3], [1, 1, 3], [1, 2, 2], [1, 1, 1, 2], [1, 1, 1, 1, 1]]
            C: 0~9 partitioning config of CPU units in the Homogeneous Server
              [[2, 3], [1, 1, 3], [1, 2, 2], [1, 1, 1, 2], [1, 1, 1, 1, 1]]
            W: 0~2 workflow dataset
               0: single-workflow (test)
               1: alibaba cluster-data-v2018 JobID 102 to 108
               2: alibaba cluster-data-v2018 JobID 119 to 149
            I: average interval between requests (in ms)
            N: number of requests
            F: 0~3 whether to fluctuate performance and network latency
               0: no fluctuation
               1: fluctuate performance
               2: fluctuate network latency
               3: fluctuate both performance and network latency
            T12E95(optional): use Markov Chain to generate arrival app_id sequence (instead of random choosen)
              T12E95: tail=1.2, entropy=0.95
            R: random seed, generating app_id sequence and arrival_time sequence
    Returns:
        env (dict): environment
            env["save_key"]         (str)         : save_key
            env["HG"]               (list[int])   : GPU units' amount (in Heterogeneous Server)
            env["HC"]               (list[int])   : CPU units' amount (in Heterogeneous Server)
            env["CC"]               (list[int])   : CPU units' amount (in Homogeneous Server)
            env["n_gpu"]            (int)         : number of GPU units
            env["n_HN"]             (int)         : number of computing units in Heterogeneous Server
            env["n_device"]         (int)         : number of computing units in both servers
            env["r"][device_id][device_id] (list): bandwidth between two devices
            env["ext"][workflow_id][task_id][device_id] (int): execution time of the task on the device (in millisecond)
            env["fuzz_perf"]        (bool)        : whether to fuzz performance
            env["fuzz_net"]         (bool)        : whether to fuzz network latency
            env["rnd_ext"][request_id][task_id]  : random numbers for fuzzing run-time performance
            env["rnd_trt"][request_id][task_id]  : random numbers for fuzzing network latency
            env["cumsum_tasks"]     (list[int])  : helper to flatten all requests' tasks in consolidating scheduler
                (request_id, task_id) to global_id : i = env['cumsum_tasks'][request_id]+task_id
            env["G"] (int): number of workflows
            env["N"][workflow_id] (int): number of tasks in each workflow
            env["pred"][workflow_id][task_id] (list): predecessor task ids
            env["succ"][workflow_id][task_id] (list): successor task ids
            env["c"][workflow_id][task_id] (int): task category: 'C'=CPU, 'G'=GPU
            env["w"][workflow_id][task_id] (float): task workload
            env["pw"][workflow_id][task_id] (float): sum of task's predecessors' workload
            env["alpha"][workflow_id][task_id] (list): input datasize
            env["beta"][workflow_id][task_id] (list): output datasize
            env["topsort"][workflow_id] (list): topsorted task ids, from entry task to exit task
            env["revert_topsort"][workflow_id] (list): revert topsorted task ids, from exit task to entry task
            env["upward_rank"][workflow_id][task_id] (float): upward rank of the task
            env["downward_rank"][workflow_id][task_id] (float): downward rank of the task
            env["upward_sort"][workflow_id] (list): sorted task ids by upward rank
            env["downward_sort"][workflow_id] (list): sorted task ids by downward rank
            env["is_critical_path"][workflow_id][task_id] (int): 1 if the task is on critical path, 0 otherwise
            env["forward_layer"][workflow_id][task_id] (int): depth of the task from entry task to exit task
            env["backward_layer"][workflow_id][task_id] (int): depth of the task from exit task to entry task
            env["ITree"][ITree_id] (list): interval tree of each device
            env["requests"] (list): instances of workflow requests
            env["mask"][workflow_id][task_id][device_id] (bool): whether a task is executable by a device
    """    
    # --- argument parse ---
    ret = re.match("G([0-9]+)H([0-9]+)C([0-9]+)W([0-9]+)I([0-9]+)N([0-9]+)F([0-9]+)(T([0-9]+)E([0-9]+))?R([0-9]+)",save_key)
    i_partition_HG = int(ret.group(1))
    i_partition_HC = int(ret.group(2))
    i_partition_CC = int(ret.group(3))
    i_workflows = int(ret.group(4))
    avg_interval = int(ret.group(5))
    n_requests = int(ret.group(6))
    fluctuation = int(ret.group(7))%4
    if ret.group(8):
        use_markov = True
        tail = float(ret.group(9))/10
        entropy = float(ret.group(10))/100
    else:
        use_markov = False
    seed = int(ret.group(11))
    # --- static part ---
    HG = np.array(p9[i_partition_HG],dtype=int)
    HC = np.array(p5[i_partition_HC],dtype=int)
    CC = np.array(p8[i_partition_CC],dtype=int)
    n_gpu = len(HG)            # 0     ≤ device_id < n_gpu    device is a logical GPU unit
    n_HN = n_gpu+len(HC)       # n_gpu ≤ device_id < n_HN     device is a logical CPU unit (on GPU server)
    n_devices = n_HN + len(CC) # n_HN  ≤ device_id < n_device device is a logical CPU unit (on CPU server)
    with open(f"{path_this}/env_workflows_{i_workflows}.json") as f:
        env = json.load(f)
    ext_multiple = np.concatenate([2/HG,6/HC,6/CC])
    ext = [(ext_multiple*np.array(workloads).reshape(-1,1)).astype(int) for workloads in env['w']]
    env['mask'] = []
    for _ext,c in zip(ext,env['c']):
        mask = np.ones_like(_ext,dtype=bool)
        mask[np.array(c)=='C',:n_gpu] = False # mask: whether task-device pair available.
        _ext[~mask] = INF
        env['mask'].append(mask)
    env['save_key'] = save_key
    env['HG']=HG.tolist()
    env['HC']=HC.tolist()
    env['CC']=CC.tolist()
    env['n_gpu']=n_gpu
    env['n_HN']=n_HN
    env['n_devices']=n_devices
    env['r']   = [[5]*n_HN+[3]*len(CC)]*n_HN + [[3]*n_HN+[5]*len(CC)]*len(CC)
    env['ext'] = [_ext.tolist() for _ext in ext]
    env['fuzz_perf'] = bool(fluctuation&1)
    env['fuzz_net'] = bool(fluctuation&2)
    # --- random part ---
    rng = np.random.RandomState(seed)
    arrival_time_sequence = [0]+np.cumsum(rng.poisson(avg_interval, n_requests-1)).tolist()
    if env['G']==1:
        app_id_sequence = [0]*n_requests
    elif use_markov:
        markov_chain = (seed, env['G'], n_requests, tail, entropy)
        markov_nodes = rng.shuffle(list(range(env['G'])))
        app_id_sequence = [markov_nodes[i] for i in markov_chain]
    else:
        app_id_sequence = rng.choice(env['G'],n_requests)
    if env['fuzz_perf']:
        env['rnd_ext'] = [rng.random((env['N'][app_id],3)).tolist() for app_id in app_id_sequence]
    if env['fuzz_net']:
        env['rnd_trt'] = [[rng.random((len(alpha),3)).tolist() for alpha in env['alpha'][app_id]] for app_id in app_id_sequence]
    env['cumsum_tasks']=[0]+np.cumsum([env['N'][app_id] for app_id in app_id_sequence]).tolist()
    # --- dynamic part ---
    env['ITree'] =[[] for _ in range(n_devices)]
    env['requests'] = [
        {
            'request_id':request_id,
            'app_id':app_id,
            'arrival_time': arrival_time,
            'n_tasks': env['N'][app_id],
            'activities':[
                {
                    'id':task_id,
                    'device_id':None,
                    'start_time':None,
                    'finish_time':INF,
                }
                for task_id in range(env['N'][app_id])
            ]
        }
        for request_id,(arrival_time,app_id) in enumerate(zip(
            arrival_time_sequence,
            app_id_sequence
        ))
    ]
    return env

def env_reset(env):
    if 'events' in env:
        del env['events']
    for ITree in env['ITree']:
        ITree.clear()
    for request in env['requests']:
        for activity in request['activities']:
            activity['device_id'] = None
            activity['start_time'] = None
            activity['finish_time'] = INF
        if 'finish_time' in request:
            del request['finish_time']

def env_score(env):
    """ 
    total time (ms) i.e. Makespan, time to complete all workflows
    response time (ms) i.e. Latency, time to complete a workflow
    energy (0.1 uJ)
    ---
    To convert (0.1 uJ) to (J), multiply by 0.0000001
    """
    total_time = max(0 if not ITree else ITree[-1] for ITree in env['ITree'])
    response_time = int(np.mean([request['activities'][-1]['finish_time']-request['arrival_time'] for request in env['requests']]))
    exec_times = [0 if not ITree else sum(b-a for a,b in zip(ITree[::2],ITree[1::2])) for ITree in env['ITree']]
    HN_up_time = max(ITree[-1] if ITree else 0 for ITree in env['ITree'][:env['n_HN']])
    CN_up_time = max(ITree[-1] if ITree else 0 for ITree in env['ITree'][env['n_HN']:])
    GPU_time = sum(a*b for a,b in zip(exec_times[:env['n_gpu']],env['HG']))
    CPU_time = sum(a*b for a,b in zip(exec_times[env['n_gpu']:],env['HC']+env['CC']))
    energy = 21600*HN_up_time + 17600*CN_up_time + 222*GPU_time + 988*CPU_time
    return [total_time,response_time,energy]


def new_TFN(t,rnd):
    """Use 'Triangular Fuzzy Number' to perturb time
    
    ref“Cost-Driven Scheduling for Workflow Decision Making Systems in Fuzzy Edge-Cloud Environments”2025       
        time(t) = workload(w) ÷ processing speed(c)
        processing speed(c) follows TFN(ξl,ξ,ξr)
        δ1=0.75, δ2=1.2
        ξl = randomly selected from the intervals [δ1ξ,ξ]
        ξr = randomly selected from the intervals [ξ,min(δ2ξ,2ξ-ξl)]
    
    Args:
        t (int): Undisturbed computation/communication time
        rnd (List(float,float,float)): 3 random numbers from 0 to 1

    Returns:
        t' (int): Disturbed computation/communication time
    """
    if t==0:
        return 0
    ξ=1/t
    ξl = 0.75*ξ + rnd[0]*0.25*ξ
    ξr = ξ + rnd[1]*min(0.2*ξ,ξ-ξl)
    fc = (ξ - ξl) / (ξr - ξl)
    u = rnd[2]
    if u <= fc:
        c = ξl + (u * (ξr - ξl) * (ξ - ξl)) ** 0.5
    else:
        c = ξr - ((1 - u) * (ξr - ξl) * (ξr - ξ)) ** 0.5
    return int(1/c)

def gen_markov_chain(seed = 0, v = 5, n=6, tail=1.2, entropy=0.9):
    """Construct a Markov chain (random state sequence) with a long tail distribution for the frequency of state occurrence
    
    Args:
        seed      (int): random seed
        v         (int): The length of a Markov chain
        n         (int): Number of Status Types
        tail    (float): Long tail parameter, steeper as larger and smoother as smaller
        entropy (float): Randomness parameter, the larger the diversity, the smaller the monotony

    Returns:
        markov_chain (List[int]): The constructed Markov chain
    """
    rng = np.random.RandomState(seed)
    hist = np.arange(1,n+1)**(-tail)
    pi = hist/hist.sum()
    P = entropy * np.outer(np.ones(n), pi)
    np.fill_diagonal(P, 1 - entropy * (1 - pi))
    cumP = np.cumsum(P, axis=1)
    rand = rng.rand(v)
    markov_chain = [0]
    for i in range(v):
        markov_chain.append(int(np.searchsorted(cumP[markov_chain[-1]], rand[i])))
    return markov_chain[1:]


##########################################################################
# The Fastest Interval Tree Operators in Pure Python
# (used for GET_FREE/SET_BUSY timeslots of a processor)
##########################################################################
from bisect import bisect
def get_interval(after_when,required_duration,ITree):
    """Find a free interval.
    Example:
        ITree = [3,5,10,15] means the Interval [3,5) [10,15) are occupied
           also means the Interval [0,3) [5,10) [15,INF) are not occupied
        get_interval(0,2,ITree) returns tuple (0, 3)
        get_interval(1,3,ITree) returns tuple (5, 10)
        get_interval(5,3,ITree) returns tuple (5, 10)
        get_interval(4,6,ITree) returns tuple (15, INF)
        get_interval(1,2,ITree) returns tuple (1, 3)
        get_interval(4,3,ITree) returns tuple (5, 10)
        get_interval(6,3,ITree) returns tuple (6, 10)
        get_interval(7,3,ITree) returns tuple (7, 10)
    """
    last_end = after_when
    i = bisect(ITree,after_when)
    i-=i&1
    for this_begin,this_end in zip(ITree[i::2],ITree[i+1::2]):
        if last_end >= after_when and this_begin-last_end>=required_duration:
            return last_end,this_begin
        last_end = this_end if this_end > after_when else after_when
    return last_end,INF

def set_interval(begin,end,ITree):
    """Occupy an interval.
    Example:
        ITree = [] means the interval tree is empty.
        set_interval(3,5,ITree)   occupies [3,5)  , ITree becomes [3, 5]
        set_interval(10,15,ITree) occupies [10,15), ITree becomes [3, 5, 10, 15]
        set_interval(5,7,ITree)   occupies [5,7)  , ITree becomes [3, 7, 10, 15]
    """
    if begin >= end or end>=INF:
        raise ValueError(f"Invalid interval [{begin},{end}]")
    if begin < end:
        i = bisect(ITree,begin)
        if i%2==0 and (i==len(ITree) or end<=ITree[i]):
            if i<len(ITree) and end == ITree[i]:
                del ITree[i]
            else:
                ITree.insert(i,end)
            if i>0 and begin == ITree[i-1]:
                del ITree[i-1]
            else:
                ITree.insert(i,begin)
            return
    raise ValueError(f"Interval [{begin},{end}] overlaps {list(zip(ITree[::2],ITree[1::2]))}")
##########################################################################



def get_AST_AFT(env,request_id,task_id,device_id,fuzz=False):
    """get Earliest/Actual Start/Finish Time"""
    request = env['requests'][request_id]
    app_id = request['app_id']
    exec_time = env['ext'][app_id][task_id][device_id]
    if exec_time>=INF:
        return INF, INF
    activities = request['activities']
    if any(activities[pre_task_id]['finish_time']>=INF for pre_task_id in env['pred'][app_id][task_id]):
        return INF, INF
    if fuzz and env['fuzz_net']:
        input_datasizes = [
            new_TFN(input_datasize,rnd_trt)
            for input_datasize,rnd_trt in zip(env['alpha'][app_id][task_id],env['rnd_trt'][request_id][task_id])
        ]
    else:
        input_datasizes = env['alpha'][app_id][task_id]
    EST = max([request['arrival_time']]+[
        activities[pre_task_id]['finish_time'] +
        input_datasize>>env['r'][device_id][activities[pre_task_id]['device_id']] # env['trt']
        # r=3 (8GB/s) or 5(32GB/s), input_datasize>>5 measure in miliseconds
        for input_datasize,pre_task_id in zip(input_datasizes,env['pred'][app_id][task_id])
    ])
    if fuzz and env['fuzz_perf']:
        required_duration = new_TFN(exec_time,env['rnd_ext'][request_id][task_id])
    else:
        required_duration = exec_time
    ITree = env['ITree'][device_id]
    AST = get_interval(EST,required_duration,ITree)[0]
    AFT = AST+required_duration
    return AST,AFT

def set_task_device(env,request_id,task_id,device_id,fuzz=False):
    AST,AFT = get_AST_AFT(env,request_id,task_id,device_id,fuzz)
    _set_activity_device(env,request_id,task_id,device_id,AST,AFT)

def _set_activity_device(env,request_id,task_id,device_id,AST,AFT):
    if AFT<INF:
        activity = env['requests'][request_id]['activities'][task_id]
        ITree = env['ITree'][device_id]
        set_interval(AST,AFT,ITree)
        activity['device_id'] = device_id
        activity['start_time'] = AST
        activity['finish_time'] = AFT


def set_task_device_HEFT(env,request_id,task_id,device_rule='EFT',device_group=None,fuzz=False,place=False):
    """Place the task to the device by EST(Earliest Start Time) or EFT(Earliest Finish Time) strategy"""
    if not device_group:
        device_group = range(env['n_devices'])
    device_id,(AST,AFT) = min(
        ((device_id,get_AST_AFT(env,request_id,task_id,device_id,fuzz))
        for device_id in device_group), 
        key=lambda x:x[1][1 if device_rule=='EFT' else 0]
    )
    if place:
        _set_activity_device(env,request_id,task_id,device_id,AST,AFT)
    return device_id,AST,AFT

def env_prepare_PEFT(env):
    oct_sorts = []
    oct_ranks_all = []
    n_devices = env['n_devices']
    ext=env['ext']
    for app_id in range(env['G']):
        n_tasks = env['N'][app_id]
        # Sort task_ids by OCT rank (PEFT algorithm)
        # “List Scheduling Algorithm for Heterogeneous Systems by an Optimistic Cost Table” 2014 CCF A
        oct=[[0 for _ in range(n_devices)] for _ in range(n_tasks)]
        oct_ranks = [0 for _ in range(n_tasks)]
        for task_id in env['revert_topsort'][app_id]:
            for pk in range(n_devices):
                arr=[0]
                for succ_task_id,output_size in zip(
                    env['succ'][app_id][task_id],
                    env['beta'][app_id][task_id],
                ):
                    arr2=[INF]
                    for pw in range(n_devices):
                        if pk != pw:
                            communication_cost = output_size>>env['r'][pk][pw] # env['trt']
                        else:
                            communication_cost = 0
                        execution_time = ext[app_id][succ_task_id][pw]
                        arr2.append(oct[succ_task_id][pw] + execution_time + communication_cost)
                    arr.append(min(arr2))
                oct[task_id][pk]=max(arr)
            oct_ranks[task_id] = sum(oct[task_id])/n_devices
        oct_sort = sorted(range(n_tasks),key=lambda i:oct_ranks[i],reverse=True)
        oct_sorts.append(oct_sort)
        oct_ranks_all.append(oct_ranks)
    env['oct_sort']=oct_sorts
    env['oct_rank'] = oct_ranks_all

def env_prepare_PPTS(env):
    pcm_sorts = []
    pcm_ranks_all = []
    n_devices = env['n_devices']
    ext=env['ext']
    for app_id in range(env['G']):
        n_tasks = env['N'][app_id]
        # Sort task_ids by PCM rank (PPTS algorithm)
        # “Task Scheduling for Heterogeneous Computing using a Predict Cost Matrix” 2019 CCF B
        pcm=[[0 for _ in range(n_devices)] for _ in range(n_tasks)]
        pcm_ranks = [0 for _ in range(n_tasks)]
        for task_id in env['revert_topsort'][app_id]:
            for pk in range(n_devices):
                #arr=[0]
                arr=[0] if env['succ'][app_id][task_id] else [ext[app_id][task_id][pk]]
                for succ_task_id,output_size in zip(
                    env['succ'][app_id][task_id],
                    env['beta'][app_id][task_id],
                ):
                    arr2=[INF]
                    for pw in range(n_devices):
                        if pk != pw:
                            communication_cost = output_size>>env['r'][pk][pw] # env['trt']
                        else:
                            communication_cost = 0
                        if ext[app_id][succ_task_id][pw]>=INF or ext[app_id][task_id][pw]>=INF or pcm[succ_task_id][pw]>=INF:
                            arr2.append(INF)
                        else:
                            execution_time = ext[app_id][succ_task_id][pw] + ext[app_id][task_id][pw]
                            arr2.append(pcm[succ_task_id][pw] + execution_time + communication_cost)
                    arr.append(min(arr2))
                pcm[task_id][pk]=max(arr)
            #pcm_ranks[task_id] = sum(pcm[task_id])/n_devices
            pcm_ranks[task_id] = np.mean([x for x in pcm[task_id] if x<INF])
        pcm_sort = sorted(range(n_tasks),key=lambda i:pcm_ranks[i],reverse=True)
        pcm_sorts.append(pcm_sort)
        pcm_ranks_all.append(pcm_ranks)
    env['pcm_sort']=pcm_sorts
    env['pcm_rank'] = pcm_ranks_all

def env_prepare_IPPTS(env):
    p_sorts = []
    p_ranks_all = []
    n_devices = env['n_devices']
    ext=env['ext']
    for app_id in range(env['G']):
        n_tasks = env['N'][app_id]
        # Sort task_ids by PCM rank (PPTS algorithm)
        # “Comments on “IPPTS: An Efficient Algorithm for Scientific Workflow Scheduling in Heterogeneous Computing Systems” 2023 CCF A
        pcm=[[0 for _ in range(n_devices)] for _ in range(n_tasks)]
        pcm_ranks = [0 for _ in range(n_tasks)]
        for task_id in env['revert_topsort'][app_id]:
            for pk in range(n_devices):
                #arr=[0]
                arr=[0] if env['succ'][app_id][task_id] else [ext[app_id][task_id][pk]]
                for succ_task_id,output_size in zip(
                    env['succ'][app_id][task_id],
                    env['beta'][app_id][task_id],
                ):
                    arr2=[INF]
                    for pw in range(n_devices):
                        if pk != pw:
                            communication_cost = output_size>>env['r'][pk][pw] # env['trt']
                        else:
                            communication_cost = 0
                        if ext[app_id][succ_task_id][pw]>=INF or ext[app_id][task_id][pw]>=INF or pcm[succ_task_id][pw]>=INF:
                            arr2.append(INF)
                        else:
                            execution_time = ext[app_id][succ_task_id][pw] + ext[app_id][task_id][pw]
                            arr2.append(pcm[succ_task_id][pw] + execution_time + communication_cost)
                    arr.append(min(arr2))
                pcm[task_id][pk]=max(arr)
            #pcm_ranks[task_id] = sum(pcm[task_id])/n_devices
            pcm_ranks[task_id] = np.mean([x for x in pcm[task_id] if x<INF])
        pcm_sort = sorted(range(n_tasks),key=lambda i:pcm_ranks[i],reverse=True)
        # Sort task_ids by P rank (IPPTS algorithm)
        # “Comments on “IPPTS: An Efficient Algorithm for Scientific Workflow Scheduling in Heterogeneous Computing Systems””2023 CCF A
        lhet=[[(x-ext[app_id][i][j]) if ext[app_id][i][j]<=x else INF for j,x in enumerate(arr)] for i,arr in enumerate(pcm)] 
        p_ranks=[pcm_ranks[task_id]*len(env['succ'][app_id][task_id]) for task_id in range(n_tasks)]
        for task_id in env['revert_topsort'][app_id]:
            if env['succ'][app_id][task_id]:
                max_succ_p_rank = max([p_ranks[succ_task_id] for succ_task_id in env['succ'][app_id][task_id]])
                if p_ranks[task_id]<=max_succ_p_rank:
                    p_ranks[task_id] += (max_succ_p_rank-p_ranks[task_id]) + 100
        p_sort = sorted(range(n_tasks),key=lambda i:p_ranks[i],reverse=True)
        p_sorts.append(p_sort)
        p_ranks_all.append(p_ranks)
    env['p_sort']=p_sorts
    env['p_rank'] = p_ranks_all

def env_run_dynamic_planning_HEFT(env,task_rule='upward_sort',device_rule='EFT',device_groups=None,fuzz=True,show=True,opt=None):
    """When a new request arrives, schedules all its tasks, including its not-ready tasks.
    ref:
        HEFT “Performance-effective and low-complexity task scheduling for heterogeneous computing” 2002 CCF A
        PEFT “List Scheduling Algorithm for Heterogeneous Systems by an Optimistic Cost Table” 2014 CCF A
        PPTS “Task Scheduling for Heterogeneous Computing using a Predict Cost Matrix” 2019 CCF B
        IPPTS “Comments on “IPPTS: An Efficient Algorithm for Scientific Workflow Scheduling in Heterogeneous Computing Systems” 2023 CCF A
    Args:
        env (dict): environment
        task_rule (str): 
            upward_sort: Prioritize tasks with HEFT algorithm 1
            downward_sort: Prioritize tasks with HEFT algorithm 2
            oct_sort: Prioritize tasks with PEFT algorithm
            pcm_sort: Prioritize tasks with PPTS algorithm
            p_sort: Prioritize tasks with IPPTS algorithm
        device_rule: 
            EST: Schedule to the device that makes the task start as early as possible.
            EFT: Schedule to the device that makes the task finish as early as possible.
        device_groups (list[list[int]], optional): configuration of resource isolation
            example: [[0,1],[2,3],[4,5]] means three logical isolated resource pools
                     each workflow will be load-balance to one of the pools
                     then schedule using HEFT within that pool.
            default: no resource isolation
        show (bool, optional): print scheduling result. Defaults to True.
        opt (int, optional): Top-k least load strategy for load-balance. default: no top-k, consider all.
    """
    if task_rule not in env:
        if task_rule == 'oct_sort':
            env_prepare_PEFT(env)
        elif task_rule == 'pcm_sort':
            env_prepare_PPTS(env)
        elif task_rule == 'p_sort':
            env_prepare_IPPTS(env)
    if not device_groups:
        device_groups=[list(range(env['n_devices']))]
    for request in env['requests']:
        # save state, then go to "no fluctuation" sandbox to make scheduling decisions
        origin_devices = env['ITree']
        origin_activities = request['activities']
        request_id = request['request_id']
        app_id = request['app_id']
        sorted_task_ids = env[task_rule][app_id]
        if not opt:
            device_group_ids = range(len(device_groups))
        if opt:
            fully_idle_times = [max((origin_devices[device_id][-1] if origin_devices[device_id] else 0) for device_id in device_group) for device_group in device_groups]
            device_group_ids = sorted(range(len(device_groups)),key=lambda device_group_id: fully_idle_times[device_group_id])[:opt]
        best_decision = []
        best_AFT = INF
        for device_group_id in device_group_ids:
            env['ITree'] = [ITree.copy() for ITree in origin_devices]
            request['activities'] = [activity.copy() for activity in origin_activities]
            decision = []
            for task_id in sorted_task_ids:
                device_id,AST,AFT = set_task_device_HEFT(env,request_id,task_id,device_rule,device_groups[device_group_id],fuzz=False,place=True)
                decision.append(device_id)
            if AFT<best_AFT:
                best_AFT = AFT
                best_decision = decision
        # restore state, back to main environment to apply scheduling decisions
        env['ITree'] = origin_devices
        request['activities'] = origin_activities
        #print(f"R{request_id:02d}|W{app_id:02d}|device_group:{device_group_id}|decision:{best_decision}")
        for task_id,device_id in zip(sorted_task_ids,best_decision):
            set_task_device(env,request_id,task_id,device_id,fuzz=fuzz)
        request['finish_time'] = max(activity['finish_time'] for activity in request['activities'])
        if show:
            print(f"R{request_id:02d}|W{app_id:02d}|arrive:{request['arrival_time']},finish:{request['finish_time']}")
    if show:
        print(f"Total_use_time: {max(request['finish_time'] for request in env['requests'])}")


def env_step(env):
    if 'events' not in env:
        env['events'] = []
        for request in env['requests']:
            for task_id,preds in enumerate(env['pred'][request['app_id']]):
                if not preds:
                    heapq.heappush(env['events'],(
                        request['arrival_time'],
                        request['request_id'],
                        request['app_id'],
                        task_id,
                    ))
    if 'set_task_device' not in env:
        def set_task_device(request_id,app_id,task_id,device_id,fuzz=True):
            AST,AFT=get_AST_AFT(env,request_id,task_id,device_id,fuzz)
            _set_activity_device(env,request_id,task_id,device_id,AST,AFT)
            for succ_task_id in env['succ'][app_id][task_id]:
                if all(
                    env['requests'][request_id]['activities'][pred_of_succ_task_id]['finish_time']<INF 
                    for pred_of_succ_task_id in env['pred'][app_id][succ_task_id]
                ):
                    heapq.heappush(env['events'],(
                        AFT,
                        request_id,
                        app_id,
                        succ_task_id,
                    ))
        env['set_task_device']=set_task_device
    if env['events']:
        tick,request_id,app_id,task_id = heapq.heappop(env['events'])
        events = [
            {
                'request_id':request_id,
                'app_id':app_id,
                'task_id':task_id
            }
        ]
        while env['events'] and env['events'][0][0]==tick:
            _,request_id,app_id,task_id = heapq.heappop(env['events'])
            events.append({
                'request_id':request_id,
                'app_id':app_id,
                'task_id':task_id
            })
        return tick, events
    return max(ITree[-1] if ITree else 0 for ITree in env['ITree']), []


def env_run_best_effort_HEFT(env,task_rule=None,task_reverse=False,device_rule='EFT',device_groups=None,fuzz=True,show=True,opt=None):
    """When a new request arrives or a task finish, schedules ready tasks.
    Args:
        env (dict): environment
        device_rule: 
            EST: Schedule to the device that makes the task start as early as possible.
            EFT: Schedule to the device that makes the task finish as early as possible.
        device_groups (list[list[int]], optional): configuration of resource isolation
            example: [[0,1],[2,3],[4,5]] means three logical isolated resource pools
                     each workflow will be load-balance to one of the pools
                     then schedule using HEFT within that pool.
            default: no resource isolation
        show (bool, optional): print scheduling result. Defaults to True.
        opt (int, optional): Top-k least load strategy for load-balance. default: no top-k, consider all.
    """
    if task_rule == None:
        lambda_sort_events = None
    else:
        if task_rule not in env:
            if task_rule == 'oct_rank':
                env_prepare_PEFT(env)
            elif task_rule == 'pcm_rank':
                env_prepare_PPTS(env)
            elif task_rule == 'p_rank':
                env_prepare_IPPTS(env)
            else:
                raise Exception(f"task_rule: {task_rule} not supported, only support upward_rank, downward_rank, upward_layer, downward_layer, oct_rank, pcm_rank, p_rank")
        lambda_sort_events = lambda events: events.sort(key=lambda x:env[task_rule][x['app_id']][x['task_id']],reverse=task_reverse)
    if not device_groups:
        device_groups=[list(range(env['n_devices']))]
    request_device_group={}
    tick, events = env_step(env)
    while events:
        for event in events:
            request_id,app_id,task_id = event['request_id'],event['app_id'],event['task_id']
            if request_id not in request_device_group:
                if len(device_groups)==1:
                    request_device_group[request_id]=device_groups[0]
                else:
                    fully_idle_times = [max((env['ITree'][device_id][-1] if env['ITree'][device_id] else 0) for device_id in device_group) for device_group in device_groups]
                    min_fully_idle_time = min(fully_idle_times)
                    device_group_id = random.choice([i for i,x in enumerate(fully_idle_times) if x == min_fully_idle_time])
                    request_device_group[request_id]=device_groups[device_group_id]
            device_group = request_device_group[request_id]
            device_id,AST,AFT = set_task_device_HEFT(env,request_id,task_id,device_rule,device_group,fuzz=False,place=False)
            env['set_task_device'](request_id,app_id,task_id,device_id,fuzz)
        tick, events = env_step(env)
    if show:
        for request in env['requests']:
            request['finish_time']=request['activities'][-1]['finish_time']
            print(f"R{request['request_id']:02d}|W{request['app_id']:02d}|arrive:{request['arrival_time']},finish:{request['finish_time']}")
        print(f"Total_use_time: {max(request['finish_time'] for request in env['requests'])}")

    

def env_global_task_id(env,request_id,task_id):
    return env['cumsum_tasks'][request_id]+task_id

def env_run_consolidating(env,device_selections,task_priorities,fuzz=True):
    tick,events = env_step(env)
    while events:
        for event in events:
            i = env_global_task_id(env,event['request_id'],event['task_id'])
            event['priority'] = task_priorities[i]
            event['device_id'] = device_selections[i]
        events.sort(key=lambda event:event['priority'])
        for event in events:
            env['set_task_device'](
                event['request_id'],event['app_id'],event['task_id'],event['device_id'],
                fuzz=fuzz
            )
        tick,events = env_step(env)

def env_get_consolidating_decision(env):
    device_selections=[]
    task_start_times = []
    for request in env['requests']:
        for activity in request['activities']:
            device_selections.append(activity['device_id'])
            task_start_times.append(activity['start_time'])
    dim = len(task_start_times)
    task_priorities = [i/dim for i in sorted(range(dim),key=lambda i:task_start_times[i])]
    return device_selections,task_priorities

def env_run_consolidating_NSGA(env,pop_size=500,n_gen=100,show=True):
    """[Prophet] Merge all requests into one large workflow and schedule them at once"""
    n_gpu = env['n_gpu']
    n_devices = env['n_devices']
    dim = env['cumsum_tasks'][-1]
    n_var = 2*dim
    lb = [n_gpu if task_category=='C' else 0 for request in env['requests'] for task_category in env['c'][request['app_id']]] + [0]*dim
    ub = [n_devices-1]*dim + [1]*dim
    import numpy as np
    from pymoo.core.problem import ElementwiseProblem
    class MyProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n_var,   # 2*dim decisions (device_id and task_priority for each tasks)
                            n_obj=3,        # 3 objectives
                            n_ieq_constr=0, # 0 constraints
                            xl=np.array(lb,dtype=np.float16),
                            xu=np.array(ub,dtype=np.float16),
                            vtype=np.float16)
        def _evaluate(self, x, out, *args, **kwargs):
            env_reset(env)
            device_selections = np.round(x[:dim]).astype(int)
            task_priorities = x[dim:]
            env_run_consolidating(env,device_selections,task_priorities,fuzz=False)
            out["F"] = env_score(env)
            out["G"] = []
    
    problem = MyProblem()
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.pcx import PCX
    from pymoo.operators.mutation.pm import PM
    from scipy.stats import qmc
    from pymoo.core.population import Population
    # Manual initialization: Use LatinHypercube to uniformly generate the initial population, and insert a population calculated by HEFT
    env_reset(env)
    env_run_dynamic_planning_HEFT(env,'upward_sort','EFT',None,fuzz=False,show=False)
    device_selections,task_priorities = env_get_consolidating_decision(env)
    known_optimal = [np.array(device_selections+task_priorities,dtype=np.float16)]
    lhs_sampler = qmc.LatinHypercube(d=n_var)
    lhs_samples = lhs_sampler.random(n=pop_size - 1)
    lhs_samples_scaled = qmc.scale(lhs_samples, lb, ub)
    initial_x = np.vstack([lhs_samples_scaled, known_optimal])
    np.random.shuffle(initial_x)
    initial_pop = Population.new("X", initial_x)
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=pop_size//2,
        sampling=initial_pop,
        crossover=PCX(),
        mutation=PM(eta=10),
        eliminate_duplicates=True
    )
    
    from pymoo.termination import get_termination
    termination = get_termination("n_gen", n_gen)
    from pymoo.optimize import minimize
    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=show)
    # TOPSIS sort
    points = res.F
    points = points / np.linalg.norm(points, axis=0, keepdims=True) # L2 normalize
    best_point = np.min(points,axis=0)
    worst_point = np.max(points,axis=0)
    distances_to_best = np.linalg.norm(points-best_point,axis=1)
    distances_to_worst = np.linalg.norm(points-worst_point,axis=1)
    similarities = distances_to_worst/(distances_to_worst+distances_to_best+1e-6)
    indices_desc = np.argsort(similarities)[::-1]
    idx = indices_desc[0]
    # choose the solution with maximum similarities
    env_reset(env)
    device_selections = np.round(res.X[idx][:dim]).astype(int)
    task_priorities = res.X[idx][dim:]
    env_run_consolidating(env,device_selections,task_priorities,fuzz=True)
    return env_score(env)


if __name__=='__main__':
    def demo_drawio():
        env = new_env('G0H2C0W1I5N10F3R0')
        env_run_dynamic_planning_HEFT(env,'upward_sort','EFT',
                                      device_groups=[[2,3,4,5,6,7],[0,1,8,9,10]],
                                      #device_groups=None,
                                      show=True,opt=None)
    def demo_compare_isolation_baseline():
        baselines = []
        isolation = []
        for seed in range(200):
            env = new_env(f'G0H2C0W1I5N60F3R{seed}')
            env_run_dynamic_planning_HEFT(env,'upward_sort','EFT',
                                    device_groups=[[0, 2, 5, 9, 10], [1, 8], [3, 6], [4, 7]],
                                    show=False,opt=None)
            isolation.append(env_score(env))
            env = new_env(f'G0H0C0W1I5N60F3R{seed}')
            env_run_dynamic_planning_HEFT(env,'upward_sort','EFT',
                                      device_groups=None,
                                      show=False,opt=None)
            baselines.append(env_score(env))
        print(f"Isolation:{np.array(isolation).mean(axis=0)}")
        print(f"baselines:{np.array(baselines).mean(axis=0)}")
    
    def demo_NSGA():
        env = new_env('G0H0C0W1I5N10F3R0')
        env_run_consolidating_NSGA(env,pop_size=100,n_gen=80,show=True)
        print(env_score(env))
        
    def demo_heuristic():
        env = new_env('G0H0C0W1I5N10F3R0')
        algorithms = {
            'HEFT1':'upward_sort',
            'HEFT2':'downward_sort',
            'PEFT':'oct_sort',
            'PPTS':'pcm_sort',
            'IPPTS':'p_sort',
        }
        for name,task_rule in algorithms.items():
            env_reset(env)
            env_run_dynamic_planning_HEFT(env,task_rule,'EFT',
                                          device_groups=None,
                                          show=False,opt=None)
            print(f"{name}:{env_score(env)}")
            
        env_reset(env)
        env_run_best_effort_HEFT(env,'EFT',device_groups=None,fuzz=True,show=False)
        name = 'BestEffort'
        print(f"{name}: {env_score(env)}")

    demo_heuristic()
