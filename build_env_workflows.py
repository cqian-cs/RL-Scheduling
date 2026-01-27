import time
num2time = lambda t=None,f="%Y%m%d-%H%M%S": time.strftime(f, time.localtime(int(t if t else time.time())))
time2num = lambda t=None,f="%Y%m%d-%H%M%S": int(time.mktime(time.strptime(t, f)) if t else time.time())
import os
path_this = os.path.dirname(os.path.abspath(__file__)).replace('\\','/')
name_this = os.path.splitext(os.path.split(__file__)[1])[0]
import json
def json_load(path_file,default=None):
    try:
        with open(path_file,'r',encoding='utf-8') as f:
            return json.loads(f.read())
    except:
        return default
def json_save(path_file,obj):
    folder = os.path.dirname(os.path.abspath(path_file))
    os.makedirs(folder,exist_ok=True)
    with open(path_file,'w',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,indent=None))
import random

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


def topsort(G):
    count = {u:sum([u in v for v in G.values()]) for u in G}
    Q = [u for u in G if count[u] == 0]
    S = []
    while Q:
        u = Q.pop()
        S.insert(0,u)
        for v in G[u]:
            count[v]-=1
            if count[v]==0:
                Q.append(v)
    return S
def get_ES_LS(workloads,edges):
    n_tasks = len(workloads)
    t_preds =[[] for _ in range(n_tasks)]
    t_succs =[[] for _ in range(n_tasks)]
    ES = [0 for _ in range(n_tasks)]
    for src,dst in edges:
        t_preds[dst].append(src)
        t_succs[src].append(dst)
    q=[0]
    while q:
        task_id = q.pop(0)
        for succ_task_id in t_succs[task_id]:
            ES[succ_task_id]=max(ES[succ_task_id],ES[task_id]+workloads[task_id])
            q.append(succ_task_id)
    LS=[ES[-1] for i in range(n_tasks)]
    q=[n_tasks-1]
    while q:
        task_id = q.pop(0)
        for pred_task_id in t_preds[task_id]:
            LS[pred_task_id]=min(LS[pred_task_id],LS[task_id]-workloads[pred_task_id])
            q.append(pred_task_id)
    return ES,LS

def build_workflows():
    # Scale workloads and datasizes
    # 1. workload from 0.2TFLOP to 1TFLOP, fuzzify with TFN
    #   the tasks less than 0.2TFlops are CPU only
    #   the tasks between 0.2TFlops and 0.4TFlops are CPU or GPU
    #   the tasks more than 0.4TFlops are GPU only
    # 2. datasize from 30MB to 3000MB, fuzzify with TFN
    # 3. task['workload'] in miliseconds, measured by x1
    #    x6: 16 vCPU 1C = 1/3 TFLOPS ext=workload*6
    #    x3: 32 vCPU 2C = 2/3 TFLOPS ext=workload*3
    #    x2: 48 vCPU 3C = 3/3 TFLOPS ext=workload*2
    #    x2: 1/9vGPU 1T = 1 TFLOPS   ext=workload*2
    #    x1: 2/9vGPU 2T = 2 TFLOPS   ext=workload*1
    #    48 vCPU = 1 TFLOPS is because modern CPU's SIMD feature,
    #      where 1 instrument operates over 8 floats
    #      2.5GHz×48 = 120 G instruments/s ≈ 1TFlops
    # 4. dataflow['datasize'] in MB
    #    inner machine communication: PCIe, 32 GB/s, trt=datasize>>5 ms
    #    inter machine communication: Ethernet, 8 GB/s, trt=datasize>>3 ms
    # 5. dataset and workload reference:
    #    Alibaba cluster trace program. [Online]. Available: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
    #    <A Deep Reinforcement Learning with Transformer Integration for Directed Acyclic Graph Scheduling in Edge Networks>
    ali2018 = json_load(f'{path_this}/workflows_ali2018_origin.json')
    for w in ali2018:
        seed = sum(bytearray(w['name'], 'utf-8'))
        rnd_gen = random.Random(seed)
        for t in w['tasks']:
            t['workload'] = rnd_gen.randint(20,100) # 20ms~100ms in the fastest device, 0.04~0.2 FLOP
            if t['workload']<=40: # 0.08 FLOP
                t['category'] = 'C'
            else:
                t['category'] = 'G'
        for df in w['dataflows']:
            # inner: 0~6ms
            # inter: 1~25ms
            df['datasize'] = rnd_gen.randint(10,200) # 10~200 MB
    for i_env,workflows in [
        [0,[ali2018[1]]],
        [1,ali2018[:4]],
        [2,ali2018[4:]]
    ]:
        for w in workflows:
            seed = sum(bytearray(w['name'], 'utf-8'))
            rnd_gen = random.Random(seed)
            for t in w['tasks']:
                rnd = [rnd_gen.random() for _ in range(3)]
                t['workload'] = new_TFN(t['workload'],rnd)
                t['preds'] = []
                t['succs'] = []
                t['input_size'] = []
                t['output_size'] = []
            for df in w['dataflows']:
                rnd = [rnd_gen.random() for _ in range(3)]
                df['datasize'] = new_TFN(df['datasize'],rnd)
                w['tasks'][df['dst']]['preds'].append(df['src'])
                w['tasks'][df['src']]['succs'].append(df['dst'])
                w['tasks'][df['dst']]['input_size'].append(df['datasize'])
                w['tasks'][df['src']]['output_size'].append(df['datasize'])
            # Topsort task_ids from root to sink, and reverse topsort from sink to root
            G={i:[] for i in range(len(w['tasks']))}
            RG={i:[] for i in range(len(w['tasks']))}
            for d in w['dataflows']:
                G[d['dst']].append(d['src'])
                RG[d['src']].append(d['dst'])
            w['topsort'] = topsort(G)
            w['revert_topsort'] = topsort(RG)
            # upward rank for HEFT algorithm (baseline)
            # <Performance-effective and low-complexity task scheduling for heterogeneous computing>
            upward_ranks = [0]*len(w['tasks'])
            for task_id in w['revert_topsort']:
                arr=[0]
                for succ_task_id,output_size in zip(
                    w['tasks'][task_id]['succs'],
                    w['tasks'][task_id]['output_size'],
                ):
                    arr.append(upward_ranks[succ_task_id] + output_size)
                upward_ranks[task_id] = w['tasks'][task_id]['workload'] + max(arr)
            upward_sort = sorted(range(len(w['tasks'])),key=lambda i:upward_ranks[i],reverse=True)
            w['upward_ranks'] = upward_ranks
            w['upward_sort'] = upward_sort
            # Sort task_ids by downward rank
            downward_ranks = [0]*len(w['tasks'])
            for task_id in w['topsort']:
                arr=[0]
                for pred_task_id,input_size in zip(
                    w['tasks'][task_id]['preds'],
                    w['tasks'][task_id]['input_size'],
                ):
                    arr.append(downward_ranks[pred_task_id] + w['tasks'][pred_task_id]['workload'] + input_size)
                downward_ranks[task_id] = max(arr)
            downward_sort = sorted(range(len(w['tasks'])),key=lambda i:downward_ranks[i])
            w['downward_sort'] = downward_sort
            w['downward_ranks'] = downward_ranks
            # get Earliest Start and Latest Start Time
            ES,LS = get_ES_LS([task['workload'] for task in w['tasks']],[(d['src'],d['dst']) for d in w['dataflows']])
            w['is_critical_path'] = [1 if es==ls else 0 for es,ls in zip(ES,LS)]
            # get task layer
            FL=[0]*len(w['tasks'])
            BL=[0]*len(w['tasks'])
            for task_id in w['topsort']:
                FL[task_id] = max([FL[task_id]]+[FL[pred_task_id]+1 for pred_task_id in w['tasks'][task_id]['preds']])
            for task_id in w['revert_topsort']:
                BL[task_id] = max([BL[task_id]]+[BL[succ_task_id]+1 for succ_task_id in w['tasks'][task_id]['succs']])
            w['forward_layer'] = FL
            w['backward_layer'] = BL
            #if all(a<b for a,b in zip(w['topsort'][::2],w['topsort'][1::2])):
            #    print(f"ok Workflow {w['name']} tasi_id is topsorted")
            #else:
            #    print(f"   Workflow {w['name']} task_id is not topsorted")
        env_workflows={
            'names': [w['name'] for w in workflows],
            'G': len(workflows),
            'N': [len(w['tasks']) for w in workflows],
            'pred':[[task['preds'] for task in w['tasks']] for w in workflows],
            'succ':[[task['succs'] for task in w['tasks']] for w in workflows],
            'c':[[task['category'] for task in w['tasks']] for w in workflows],
            'w':[[task['workload'] for task in w['tasks']] for w in workflows],
            'pw':[[sum(w['tasks'][t]['workload'] for t in task['preds']) for task in w['tasks']] for w in workflows],
            'alpha': [[task['input_size'] for task in w['tasks']] for w in workflows],
            'beta': [[task['output_size'] for task in w['tasks']] for w in workflows],
            'topsort': [w['topsort'] for w in workflows],
            'revert_topsort': [w['revert_topsort'] for w in workflows],
            'upward_rank': [w['upward_ranks'] for w in workflows],
            'upward_sort': [w['upward_sort'] for w in workflows],
            'downward_rank': [w['downward_ranks'] for w in workflows],
            'downward_sort': [w['downward_sort'] for w in workflows],
            'is_critical_path': [w['is_critical_path'] for w in workflows],
            'forward_layer': [w['forward_layer'] for w in workflows],
            'backward_layer': [w['backward_layer'] for w in workflows],
        }
        json_save(f'{path_this}/env_workflows_{i_env}.json',env_workflows)

if __name__ == '__main__':
    build_workflows()