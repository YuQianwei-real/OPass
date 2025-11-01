import json, os, glob
from tvm import relay

from Autotuning.util import simu_mem_from_relay, serenity_mem_from_relay
#from GenCoG_cl.gencog.graph.relay import build_graph

approach = ['Origin', 'Default', 'Greedy', 'Beam', 'SA', 'Transfer']

serenity25 = {'beam': [0.01129150390625], 'default': [0.016326904296875], 'sa': [0.01129150390625], 'greedy': [0.011293411254882812], 'transfer_graph': [0.01129150390625]}
serenity27 = {}


TVM = {'Origin': [], 'Default': [], 'Greedy': [] , 'Beam':[], 'SA':[], 'Transfer':[]}
Serenity = {'Origin': [], 'Default': [], 'Greedy': [] , 'Beam':[], 'SA':[], 'Transfer':[]}
Hmcos = {'Origin': [], 'Default': [], 'Greedy': [] , 'Beam':[], 'SA':[], 'Transfer':[]}

def collect_json_data():
    
    for i in range(1, 37):
        f_1 = f'./ReBench/{i}/tune_results.json'
        f_2 = f'./ReBench/{i}/tune_results_serenity.json'
        f_3 = f'./ReBench/{i}/tune_results_hmcos.json'
        dic_1 = json.load(open(f_1))
        dic_2 = json.load(open(f_2))
        dic_3 = json.load(open(f_3))
        for app in approach:
            TVM[app].append(dic_1[app][0])
            Serenity[app].append(dic_2[app][0])
            Hmcos[app].append(dic_3[app][0])

    return 

def eval_failed_results(folder_path: str):
    
    lowest_mem = float('inf')

    for filename in os.listdir(folder_path):
    # 检查是否为txt文件
        if filename.endswith('.txt'):
        # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
    
            with open(file_path, 'r', encoding='utf-8') as file:
                mod =  relay.parse(file.read())
                static_mod = relay.transform.DynamicToStatic()(relay.transform.InferType()(mod))
                
                mem = serenity_mem_from_relay(static_mod)
                if mem < lowest_mem:
                    lowest_mem = mem

    return lowest_mem


if __name__ == '__main__':
    Origin_percent, Default_percent, Greedy_percent, Beam_percent, SA_percent, Transfer_percent = 0, 0, 0, 0, 0, 0
    for i in range(1, 37):
        if i == 24 or i ==25 or i == 27:
            continue
    selected =  [2,5, 9, 10, 14, 15, 18, 19, 21, 22, 30, 32]#, 6, 7, 23, 26, 29, 31, 34, 35, 36]
    #for i in selected:
    for i in range(1, 37):
        if i== 25 or i == 24 or i == 27:
            continue
        source = json.load(open(f'./ReBench/{i}/tune_results.json'))
        source_Origin = source['Origin'][0]
        native = json.load(open(f'./ReBench/{i}/tune_results_hmcos.json'))
        
        current_Origin = native['Origin'][0]
        Origin_percent += (current_Origin - current_Origin) / current_Origin 

        current_Default = native['Default'][0] 
        Default_percent += (current_Origin - current_Default) /current_Origin

        current_Greedy = native['Greedy'][0]
        Greedy_percent += (current_Origin - current_Greedy) /current_Origin

        current_Beam = native['Beam'][0]
        Beam_percent += (current_Origin - current_Beam) /current_Origin

        current_SA = native['SA'][0]
        SA_percent += (current_Origin - current_SA) /current_Origin
        
        current_Transfer = native['Transfer'][0]
        Transfer_percent += (current_Origin - current_Transfer) /current_Origin

    print('Origin_percent=', Origin_percent *100/33)#len(selected), '%')
    print('Default_percent=', Default_percent *100/33)#len(selected), '%')
    print('Greedy_percent=',Greedy_percent *100/33)#len(selected), '%')
    print('Beam_percent=',Beam_percent *100/33)#len(selected), '%')
    print('SA_percent =', SA_percent *100/33)#len(selected), '%')
    print('Transfer_percent= ',Transfer_percent *100/33)#len(selected), '%')