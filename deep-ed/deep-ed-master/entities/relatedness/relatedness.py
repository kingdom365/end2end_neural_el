'''
    读取entities_universe.txt，生成wikiid2nnid.txt和nnid2wikiid.txt
'''
import os

data_path = '../../../../data/entities/'
universe_path = data_path + 'entities_universe.txt'
wikiid2nnid_path = data_path + 'wikiid2nnid/wikiid2nnid.txt'
nnid2wikiid_path = data_path + 'nnid2wikiid/nnid2wikiid.txt'

if __name__ == '__main__':
    rltd_all_wikiids = dict()
    with open(universe_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            pair = line.split("\t")
            wiki_id, name = pair[0], pair[1]
            rltd_all_wikiids[wiki_id] = 1 
        
    unlink_wikiid = 1
    rltd_all_wikiids[unlink_wikiid] = 1

    sorted_rltd_all_wikiids = []
    for wiki_id, val in rltd_all_wikiids.items():
        sorted_rltd_all_wikiids.append(int(wiki_id))
    sorted_rltd_all_wikiids = sorted(sorted_rltd_all_wikiids)

    map_sorted_rltd = dict()
    for map_id, wiki_id in enumerate(sorted_rltd_all_wikiids):
        map_sorted_rltd[wiki_id] = map_id + 1

    # wikiid2nnid
    with open(wikiid2nnid_path, "w", encoding='utf-8') as fout:
        for wiki_id, map_id in map_sorted_rltd.items():
            line = "{k}\t{v}\n".format(k=wiki_id, v=map_id)            
            fout.write(line)
    
    with open(nnid2wikiid_path, "w", encoding='utf-8') as fout:
        for map_id, wiki_id in enumerate(sorted_rltd_all_wikiids):
            line = "{k}\t{v}\n".format(k=map_id, v=wiki_id)
            fout.write(line)