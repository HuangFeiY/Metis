import json
import os
import pickle

import random
from pydash.arrays import compact
from automata_tools import NFAtoDFA, DFAtoMinimizedDFA
# from automata_tools import  Automata
from dfa_from_rule import NFAFromRegex
from fsa_to_tensor import Automata, drawGraph
from utils.utils import mkdir, create_datetime_str
import argparse



def create_snort_automata(args):
    # 这是存放转化为json形式的规则文件，key是idx，value是规则（原始规则的content内容并且将其转化为16进制的形式）
    file_path = os.path.dirname(__file__) + "../data/snort/rules/emerging-{}.txt".format(args.dataset_name)
    cat_path = os.path.dirname(__file__) + '../mydata/snort/{}'.format(args.dataset_name)
    dir_path = os.path.dirname(__file__) + '../mydata/snort/{}/automata'.format(args.dataset_name)
    # mode = 'reversed' if reversed else 'split'

    print('GETTING RULE FILES')

    automaton = {}
    all_automata = Automata()
    # 将初始状态设置为0
    all_automata.setstartstate(0)
    state_idx = 1

    with open(file_path, "r") as f:
        rules = json.loads(f.read())

    print('COLLECTING ALL AUTOMATAS')
    mkdir(cat_path)
    mkdir(dir_path)

    contents = []
    # print(keys)
    for key, rule in rules.items():
        # compact 移除假值元素，如None,0,空列表等
        content = " ".join(compact(rule))
        # print(content)
        contents.append(content)
        # # print(concatenatedRule)
        # # print('\n')
        # nfa = NFAFromRegex().buildNFA(concatenatedRule, ruletokens=rule, reversed=reversed)
        # dfa = NFAtoDFA(nfa)
        # minDFA = DFAtoMinimizedDFA(dfa)
        # automaton[key] = minDFA
        # drawGraph(minDFA, os.path.dirname(__file__) + '/data/snort/{}/automata/{}'.format(args.dataset_name, key))
    # print(contents)
    concatenatedRule = f' ( {" ) | ( ".join(compact(contents))} ) '
    concatenatedRule = concatenatedRule.split()
    # print(concatenatedRule)
    nfa = NFAFromRegex().buildNFA(concatenatedRule, ruletokens=concatenatedRule, reversed=reversed)
    dfa = NFAtoDFA(nfa)
    minDFA = DFAtoMinimizedDFA(dfa)
    automaton[1] = minDFA
    drawGraph(minDFA, os.path.dirname(__file__) + '../mydata/snort/{}/automata/{}'.format(args.dataset_name,
                                                                                      'all_content_automata'))

    print('MERGING AUTOMATA')
    for label, automata in automaton.items():
        tok = 'BOS'
        # if reversed:
        #     tok = 'EOS'
        all_automata.addtransition(0, state_idx, tok)  # may cause bug when the RE is accosiate with BOS
        states = list(automata.states)
        final_states = list(automata.finalstates)
        num_states = len(states)
        used_states = [i for i in range(state_idx, state_idx + num_states)]
        states2idx = {states[i]: used_states[i] for i in range(num_states)}

        for fr_state, to in automata.transitions.items():
            for to_state, to_edges in to.items():
                for edge in to_edges:
                    all_automata.addtransition(states2idx[fr_state], states2idx[to_state], edge)

        all_automata.addfinalstates([states2idx[i] for i in final_states])
        all_automata.addfinalstates_label([states2idx[i] for i in final_states], label)
        state_idx += (num_states)
        # all_automata = DFAtoMinimizedDFA(all_automata)
        # drawGraph(all_automata, label)

    # all_automata = DFAtoMinimizedDFA(all_automata)
    merged_automata = all_automata.to_dict()
    # time_str = create_datetime_str()

    path = os.path.dirname(__file__) + '../mydata/snort/{}/automata'.format(args.dataset_name)
    print("Drawing Graph and save at: {}".format(path))
    drawGraph(all_automata,
              os.path.dirname(__file__) + '../mydata/snort/{}/automata/{}'.format(args.dataset_name, 'merged_automata'))
    path = os.path.dirname(__file__) + '../mydata/snort/{}/automata/{}.pkl'.format(args.dataset_name, args.automata_name)
    print("Save the automata object at: {}".format(path))
    pickle.dump(merged_automata, open(path, 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='snort', help="category name")
    parser.add_argument('--automata_name', type=str, default='all', help="automata name prefix")

    args = parser.parse_args()
    create_snort_automata(args)
