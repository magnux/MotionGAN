from __future__ import absolute_import, division, print_function

import json

def flatten(l):
    for el in l:
        if isinstance(el, list):
            for sub in flatten(el):
                yield sub
        else:
            yield el

groups = ['head', 'torso', 'l-arm', 'r-arm', 'l-leg', 'r-leg']

with open("../data/dataset_motion_primitives.json", "r") as read_file:
    data = json.load(read_file)
    # print(data.keys())
    for seq in data.values():
        flat_seq = [item for item in flatten(seq)]  # What a mess, list had to be flattened from arbitrary depths
        # print(flat_seq)
        for item in flat_seq:
            if 'time' in item:  # What a mess 2: the key alternates between milliseconds and time
                item['milliseconds'] = item['time']
                del item['time']

        end_millis = [item['milliseconds'][1] for item in flat_seq]
        # print(max(end_millis))

        for item in flat_seq:
            if ':' not in item['label']:
                print(item['label'])  # What a mess 3: labels not always splitted by :, also a few labels are in a complete different format ...
            # group, primitive = item['label'].split(':')
            #
            # if group not in groups:
            #     print(group, primitive)

        # for g in groups:
        #
        #
        # for item in sorted(flat_seq, key=lambda x: x['milliseconds']):
        #      print(item['label'], item['milliseconds'])
