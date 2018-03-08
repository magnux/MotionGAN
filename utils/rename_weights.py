from __future__ import absolute_import, division, print_function
import h5py as h5
import glob
import re

file_names = glob.glob('*weights.hdf5')
search_str = 'vae'
replace_str = 'fae'
replacer = re.compile(search_str)

for file_name in file_names:
    weight_file = h5.File(file_name, 'r+')

    def find_name(g_name):
        if search_str in g_name:
            return g_name
        return None

    while True:
        to_rename = weight_file.visit(find_name)
        if to_rename is not None:
            new_name = replacer.sub(replace_str, to_rename)
            weight_file.move(to_rename, new_name)
            print(to_rename, '  --->  ', new_name)
        else:
            break

    weight_file.flush()
    weight_file.close()
