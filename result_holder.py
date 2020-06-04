'''
Saving and ordering data in a human readable way. [not json format, just str]

Keys should be sortable. e.g. tuple
Every element in keys/values will be converted to str for sorting
'''
import os

class holder(): #for loss, accuracy ...
    def __init__(self, keys_name:list, values_name:list, save_dir=None, modes=["training","validation","testing","debug"], overwrite_data=True, overwrite_file=True):
        """
        All name in keys_name, values name should be str, or any type that can be converted to string
        """
        assert type(keys_name) in [tuple, list] and type(values_name) in [tuple, list], "Bad input types, use tuple or list."
        self.keys_name = keys_name
        self.values_name = values_name
        self.save_dir = save_dir
        self.overwrite_data = overwrite_data
        self.overwrite_file = overwrite_file
        self.history = {}
        for mode in modes:
            self.history[mode]={}

    def check_length(self, x):
        try:
            length = len(x)
        except:
            length = 1
        return length
    
    def push(self, mode, key, value):
        key_len = self.check_length(key)
        value_len = self.check_length(value)
        key = tuple([key]) if key_len==1 else key
        value = tuple([value]) if value_len==1 else value
        #key = tuple(map(str, key))
        #value = tuple(map(str, value))
        assert key_len == len(self.keys_name) and value_len == len(self.values_name), "Input not matched with names."
        if not self.overwrite_data and key in self.history[mode]:
            raise TypeError("Trying overwrite in mode={}, key={}, prev_value={}, new_value={}".format(mode, key, self.history[mode][key], value))

        self.history[mode][key]=value

    def pop(self, mode, key):
        self.history[mode].pop(key)
        
    def save(self, save_dir=None):
        save_dir = self.save_dir if save_dir==None else save_dir
        assert save_dir != None, "Please state your saving path..."
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for mode in self.history:
            if self.history[mode] != {}: #if not empty
                outf = save_dir+'/'+mode+".log"
                if not os.path.exists(outf) or self.overwrite_file:
                    with open(save_dir+'/'+mode+".log", "w") as f:
                        #call writer
                        out = self.writer(mode)
                        f.write(out)
                        
                        
    def writer(self, mode):
        '''
        Return a beautiful(?) output format
        For each project, this function should be editted if needed
        '''
        data = self.history[mode]
        out=""
        for key in sorted(data):
            line="\n"
            for i in range(len(self.keys_name)):
                line += "{}: {}  ".format(self.keys_name[i], key[i])
            line += "||"
            for j in range(len(self.values_name)):
                line += "  {}: {}".format(self.values_name[j], data[key][j])
            out += line
        return out
                
        
    
