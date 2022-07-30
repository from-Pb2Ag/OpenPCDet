import pickle 
import numpy as np

path = '/home/barza/OpenPCDet/data/dense/dense_infos_val_clear_15.pkl'
new_path = '/home/barza/OpenPCDet/data/dense/dense_infos_val_clear_short.pkl'

with open(path, 'rb') as f:
    infos = pickle.load(f)

SIZE = 20 #len(infos) * 0.05
idx = np.random.choice(len(infos), size=SIZE, replace=False)
new_infos = np.array(infos)[idx]
new_infos = new_infos.tolist()

with open(new_path, 'wb') as f:
    pickle.dump(new_infos, f)
