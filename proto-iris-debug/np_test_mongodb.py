#%%
from biolab import IrisDB
#%%
db=IrisDB(ds_id='iris_syn',db_ip='10.171.18.104')
db=IrisDB('iris_syn')
#%%
docs = list(db.find({},{'fv':1,'eye_id':1}))

#%%
print(docs[0])
print(len(docs))
# %%
# doc = db.find_one({})
# doc
#%%
# doc['eye_id']
#%%
# db.avail_ds

#%%
# len(docs)

#%%
import numpy as np
test=docs
fv_code=[]
labels=[]
# np.frombuffer(bitstr.encode('ascii'), dtype=np.uint8) - ord('0')
for d in test:
    labels.append(d['eye_id'])
    fv_code.append(np.frombuffer(d['fv'].encode('ascii'), dtype=np.uint8).reshape((8,256)) - ord('0'))
fv_code=np.stack(fv_code)
print(len(fv_code))
#%%
# from pprint import pprint
# pprint(feat_codes[0])
# labels
#%%
import h5py
with h5py.File('test.h5','w') as f:
    f.create_dataset("fv", data=fv_code, compression="gzip")
    f.create_dataset("labels", data=np.array(labels, dtype="S"))

#%%
# fv_code.dtype