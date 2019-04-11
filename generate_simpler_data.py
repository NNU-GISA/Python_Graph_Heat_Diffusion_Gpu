import numpy as np
#generate the data
#that is the nbrs and the wgts
ngbrs = np.random.randint(5, size=(15,1))
wgts = np.random.randint(100,size=(15,1))
texture = np.random.randint(256,size=(5,1))

ngbrs = ngbrs.astype('int32')
wgts = wgts.astype('float32')
texture = texture.astype('float32')

np.savez("./data.npz", weights=wgts, ngbrs=ngbrs)
np.savez("./gray.npz", texture=texture)
