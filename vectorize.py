import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.decomposition import PCA

def main():
    item_df = pd.read_csv('item_df.csv')
    tag_df = pd.read_csv('tag_df.csv')
    user_df = pd.read_csv('user_df.csv')
    coord_tag = pd.read_csv('coord_tag.csv')
    coord_item = pd.read_csv('coord_item.csv')
    orderLog = pd.read_csv('orderLog.csv')
    
    itemN = item_df.shape[0]
    tagN = tag_df.shape[0]
    userN = user_df.shape[0]
    
    cooccurrence = []
    coordIDs = coord_tag['coordID'].unique()
    for i,ID in enumerate(coordIDs):
        df = coord_tag[coord_tag['coordID']==ID]
        tmp = []
        for tag in df['tagID']:
            tmp.append(str(tag))
        cooccurrence.append(tmp)

    model = word2vec.Word2Vec(cooccurrence, size=100, min_count=5, window=5)
    
    tagVec = np.array([np.array(model.wv[str(t)]) for t in range(tagN)])
    pca = PCA()
    pca.fit(tagVec)
    tagVecPCA = pca.fit_transform(tagVec)

    R = 0
    features = 0
    for rate in pca.explained_variance_ratio_:
        R += rate
        features += 1
        if R>0.9:
            break
    tagVec = tagVec[:,:features]
    tagVec = tagVec/(np.linalg.norm(tagVec,axis=1))[:,np.newaxis]
    
    itemVec = np.zeros((3*itemN,features))
    for item_id in range(3*itemN):
        coordIDs = coord_item['coordID'][coord_item['itemID']==item_id].values
        tags = []
        for ID in coordIDs:
            tags.extend(coord_tag['tagID'][coord_tag['coordID']==ID].values)
        itemVec[item_id,:] = np.sum(tagVec[tags,:],axis=0)
    idx = np.where(np.sum(np.abs(itemVec),axis=1)>0)[0]
    itemVec[idx,:] = itemVec[idx,:]/(np.linalg.norm(itemVec[idx,:],axis=1))[:,np.newaxis]
    
    userVec = np.zeros((userN,features))
    for user_id in range(userN):
        items = orderLog['itemID'][orderLog['userID']==user_id].values
        if len(items)>0:
            tmp = itemVec[items,:]
            userVec[user_id,:] = np.sum(tmp,axis=0)
    idx = np.where(np.sum(np.abs(userVec),axis=1)>0)[0]
    userVec[idx,:] = userVec[idx,:]/(np.linalg.norm(userVec[idx,:],axis=1))[:,np.newaxis]
    
    np.save('tagVec.npy',tagVec)
    np.save('itemVec.npy',itemVec)
    np.save('userVec.npy',userVec)
    
if __name__=='__main__':
    main()