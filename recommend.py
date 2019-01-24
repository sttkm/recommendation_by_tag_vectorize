import numpy as np
import sys
import pandas as pd

def main():
    user_id = int(sys.argv[1])
    N = 5
    if len(sys.argv)>1:
        N = int(sys.argv[2])
        
    itemVec = np.load('itemVec.npy')
    userVec = np.load('userVec.npy')
    orderLog = pd.read_csv('orderLog.csv')
    itemN = itemVec.shape[0]
    user_item = list(set(range(itemN))-set(orderLog['itemID'][orderLog['userID']==user_id].values))
    itemVec_ = itemVec[user_item,:]
    similarity = np.sum(itemVec_*userVec[user_id,:],axis=1)
    ranking = np.argsort(similarity)[::-1]
    print('Recommend %d itemIds for user%02d: '%(N,user_id)+', '.join(['%03d'%user_item[ranking[i]] for i in range(N)]))
    
if __name__=='__main__':
    main()