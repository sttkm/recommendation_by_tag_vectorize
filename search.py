import numpy as np
import sys
import pandas as pd

def main():
    search_tag = sys.argv[1]
    N = 30
    if len(sys.argv)>1:
        N = int(sys.argv[2])
    
    tagVec = np.load('tagVec.npy')
    itemVec = np.load('itemVec.npy')
    tag_df = pd.read_csv('tag_df.csv')
    tag_id = tag_df['tagID'][tag_df['tagName']==search_tag].values[0]
    tag_vec = tagVec[tag_id,:]
    similarity = np.sum(itemVec*tag_vec,axis=1)
    ranking = np.argsort(similarity)[::-1]
    print('Search 30 results of %s: '%search_tag+', '.join(['%03d'%ranking[i] for i in range(N)]))
    
if __name__=='__main__':
    main()