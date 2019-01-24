import numpy as np
import pandas as pd
import random, string

def main():
    itemN = 50
    tagN = 100
    userN = 50
    coordN = 1000

    itemID = np.random.permutation(np.arange(3*itemN).astype(int))
    topsID = np.sort(itemID[:itemN])
    outerID = np.sort(itemID[itemN:2*itemN])
    pantsID = np.sort(itemID[2*itemN:])
    category = np.zeros(3*itemN).astype(str)
    category[topsID] = 'tops'
    category[outerID] = 'outer'
    category[pantsID] = 'pants'
    tagID = np.arange(tagN).astype(int)
    userID = np.arange(userN).astype(int)
    logN = 150

    item_df = pd.DataFrame({ 'itemID': np.arange(3*itemN).astype(int),
                             'category': category })

    tag_df = pd.DataFrame({ 'tagID': tagID,
                            'tagName': ['tag%03d'%i for i in range(tagN)] })

    user_df = pd.DataFrame({ 'userID': userID,
                             'name': [''.join([random.choice(string.ascii_letters + string.digits) for i in range(10)]) for j in range(userN)] })

    coord_tag = pd.DataFrame({ 'coordID': (np.arange(coordN*5)/5).astype(int),
                               'tagID': np.array([np.random.choice(tagID,5,replace=False) for i in range(coordN)]).flatten() })

    coord_item = pd.DataFrame({ 'coordID': (np.arange(coordN*3)/3).astype(int),
                                'itemID': np.array([np.array([np.random.choice(topsID,1)[0],np.random.choice(outerID,1)[0],np.random.choice(pantsID,1)[0]]) for i in range(coordN)]).flatten() })

    orderLog = pd.DataFrame({ 'orderID': np.arange(logN).astype(int),
                              'userID': np.random.choice(userID,logN,replace=True),
                              'itemID': np.random.choice(itemID,logN,replace=True) })
    
    item_df.to_csv('item_df.csv')
    tag_df.to_csv('tag_df.csv')
    user_df.to_csv('user_df.csv')
    coord_tag.to_csv('coord_tag.csv')
    coord_item.to_csv('coord_item.csv')
    orderLog.to_csv('orderLog.csv')

if __name__=='__main__':
    main()