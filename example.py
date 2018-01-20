import pandas as pd
from my_decision_tree import MyDecisionTree


data = pd.DataFrame({
    'target':[1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    'windy':[1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    'temperature':['cool', 'hot', 'hot', 'mild', 'cool', 'mild', 'cool', 'mild',
                   'mild', 'hot', 'hot', 'mild', 'cool', 'mild', 'mild'],
    'humidity':['normal', 'high', 'normal', 'high', 'normal', 'high', 'normal',
                'high', 'normal', 'high', 'high', 'high', 'normal', 'normal',
                'high'],
    'outlook':['overcast', 'overcast', 'overcast', 'overcast', 'rainy', 'rainy',
               'rainy', 'rainy', 'rainy', 'sunny', 'sunny', 'sunny', 'sunny',
               'sunny', 'overcast']
})

target = data['target']
del data['target']
clf = MyDecisionTree(min_samples_split=2)
clf.fit(data, target)
predicts = clf.predict(data)
print(predicts == target)
