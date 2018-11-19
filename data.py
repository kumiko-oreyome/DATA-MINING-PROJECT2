import random
from textutil import write_json_utf8,write_csv
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
class Substance():
    VOLUME_RANGE = (100,500)
    MASS_RANGE = (50,200)
    LENGTH_RANGE = (150,200)
    SHAPE_ENUM =  ('SQUA','REC','TRI')
    COLOR_ENUM =  ('R','G','Y','B')
    def __init__(self,volume,mass,length,shape,color,is_rough,is_light):
        self.volume = volume
        self.mass = mass
        self.length = length
        self.shape = shape
        self.color = color
        self.is_rough = is_rough
        self.is_light = is_light

    def features(self):
        return (self.volume,self.mass,self.length,self.shape,self.color,self.is_rough)
    

    @staticmethod
    def generate_sample(rule):
        def sample_from_range(lower,upper):
            return random.randint(lower,upper)
        def random_select(items):
            return random.choice(items)
        volume,mass,length,shape,color,is_rough = sample_from_range(*Substance.VOLUME_RANGE),\
                                         sample_from_range(*Substance.MASS_RANGE),\
                                         sample_from_range(*Substance.LENGTH_RANGE),\
                                         random_select(Substance.SHAPE_ENUM),\
                                         random_select(Substance.COLOR_ENUM),\
                                         sample_from_range(0,1)
        label =  rule.label_of(volume,mass,length,shape,color,is_rough)
        return Substance(volume,mass,length,shape,color,is_rough,label)

class LabelRule():
    def __init__(self):
        pass
    def label_of(self,volume,mass,length,shape,color,is_rough):
        if mass/volume > 0.5 and length >170:
            return 1
        if length-mass > 60 and  volume >200:
            return 1
        if length-mass > 60 and color in ('R','G'):
            return 1
        if color in ('R','Y') and not is_rough:
            return 1
        if  color !='B' and  shape in ('REC','TRI'):
            return 1
        return 0


class LabelRule2():
    def __init__(self):
        pass
    def label_of(self,volume,mass,length,shape,color,is_rough):
        if mass+volume-length-is_rough*100 > 200:
            return 1
        return 0

class LabelRule3():
    def __init__(self):
        pass
    def label_of(self,volume,mass,length,shape,color,is_rough):
        if mass/volume > 0.5 and length >170:
            return 1
        return 0

def generate_dataset(pos_num,neg_num,rule_name,path=None):
    def generate_examples(num,label):
        examples = []
        cnt = 0
        while cnt < num:
            example = Substance.generate_sample(rule)
            if label == example.is_light:
                cnt+=1
                examples.append(example)
        return examples
    if rule_name == 'rule1':
        rule = LabelRule()
    elif rule_name == 'rule2':
        rule = LabelRule2()
    elif rule_name == 'rule3':
        rule =LabelRule3()
    else:
        print('rule name error')
        assert False
    
    examples =  generate_examples(pos_num,1)+generate_examples(neg_num,0)
    if path is not None:
        header = ["volume","mass","length","shape","color","rough","light"]
        write_csv(path,header,[ example.features()+(example.is_light,) for example in examples]) 
    return examples
 


def read_dataset(path):
    df = pd.read_csv(path)
    df['shape'] = df['shape'].apply(lambda x:Substance.SHAPE_ENUM.index(x))
    df['color'] = df['color'].apply(lambda x:Substance.COLOR_ENUM.index(x))
    return df

def get_Xy(path):
    df = read_dataset(path)
    X,y = df.iloc[:, :-1].values,df.iloc[:, -1].values
    return X,y

def evaluate( X,y_true,model):
    y_pred = model.predict(X)
    print('accuracy')
    print(accuracy_score(y_true, y_pred))

#def classfication(clf,X,y):
    

#def forest_classification(train_path,test_path):
#    X,y = get_Xy(train_path)
#    print('random forest')
#    clf = RandomForestClassifier(n_estimators=100)
#    clf.fit(X,y)
#    evaluate(test_path,clf)
#    print(clf.feature_importances_)
#
#def tree_classification(train_path,test_path):
#    import pydotplus
#    X,y = get_Xy(train_path)
#    print('tree')
#    clf = DecisionTreeClassifier()
#    clf.fit(X,y)
#    evaluate(test_path,clf)
#    print(clf.feature_importances_)




#TRAIN_FILE = './rule3/train_100_100.csv'
#TEST_FILE = './rule3/test_100_100.csv'
#
#generate_dataset(100,100,LabelRule3(),TRAIN_FILE)
#generate_dataset(100,100,LabelRule3(),TEST_FILE)
#main_forest(TRAIN_FILE,TEST_FILE)
#main_tree(TRAIN_FILE,TEST_FILE)