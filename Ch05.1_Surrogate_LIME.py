from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:])
              for x in newsgroups_train.target_names]

print(class_names)

class_name[3] = 'pc.hardware'
class_name[4] = 'mac.hardware'

print(class_name)