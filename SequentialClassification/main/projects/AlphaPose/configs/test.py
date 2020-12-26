import os, glob

users = os.listdir(os.getcwd())

for user in users:
    if os.path.isdir(user):
        
        user = os.path.join(os.getcwd(), user)
        result = [y for x in os.walk(user) for y in glob.glob(os.path.join(x[0], '*.json'))]
        for res in result:
            name = '.'.join(res.split('/')[-4:-1]) + '.json'
            d = res.split('/')[:-1]
            d.append('alphapose_' + name)
            name = '/' + os.path.join(*d)
            print(res)
            print(name)
            os.rename(res, name)