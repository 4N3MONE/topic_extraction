PATH = './private/key.properties'

def get_key(query):
    with open(PATH, 'r') as f:
        d = {}
        for line in f.readlines():
            [k, v] = line.split('=')
            d[k] = v
    if(query not in d.keys()):
        return "ERROR: invalid query!"
    return d[query]