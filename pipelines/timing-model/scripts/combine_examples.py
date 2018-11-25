import glob
import pickle

examples = []

for filename in glob.glob('examples/*.p'):
    e = pickle.load(open(filename, 'rb'))
    examples.extend(e)

print('Processed a total of {} examples'.format(len(examples)))

pickle.dump(examples, open('examples.p', 'wb'))
