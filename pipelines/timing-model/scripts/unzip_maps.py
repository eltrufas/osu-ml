import glob
import zipfile
import re

id_re = re.compile('maps\/(\d+)*.')
for filename in glob.glob('maps/*.osz'):
    id = id_re.match(filename).group(1)
    try:
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall('extracted_maps/{}'.format(id))
    except:
        pass

