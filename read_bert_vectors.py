import json


class json_data_generator:
    def __init__(self, dir, filename):
        self.dir = dir
        self.filename = filename

    def read_json_file(self):
        with open("{}/{}".format(self.dir, self.filename)) as f:
            for line in f:
                yield json.loads(line)


test = json_data_generator('large_data', 'pretraining_data_vector_output.json')
hi = test.read_json_file()
for k in hi:
    print('one_json')
    for test in hi:
        if test['features'][0]['token'] != '[CLS]':
            print(test['features'][0]['token'])

