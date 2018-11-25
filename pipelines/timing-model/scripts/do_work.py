# We'll need numpy for some mathematical operations
import numpy as np

# Librosa for audio
import librosa
import glob
import re
import sys
import pickle

# module for beatmap parsing
import osuparse

thread_id = sys.argv[1]
work_to_do = pickle.load(open("work/{}.p".format(sys.argv[1]),
                         'rb'))

print("[Process {}] Processing {} sets".format(thread_id, len(work_to_do)))

id_re = re.compile('(extracted_maps\/.+\/).*')

filenames = [filename for directory in work_to_do
             for filename in glob.glob("{}/*.osu".format(directory))]

maps = []
for filename in filenames:
    try:
        maps.append((id_re.match(filename).group(1),osuparse.parse_beatmap(filename)))
    except ValueError:
        pass

maps = [(p, map) for (p, map) in maps if len(map['timing_points']) < 100]

print("[Process {}]: starting work".format(thread_id))

def load_and_process_audio_file(path):
    y, sr = librosa.load(path)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    o_env = librosa.onset.onset_strength(y_percussive, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times

def make_key(beatmap):
    return (beatmap['metadata']['beatmap_set_id'],
            beatmap['general']['audio_filename'])


def make_val(path, beatmap):
    return path + beatmap['general']['audio_filename']

audio_file_feature_index = {make_key(beatmap): make_val(path, beatmap)
                            for (path, beatmap) in maps}

n_items = len(audio_file_feature_index)

for i, item in enumerate(audio_file_feature_index.items()):
    key, path = item
    print('[Process {}]: Processing {} ({}/{})'.format(thread_id, path, i, n_items))
    try:
        audio_file_feature_index[key] = load_and_process_audio_file(path)
    except:
        audio_file_feature_index[key] = None

examples = []

for (_, beatmap) in maps:
    features = audio_file_feature_index[(beatmap['metadata']['beatmap_set_id'],
                                         beatmap['general']['audio_filename'])]
    examples.append((beatmap, features))

example_file_path = 'examples/{}.p'.format(sys.argv[1])
print("[Process {}]: writing examples to {}".format(thread_id, example_file_path))
pickle.dump(examples, open(example_file_path, 'wb'))


