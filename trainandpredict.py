#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
from local_settings import API_KEY, API_SECRET
from pprint import pformat
from facepp import API, APIError


def print_result(hint, result):
    def encode(obj):
        if type(obj) is unicode:
            return obj.encode('utf-8')
        if type(obj) is dict:
            return {encode(k): encode(v) for (k, v) in obj.iteritems()}
        if type(obj) is list:
            return [encode(i) for i in obj]
        return obj
    print hint
    result = encode(result)
    print '\n'.join(['  ' + i for i in pformat(result, width=75).split('\n')])


api = API(API_KEY, API_SECRET)
dataset_name = 'testdata_1'


def train(persons):
    # Step 1: Detect faces in the 3 pictures and find out their positions and attributes

    names = set()
    for name, url in persons:
        try:
            detect_data = api.detection.detect(url=url)
        except:
            continue
        if 'face' in detect_data and len(detect_data['face']) > 0:
            face_id = detect_data['face'][0]['face_id']

            try:
                api.person.get_info(person_name=name)
                rst = api.person.add_face(person_name=name, face_id=face_id)
                print_result('create face to person {}'.format(name), rst)
            except APIError:
                rst = api.person.create(person_name=name, face_id=face_id)
                print_result('create person {}'.format(name), rst)
            names.add(name)

    # Step 3: create a new group and add those persons in it
    rst = api.group.create(group_name=dataset_name)
    print_result('create group', rst)
    rst = api.group.add_person(group_name=dataset_name, person_name=list(names))
    print_result('add these persons to group', rst)

    # Step 4: train the model
    rst = api.train.identify(group_name=dataset_name)
    print_result('train', rst)
    rst = api.wait_async(rst['session_id'])
    print_result('wait async', rst)


def pred(url):
    rst = api.recognition.identify(group_name=dataset_name, url=url)
    if 'face' in rst and len(rst['face']) > 0:
        predicted = rst['face'][0]['candidate'][0]
        score = predicted['confidence']
        return predicted['person_name'], score
    return '', ''


def predict_all():
    result_file = 'face_recognition_result.txt'
    if os.path.exists(result_file):
        finished_urls = [line.strip().split()[0] for line in open(result_file).readlines() if line.strip()]
    else:
        finished_urls = []

    to_predict_urls = [url.strip().split() for url in open('/Users/xiaogang/Desktop/viki/10591c_testset.txt').readlines() if url.strip()]

    with open(result_file, 'a') as f:
        for ind, (url, true_label) in enumerate(to_predict_urls):
            if (url not in training_urls) and (url not in finished_urls):
                pred_label, pred_score = pred(url)
                line = '{}\t{}\t{}\t{}'.format(url, true_label, pred_label, pred_score)
                print '{}/{}\t{}'.format(ind, len(to_predict_urls), line)
                f.write('{}\n'.format(line))

    labels = [line.strip().split()[1:3] for line in open(result_file).readlines() if line.strip()]
    correct_labels = [1 for t, p in labels if p.lower().startswith(t.lower())]
    print 'accuracy: {}'.format(float(len(correct_labels)) / len(labels))


def cleanup(names):
    try:
        print 'cleaning up group {}'.format(dataset_name)
        api.group.delete(group_name=dataset_name)
    except:
        pass
    for name in set(names):
        try:
            print 'cleaning up person {}'.format(name)
            api.person.delete(person_name=[name])
        except:
            pass

if __name__ == '__main__':
    persons = []
    training_data_file = '../training_data.txt'
    for line in open(training_data_file).readlines():
        url, name = line.strip().split('\t')
        url = url.strip()
        name = name.strip()
        if name and url:
            persons.append((name, url))
    training_urls = [url for name, url in persons]

    # cleanup([name for name, url in persons])
    # train(persons)

    predict_all()
