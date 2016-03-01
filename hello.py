#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from local_settings import API_KEY, API_SECRET
from pprint import pformat
from facepp import API


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
dataset_name = 'testdata'


def train(persons):
    # Step 1: Detect faces in the 3 pictures and find out their positions and attributes
    FACES = {name: api.detection.detect(url=url) for name, url in persons}
    for name, face in FACES.iteritems():
        print_result(name, face)

    # Step 2: create persons using the face_id
    for name, face in FACES.iteritems():
        rst = api.person.create(person_name=name, face_id=face['face'][0]['face_id'])
        print_result('create person {}'.format(name), rst)

    # Step 3: create a new group and add those persons in it
    rst = api.group.create(group_name=dataset_name)
    print_result('create group', rst)
    rst = api.group.add_person(group_name=dataset_name, person_name=FACES.iterkeys())
    print_result('add these persons to group', rst)

    # Step 4: train the model
    rst = api.train.identify(group_name=dataset_name)
    print_result('train', rst)
    rst = api.wait_async(rst['session_id'])
    print_result('wait async', rst)


def pred():
    TARGET_IMAGE = 'http://cn.faceplusplus.com/static/resources/python_demo/4.jpg'
    rst = api.recognition.identify(group_name=dataset_name, url=TARGET_IMAGE)
    print_result('recognition result', rst)
    print '=' * 60
    print 'The person with highest confidence:', rst['face'][0]['candidate'][0]['person_name']


# def cleanup():
#     api.group.delete(group_name=dataset_name)
#     api.person.delete(person_name=FACES.iterkeys())


if __name__ == '__main__':
    persons = []
    for line in open('dataset.txt').readlines():
        split_ind = line.find(' ')
        url = line[:split_ind].strip()
        name = line[split_ind:].strip()
        if name and url:
            persons.append((name, url))

    # train(persons)
    pred()
