import json
import argparse
import os
import numpy as np

img_size = 512

def add_extr(extr, cx, cy, fx, fy):
    fx = fx/img_size
    fy = fy/img_size
    cx = cx/img_size
    cy = cy/img_size

    krt = np.zeros(9)
    krt[0] = fx
    krt[2] = cx
    krt[4] = fy
    krt[5] = cy
    krt[8] = 1

    krt = krt.tolist()
    extr.extend(krt)

    return extr

def read_procrust_json(path):

    if not os.path.exists(path):
        assert False, 'Path does not exist'

    with open(path) as f:
        rt_procrust = json.load(f)["procrust_transform"]

    rt_procrust = np.array(rt_procrust).reshape(4, 4)
    axis = rt_procrust[0:3, 0]

    scale = np.linalg.norm(axis)
    translation = rt_procrust[0:3, 3]

    return scale, translation

def process(actor, out_f):
    gantransformjson = {'labels': []}
    print('Processing actor: {}'.format(actor))

    procrust_path = "./procrust.json"
    scale, translation = read_procrust_json(procrust_path)
    
    transforms_json = os.path.join(actor, 'transforms_test.json')
    with open(transforms_json) as f:
        data = json.load(f)
    frames = data['frames']

    for frame in frames:
        name = frame['file_path'].replace('images/', '')

        cam2world = np.array(frame['transform_matrix']).reshape(4, 4)
        world2cam = np.linalg.inv(cam2world)

        world2cam[0:3,3] = scale * world2cam[0:3,3]
        cam2world = np.linalg.inv(world2cam)

        # cam2world[0,3] -= translation[0]
        # cam2world[1,3] -= translation[1]
        # cam2world[2,3] -= translation[2]

        cx = data['cx']
        cy = data['cy']
        fx = data['fl_x']
        fy = data['fl_y']

        extr = cam2world.reshape(16).tolist()
        krt = add_extr(extr, cx, cy, fx, fy)

        gantransformjson['labels'].append([name, krt])

    if not os.path.exists(out_f):
        os.makedirs(out_f, exist_ok=True)

    out_file = os.path.join(out_f, 'dataset_test.json')
    print(out_file)
    with open(out_file, 'w') as f:
        json.dump(gantransformjson, f, indent=4)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actor = '/dataset/insta/nf_01'
    out_f = '/dataset/ganavatar/nf_01'
    process(actor, out_f)
    #copyimgs(actor, out_f)

