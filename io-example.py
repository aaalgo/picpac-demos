#!/usr/bin/env python

# This file illustrates how to import records into picpac db and
# how to read from the db.

# Usually one either import images for classification or segmentation for
# training purpose, even though both kinds of records are imported in this
# example.

from __future__ import absolute_import, division, print_function
import os
import cv2
import simplejson as json
import picpac

# create a picpac db
# make sure the file doesn't already exist
try:
    os.remove('test.db')
except:
    pass

db = picpac.Writer('test.db')

# append images for classification

label = 1
# directly append file content
with open('lenna.jpg', 'rb') as f:
    # append label, image
    buf = f.read()
    db.append(label, buf)

image = cv2.imread('lenna.jpg', cv2.IMREAD_COLOR)
# encode np.array image to string and append
buf = cv2.imencode('.png', image)[1].tostring()
db.append(label, buf)


# append image and annotations for classification

# all coordinates are normalized to [0,1]
# that is, x and width are divided by rows
# and y and height are divided by cols

anno = {"shapes": [ # annotation can contain multiple shapes
            # shape1, rect
            { "type": "rect",
            # "label": 1  -- by default all labels are 1
              "geometry": {"y": 0.384,
                          "x": 0.419,
                          "height": 0.377,
                          "width":  0.281}},
            {"type": "polygon",
             "label": 2,
                "geometry": {"points": [{"y": 0.459, "x": 0.265},
                    {"y": 0.456, "x": 0.265}, {"y": 0.456, "x": 0.265},
                    {"y": 0.322, "x": 0.225}, {"y": 0.322, "x": 0.225},
                    {"y": 0.172, "x": 0.259}, {"y": 0.172, "x": 0.259},
                    {"y": 0.086, "x": 0.35}, {"y": 0.086, "x": 0.350},
                    {"y": 0.081, "x": 0.461}, {"y": 0.081, "x": 0.461},
                    {"y": 0.131, "x": 0.577}, {"y": 0.131, "x": 0.577},
                    {"y": 0.279, "x": 0.681}, {"y": 0.279, "x": 0.681},
                    {"y": 0.213, "x": 0.786}, {"y": 0.213, "x": 0.786},
                    {"y": 0.225, "x": 0.809}, {"y": 0.225, "x": 0.809},
                    {"y": 0.25, "x": 0.815}, {"y": 0.25, "x": 0.815},
                    {"y": 0.359, "x": 0.795}, {"y": 0.359, "x": 0.795},
                    {"y": 0.418, "x": 0.720}, {"y": 0.418, "x": 0.720},
                    {"y": 0.363, "x": 0.702}, {"y": 0.363, "x": 0.702},
                    {"y": 0.443, "x": 0.55}, {"y": 0.443, "x": 0.55},
                    {"y": 0.543, "x": 0.45}, {"y": 0.543, "x": 0.45},
                    {"y": 0.747, "x": 0.297}, {"y": 0.747, "x": 0.297},
                    {"y": 0.684, "x": 0.277}, {"y": 0.684, "x": 0.277},
                    {"y": 0.615, "x": 0.302}, {"y": 0.615, "x": 0.302}]}},
        ]}

# append annotation, this example doesn't have an classification label
# 
db.append(buf, json.dumps(anno))

# ellipse has the same form of its bounding rectangle
anno = {"shapes": [ # annotation can contain multiple shapes
            {"type": "ellipse",
             "label": 1,
                "geometry": {"y": 0.384,
                          "x": 0.419,
                          "height": 0.377,
                          "width":  0.281}},
        ]}
db.append(buf, json.dumps(anno))

del db



db = picpac.Reader('test.db')

for label, _, _, fields in db:
    print("image size is %d" % len(fields[0]))

    if len(fields) == 1:
        print("  classification example, label = %f" % label)
    else:
        anno = json.loads(fields[1])
        shapes = [s['type'] for s in anno['shapes']]
        print("  annotation example, annotation shapes are %s" % shapes)
        pass
