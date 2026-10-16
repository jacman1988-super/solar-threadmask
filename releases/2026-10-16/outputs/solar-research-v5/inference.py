"""Verified image-only local inference; never reads annotations or submits."""
import argparse
import numpy as np
from PIL import Image
from runtime import prepare,segment,METHODS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image');p.add_argument('output');p.add_argument('--method',choices=METHODS)
    a=p.parse_args()
    try:
        with Image.open(a.image) as img:image=np.asarray(img)
        labels=segment(image,a.method);np.save(a.output,labels)
    except (ValueError,OSError) as error:p.error(str(error))
    print('Local instances:',int(labels.max()))
