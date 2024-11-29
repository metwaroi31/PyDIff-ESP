# flake8: noqa
import os.path as osp
from basicsr.infer import infer_pipeline

import pydiff.archs
import pydiff.data
import pydiff.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    infer_pipeline(root_path)
