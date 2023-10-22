
# MIT License
#
# Copyright (c) 2023 Masaru Nakajima
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Train with ASL fingerspelling data.
"""

import argparse
import os
from os import path
import json

def main():
    """
    Main function for training.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="Input data directory")

    args = ap.parse_args()

    dir_path = args.data
    preproc_path = path.join(dir_path, "preproc_args.json")
    data_path = path.join(dir_path, "train.tfrecords")

    if not path.exists(preproc_path) or not path.exists(data_path):
        raise RuntimeError("Data files not found")

    param = json.load(open(preproc_path))
    print(param.keys())






if __name__ == "__main__":
    main()




