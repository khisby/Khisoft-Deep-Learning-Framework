try:
    import cv2
    import numpy as np
    import os
    import random
    import sys
    import matplotlib.pyplot as plt
except:
    print("Library tidak ditemukan !")
    print("Pastikan library cv2, os, numpy, random, sys sudah terinstall")


class ProgressBar:
    def __init__(self):
        self.callback = False

    def set(self, progressbar=False):
        self.callback = progressbar

    def get(self):
        return self.callback
