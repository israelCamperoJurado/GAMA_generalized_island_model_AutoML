# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:03:15 2021

@author: 20210595
"""

import psutil

def on_terminate(proc):
    print("process {} terminated with exit code {}".format(proc, proc.returncode))

procs = psutil.Process().children()
for p in procs:
    p.terminate()
gone, alive = psutil.wait_procs(procs, timeout=3, callback=on_terminate)
for p in alive:
    p.kill()