# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:39:21 2022

@author: 20210595
"""
from typing import List, Optional, Dict, Tuple, Any, Union

import math
maximum_resource = 3000
minimum_resource = 100
reduction_factor = 2
minimum_early_stopping_rate=1

max_rung = math.ceil(
    math.log(maximum_resource / minimum_resource, reduction_factor)
)
rungs = range(minimum_early_stopping_rate, max_rung + 1)
rung_resources = {
    rung: min(minimum_resource * (reduction_factor ** rung), maximum_resource)
    for rung in rungs
}


# Highest rungs first is how we typically iterate them
# Should we just use lists of lists/heaps instead?
rung_individuals = {
    rung: [] for rung in reversed(rungs)
}

promoted_individuals = {
        rung: [] for rung in reversed(rungs)
}


for rung, individuals in list(rung_individuals.items())[1:]:
    print("rung", rung)
    print("individual",  individuals)
    
# Nosotros usaremos la funcion  def get_job(): solamente para orderar los individuos