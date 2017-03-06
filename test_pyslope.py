# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
import cProfile
import numpy as np
from pprint import pprint
import datetime

import ccd
import test
from test.shared import read_data
import pyslope.pvalue_slope as slope

def test_validate_sample_2_detection_results():
    """ Sample 2 contains two changes and should test the full path through
    the algorithm"""
    col = 0
    
    fmt= '%Y%j'
    import datetime

    start_days=[30793,32281,34441,37801,40657]
    end_days=[32249,34105,37785,40473,41833]

    data = read_data("c:\\temp\\test_3657_3610(Observation)_full.csv")
    processing_mask = read_data("c:\\temp\\test_3657_3610_processing_mask.csv")

    print(data[0][0], data[1][0], data[2][0], data[3][0], data[4][0],
                         data[5][0], data[6][0], data[7][0], data[8][0])
    print(processing_mask[0],processing_mask[1])

    # results = pyslope.slope(data[0], data[1], data[2], data[3], data[4],
    #                      data[5], data[6], data[7], data[8],duplicate_dates=True)

    dates = data[0]
    observations = data[1:8]

    results = slope.calc_pvalue_slope(dates,observations,
                                      processing_mask,start_days,end_days)

    return results

results = test_validate_sample_2_detection_results()
