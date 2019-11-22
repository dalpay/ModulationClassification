#!/usr/bin/env python
from gnuradio import gr
import numpy as np
from features import cumulants, amplitude_stats, phase_stats
from FeedforwardNetwork import FeedforwardNetwork


class classifier(gr.basic_block):
    def __init__(self, feature_types, model_path):
        gr.basic_block.__init__(self, 
            name="cumulant_features",
            in_sig=[np.complex64],
            out_sig=[np.float32])

        self.ffnn = FeedforwardNetwork(model_path, 3)


    def work(self, input_items, output_items):
        
        inp = input_items[0]
        out = output_items[0]
        rtn = 0

        if (len(inp) >= 1024):

            cums = cumulants(inp)
            amp_stats = amplitude_stats(inp)
            phs_stats = phase_stats(inp)
            features = np.hstack((cums, amp_stats, phs_stats))

            out[:len(features)] = features
            self.consume_each(len(inp))
            
            prediction = self.ffnn.predict(features)
            print(prediction)

            return len(features)

        else:
            return 0.0



