#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import numpy as np
import time
import os

from features import cumulants, amplitude_stats, phase_stats
from FeedforwardNetwork import FeedforwardNetwork


class top_block(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Top Block")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 1e6
        self.noise_level = noise_level = 1e-6

        ##################################################
        # Blocks
        ##################################################
        self.random_source = blocks.vector_source_b(
                map(int, np.random.randint(0, 255, 10000)), False)
        # self.modulator = digital.gfsk_mod(
        #     samples_per_symbol=2,
        #     sensitivity=1.0,
        #     bt=0.5,
        #     verbose=False,
        #     log=False,
        # )
        # self.modulator = digital.gmsk_mod(
        #     samples_per_symbol=2,
        #     bt=0.5,
        #     verbose=False,
        #     log=False,
        # )
        # self.modulator = digital.psk.psk_mod(
        #     constellation_points=8, # 2, 4, 8
        #     mod_code="gray",
        #     differential=True,
        #     samples_per_symbol=2,
        #     excess_bw=0.5,
        #     verbose=False,
        #     log=False,
        # )
        self.modulator = digital.qam.qam_mod(
            constellation_points=4, # 4, 16, 64
            mod_code="gray",
            differential=True,
            samples_per_symbol=2,
            excess_bw=0.5,
            verbose=False,
            log=False,
        )
        self.channel_model = channels.channel_model(
            noise_voltage=noise_level,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=(1.0 + 1.0j, ),
            noise_seed=0,
            block_tags=False
        )
        num_samps = 2048
        self.stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex*1, num_samps)
        self.probe_signal = blocks.probe_signal_vc(num_samps)

        ##################################################
        # Connections
        ##################################################
        self.connect(self.random_source, self.modulator, self.channel_model,
            self.stream_to_vector, self.probe_signal)
        
    def get_signal(self):
        return self.probe_signal.level()
        
    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_noise_level(self):
        return self.noise_level

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level
        self.channel_model.set_noise_voltage(self.noise_level)

def main(top_block_cls=top_block, options=None):

    tb = top_block_cls()
    tb.start()

    model_path = '../model_1/model_1'
    ffnn = FeedforwardNetwork(model_path, 3)
    mod_types = {
                0: 'GFSK', 
                1: 'GMSK',
                2: 'QAM4',
                3: 'QAM16',
                4: 'QAM64', 
                5: 'PSK2', 
                6: 'PSK4', 
                7: 'PSK8'
    }
    
    while (1):
        
        tb.stop()
        tb.wait()
        tb = top_block_cls()
        tb.start()

        time.sleep(1)
           
        iq_data = tb.get_signal()
        iq_data = np.array(iq_data, dtype=np.complex64)

        cums = cumulants(iq_data)
        amp_stats = amplitude_stats(iq_data)
        phs_stats = phase_stats(iq_data)
        features = np.hstack((cums, amp_stats, phs_stats))

        pred = ffnn.predict(features)
        print(mod_types[pred])
    
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
