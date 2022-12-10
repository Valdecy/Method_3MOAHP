################################################################################

# Required Libraries
import math
import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats

from fractions import Fraction
from method_3mo_ahp.util.ahp import ahp_method
from method_3mo_ahp.util.fuzzy_ahp import fuzzy_ahp_method
from method_3mo_ahp.util.u_n_iii import fast_non_dominated_sorting, unified_non_dominated_sorting_genetic_algorithm_III

################################################################################

# 3MOAHP Class
class load_3moahp():
    def __init__(self, original, min_limit = [], max_limit = [], wdm = 'geometric', fuzzy_scale = False, custom_fuzzy_scale = []):
      self.wd      = wdm 
      self.f_flag  = fuzzy_scale
      max_abs_diff = 0 
      if (fuzzy_scale == False):
        self.dataset = original
        if (len(min_limit) > 0):
          self.dataset_min = min_limit
        else:
          self.dataset_min = np.zeros(( self.dataset.shape))
          for i in range(0, self.dataset.shape[0]):
            for j in range(1, self.dataset.shape[1]):
              if (j > i):
                self.dataset_min[i, j] = 1/9
        if (len(max_limit) > 0):
          self.dataset_max = max_limit
        else:
          self.dataset_max = np.zeros(( self.dataset.shape))
          for i in range(0, self.dataset.shape[0]):
            for j in range(1, self.dataset.shape[1]):
              if (j > i):
                self.dataset_max[i, j] = 9
      elif (fuzzy_scale == True):
        self.dataset = np.zeros((len(original), len(original[0])), dtype = object)
        for i in range(0, len(original)):
            for j in range(0, len(original[i])):
              self.dataset[i, j] = original[i][j]
        if (len(min_limit) > 0):
          self.dataset_min = np.zeros((len(min_limit), len(min_limit[0])), dtype = object)
          for i in range(0, len(min_limit)):
              for j in range(0, len(min_limit[i])):
                self.dataset_min[i, j] = min_limit[i][j]
        else:
          self.dataset_min = np.zeros((len(original), len(original[0])), dtype = object)
          for i in range(0, self.dataset.shape[0]):
            for j in range(1, self.dataset.shape[1]):
              if (j > i):
                self.dataset_min[i, j] = (1/9, 1/9, 1/9)
        if (len(max_limit) > 0):
          self.dataset_max = np.zeros((len(max_limit), len(max_limit[0])), dtype = object)
          for i in range(0, len(max_limit)):
              for j in range(0, len(max_limit[i])):
                self.dataset_max[i, j] = max_limit[i][j]
        else:
          self.dataset_max = np.zeros((len(original), len(original[0])), dtype = object)
          for i in range(0, self.dataset.shape[0]):
            for j in range(1, self.dataset.shape[1]):
              if (j > i):
                self.dataset_max[i, j] = (9, 9, 9)
      if (fuzzy_scale == False):
        self.saaty_scale = [1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.saaty_strg  = [str(Fraction(item).limit_denominator()) if item < 1 else str(item) for item in self.saaty_scale]
      elif (fuzzy_scale == True and len(custom_fuzzy_scale) == 0):
        self.saaty_scale = [ (1/9, 1/9, 1/9), (1/9, 1/8, 1/7), (1/8, 1/7, 1/6), (1/7, 1/6, 1/5), (1/6, 1/5, 1/4),(1/5, 1/4, 1/3), (1/4, 1/3, 1/2), (1/3, 1/2, 1), (1, 1, 1), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7), (6, 7, 8), (7, 8, 9), (9, 9, 9) ]
        self.saaty_strg  = []
        for fuzzy_number in self.saaty_scale:
          srt = '('
          for item in fuzzy_number:
            if (item < 1):
              srt = srt + str(Fraction(item).limit_denominator()) + ', '
            else:
              srt = srt + str(item) + ', '
          srt = srt[:-2]+')'
          self.saaty_strg.append(srt)
      elif (fuzzy_scale == True and len(custom_fuzzy_scale) > 0):
        self.saaty_scale = [item for item in custom_fuzzy_scale]
        self.saaty_strg  = []
        for fuzzy_number in self.saaty_scale:
          srt = '('
          for item in fuzzy_number:
            if (item < 1):
              srt = srt + str(Fraction(item).limit_denominator()) + ', '
            else:
              srt = srt + str(item) + ', '
          srt = srt[:-2]+')'
          self.saaty_strg.append(srt)
      limit_idx       = [i for i in range(0, len(self.saaty_scale))]
      self.dict_limit = dict(zip(self.saaty_scale, limit_idx))
      for i in range(0, self.dataset.shape[0]):
        for j in range(1, self.dataset.shape[1]):
          if (j > i):
            a = self.dict_limit[self.dataset[i,j]]
            b = self.dict_limit[self.dataset_min[i,j]]
            c = self.dict_limit[self.dataset_max[i,j]]
            if (abs(a - b) > max_abs_diff):
              max_abs_diff = abs(a - b)
            if (abs(c - a) > max_abs_diff):
              max_abs_diff = abs(c - a) 
      self.limit     = max_abs_diff  
      self.deviation = list(range(-self.limit, self.limit+1))  
      self.ranges    = [1/len(self.deviation)*i for i in range(0, len(self.deviation)+1)] 
      self.dict_rng  = dict(zip(self.deviation, self.ranges))
      self.minv      = []
      self.maxv      = []
      for i in range(0, self.dataset.shape[0]):
        for j in range(1, self.dataset.shape[1]):
          if (j > i):
            a = self.dict_limit[self.dataset[i,j]]
            b = self.dict_limit[self.dataset_min[i,j]]
            c = self.dict_limit[self.dataset_max[i,j]]
            d = self.dict_rng[b-a]
            e = self.dict_rng[c-a]
            self.minv.append(d)
            self.maxv.append(e)
      self.cols  = ['#6929c4', '#9f1853', '#198038', '#b28600', '#8a3800', '#1192e8', '#fa4d56', '#002d9c', 
                    '#009d9a', '#a56eff', '#005d5d', '#570408', '#ee538b', '#012749', '#da1e28', '#f1c21b', 
                    '#ff832b', '#198038', '#bdd9bf', '#929084', '#ffc857', '#a997df', '#e5323b', '#2e4052', 
                    '#e1daae', '#ff934f', '#cc2d35', '#058ed9', '#848fa2', '#2d3142', '#62a3f0', '#cc5f54', 
                    '#e6cb60', '#523d02', '#c67ce6', '#00b524', '#4ad9bd', '#f53347', '#565c55',
                    '#000000', '#ffff00', '#1ce6ff', '#ff34ff', '#ff4a46', '#008941', '#006fa6', '#a30059',
                    '#ffdbe5', '#7a4900', '#0000a6', '#63ffac', '#b79762', '#004d43', '#8fb0ff', '#997d87',
                    '#5a0007', '#809693', '#feffe6', '#1b4400', '#4fc601', '#3b5dff', '#4a3b53', '#ff2f80',
                    '#61615a', '#ba0900', '#6b7900', '#00c2a0', '#ffaa92', '#ff90c9', '#b903aa', '#d16100',
                    '#ddefff', '#000035', '#7b4f4b', '#a1c299', '#300018', '#0aa6d8', '#013349', '#00846f',
                    '#372101', '#ffb500', '#c2ffed', '#a079bf', '#cc0744', '#c0b9b2', '#c2ff99', '#001e09',
                    '#00489c', '#6f0062', '#0cbd66', '#eec3ff', '#456d75', '#b77b68', '#7a87a1', '#788d66',
                    '#885578', '#fad09f', '#ff8a9a', '#d157a0', '#bec459', '#456648', '#0086ed', '#886f4c',
                    '#34362d', '#b4a8bd', '#00a6aa', '#452c2c', '#636375', '#a3c8c9', '#ff913f', '#938a81',
                    '#575329', '#00fecf', '#b05b6f', '#8cd0ff', '#3b9700', '#04f757', '#c8a1a1', '#1e6e00',
                    '#7900d7', '#a77500', '#6367a9', '#a05837', '#6b002c', '#772600', '#d790ff', '#9b9700',
                    '#549e79', '#fff69f', '#201625', '#72418f', '#bc23ff', '#99adc0', '#3a2465', '#922329',
                    '#5b4534', '#fde8dc', '#404e55', '#0089a3', '#cb7e98', '#a4e804', '#324e72', '#6a3a4c',
                    '#83ab58', '#001c1e', '#d1f7ce', '#004b28', '#c8d0f6', '#a3a489', '#806c66', '#222800',
                    '#bf5650', '#e83000', '#66796d', '#da007c', '#ff1a59', '#8adbb4', '#1e0200', '#5b4e51',
                    '#c895c5', '#320033', '#ff6832', '#66e1d3', '#cfcdac', '#d0ac94', '#7ed379', '#012c58',
                    '#7a7bff', '#d68e01', '#353339', '#78afa1', '#feb2c6', '#75797c', '#837393', '#943a4d',
                    '#b5f4ff', '#d2dcd5', '#9556bd', '#6a714a', '#001325', '#02525f', '#0aa3f7', '#e98176',
                    '#dbd5dd', '#5ebcd1', '#3d4f44', '#7e6405', '#02684e', '#962b75', '#8d8546', '#9695c5',
                    '#e773ce', '#d86a78', '#3e89be', '#ca834e', '#518a87', '#5b113c', '#55813b', '#e704c4',
                    '#00005f', '#a97399', '#4b8160', '#59738a', '#ff5da7', '#f7c9bf', '#643127', '#513a01',
                    '#6b94aa', '#51a058', '#a45b02', '#1d1702', '#e20027', '#e7ab63', '#4c6001', '#9c6966',
                    '#64547b', '#97979e', '#006a66', '#391406', '#f4d749', '#0045d2', '#006c31', '#ddb6d0',
                    '#7c6571', '#9fb2a4', '#00d891', '#15a08a', '#bc65e9', '#fffffe', '#c6dc99', '#203b3c',
                    '#671190', '#6b3a64', '#f5e1ff', '#ffa0f2', '#ccaa35', '#374527', '#8bb400', '#797868',
                    '#c6005a', '#3b000a', '#c86240', '#29607c', '#402334', '#7d5a44', '#ccb87c', '#b88183',
                    '#aa5199', '#b5d6c3', '#a38469', '#9f94f0', '#a74571', '#b894a6', '#71bb8c', '#00b433',
                    '#789ec9', '#6d80ba', '#953f00', '#5eff03', '#e4fffc', '#1be177', '#bcb1e5', '#76912f',
                    '#003109', '#0060cd', '#d20096', '#895563', '#29201d', '#5b3213', '#a76f42', '#89412e',
                    '#1a3a2a', '#494b5a', '#a88c85', '#f4abaa', '#a3f3ab', '#00c6c8', '#ea8b66', '#958a9f',
                    '#bdc9d2', '#9fa064', '#be4700', '#658188', '#83a485', '#453c23', '#47675d', '#3a3f00',
                    '#061203', '#dffb71', '#868e7e', '#98d058', '#6c8f7d', '#d7bfc2', '#3c3e6e', '#d83d66',
                    '#2f5d9b', '#6c5e46', '#d25b88', '#5b656c', '#00b57f', '#545c46', '#866097', '#365d25',
                    '#252f99', '#00ccff', '#674e60', '#fc009c', '#92896b', '#1e2324', '#dec9b2', '#9d4948',
                    '#85abb4', '#342142', '#d09685', '#a4acac', '#00ffff', '#ae9c86', '#742a33', '#0e72c5',
                    '#afd8ec', '#c064b9', '#91028c', '#feedbf', '#ffb789', '#9cb8e4', '#afffd1', '#2a364c',
                    '#4f4a43', '#647095', '#34bbff', '#807781', '#920003', '#b3a5a7', '#018615', '#f1ffc8',
                    '#976f5c', '#ff3bc1', '#ff5f6b', '#077d84', '#f56d93', '#5771da', '#4e1e2a', '#830055',
                    '#02d346', '#be452d', '#00905e', '#be0028', '#6e96e3', '#007699', '#fec96d', '#9c6a7d',
                    '#3fa1b8', '#893de3', '#79b4d6', '#7fd4d9', '#6751bb', '#b28d2d', '#e27a05', '#dd9cb8',
                    '#aabc7a', '#980034', '#561a02', '#8f7f00', '#635000', '#cd7dae', '#8a5e2d', '#ffb3e1',
                    '#6b6466', '#c6d300', '#0100e2', '#88ec69', '#8fccbe', '#21001c', '#511f4d', '#e3f6e3',
                    '#ff8eb1', '#6b4f29', '#a37f46', '#6a5950', '#1f2a1a', '#04784d', '#101835', '#e6e0d0',
                    '#ff74fe', '#00a45f', '#8f5df8', '#4b0059', '#412f23', '#d8939e', '#db9d72', '#604143',
                    '#b5bace', '#989eb7', '#d2c4db', '#a587af', '#77d796', '#7f8c94', '#ff9b03', '#555196',
                    '#31ddae', '#74b671', '#802647', '#2a373f', '#014a68', '#696628', '#4c7b6d', '#002c27',
                    '#7a4522', '#3b5859', '#e5d381', '#fff3ff', '#679fa0', '#261300', '#2c5742', '#9131af',
                    '#af5d88', '#c7706a', '#61ab1f', '#8cf2d4', '#c5d9b8', '#9ffffb', '#bf45cc', '#493941',
                    '#863b60', '#b90076', '#003177', '#c582d2', '#c1b394', '#602b70', '#887868', '#babfb0',
                    '#030012', '#d1acfe', '#7fdefe', '#4b5c71', '#a3a097', '#e66d53', '#637b5d', '#92bea5',
                    '#00f8b3', '#beddff', '#3db5a7', '#dd3248', '#b6e4de', '#427745', '#598c5a', '#b94c59',
                    '#8181d5', '#94888b', '#fed6bd', '#536d31', '#6eff92', '#e4e8ff', '#20e200', '#ffd0f2',
                    '#4c83a1', '#bd7322', '#915c4e', '#8c4787', '#025117', '#a2aa45', '#2d1b21', '#a9ddb0',
                    '#ff4f78', '#528500', '#009a2e', '#17fce4', '#71555a', '#525d82', '#00195a', '#967874',
                    '#555558', '#0b212c', '#1e202b', '#efbfc4', '#6f9755', '#6f7586', '#501d1d', '#372d00',
                    '#741d16', '#5eb393', '#b5b400', '#dd4a38', '#363dff', '#ad6552', '#6635af', '#836bba',
                    '#98aa7f', '#464836', '#322c3e', '#7cb9ba', '#5b6965', '#707d3d', '#7a001d', '#6e4636',
                    '#443a38', '#ae81ff', '#489079', '#897334', '#009087', '#da713c', '#361618', '#ff6f01',
                    '#006679', '#370e77', '#4b3a83', '#c9e2e6', '#c44170', '#ff4526', '#73be54', '#c4df72',
                    '#adff60', '#00447d', '#dccec9', '#bd9479', '#656e5b', '#ec5200', '#ff6ec2', '#7a617e',
                    '#ddaea2', '#77837f', '#a53327', '#608eff', '#b599d7', '#a50149', '#4e0025', '#c9b1a9',
                    '#03919a', '#1b2a25', '#e500f1', '#982e0b', '#b67180', '#e05859', '#006039', '#578f9b',
                    '#305230', '#ce934c', '#b3c2be', '#c0bac0', '#b506d3', '#170c10', '#4c534f', '#224451',
                    '#3e4141', '#78726d', '#b6602b', '#200441', '#ddb588', '#497200', '#c5aab6', '#033c61',
                    '#71b2f5', '#a9e088', '#4979b0', '#a2c3df', '#784149', '#2d2b17', '#3e0e2f', '#57344c',
                    '#0091be', '#e451d1', '#4b4b6a', '#5c011a', '#7c8060', '#ff9491', '#4c325d', '#005c8b',
                    '#e5fda4', '#68d1b6', '#032641', '#140023', '#8683a9', '#cfff00', '#a72c3e', '#34475a',
                    '#b1bb9a', '#b4a04f', '#8d918e', '#a168a6', '#813d3a', '#425218', '#da8386', '#776133',
                    '#563930', '#8498ae', '#90c1d3', '#b5666b', '#9b585e', '#856465', '#ad7c90', '#e2bc00',
                    '#e3aae0', '#b2c2fe', '#fd0039', '#009b75', '#fff46d', '#e87eac', '#dfe3e6', '#848590',
                    '#aa9297', '#83a193', '#577977', '#3e7158', '#c64289', '#ea0072', '#c4a8cb', '#55c899',
                    '#e78fcf', '#004547', '#f6e2e3', '#966716', '#378fdb', '#435e6a', '#da0004', '#1b000f',
                    '#5b9c8f', '#6e2b52', '#011115', '#e3e8c4', '#ae3b85', '#ea1ca9', '#ff9e6b', '#457d8b',
                    '#92678b', '#00cdbb', '#9ccc04', '#002e38', '#96c57f', '#cff6b4', '#492818', '#766e52',
                    '#20370e', '#e3d19f', '#2e3c30', '#b2eace', '#f3bda4', '#a24e3d', '#976fd9', '#8c9fa8',
                    '#7c2b73', '#4e5f37', '#5d5462', '#90956f', '#6aa776', '#dbcbf6', '#da71ff', '#987c95',
                    '#52323c', '#bb3c42', '#584d39', '#4fc15f', '#a2b9c1', '#79db21', '#1d5958', '#bd744e',
                    '#160b00', '#20221a', '#6b8295', '#00e0e4', '#102401', '#1b782a', '#daa9b5', '#b0415d',
                    '#859253', '#97a094', '#06e3c4', '#47688c', '#7c6755', '#075c00', '#7560d5', '#7d9f00',
                    '#c36d96', '#4d913e', '#5f4276', '#fce4c8', '#303052', '#4f381b', '#e5a532', '#706690',
                    '#aa9a92', '#237363', '#73013e', '#ff9079', '#a79a74', '#029bdb', '#ff0169', '#c7d2e7',
                    '#ca8869', '#80ffcd', '#bb1f69', '#90b0ab', '#7d74a9', '#fcc7db', '#99375b', '#00ab4d',
                    '#abaed1', '#be9d91', '#e6e5a7', '#332c22', '#dd587b', '#f5fff7', '#5d3033', '#6d3800',
                    '#ff0020', '#b57bb3', '#d7ffe6', '#c535a9', '#260009', '#6a8781', '#a8abb4', '#d45262',
                    '#794b61', '#4621b2', '#8da4db', '#c7c890', '#6fe9ad', '#a243a7', '#b2b081', '#181b00',
                    '#286154', '#4ca43b', '#6a9573', '#a8441d', '#5c727b', '#738671', '#d0cfcb', '#897b77',
                    '#1f3f22', '#4145a7', '#da9894', '#a1757a', '#63243c', '#adaaff', '#00cde2', '#ddbc62',
                    '#698eb1', '#208462', '#00b7e0', '#614a44', '#9bbb57', '#7a5c54', '#857a50', '#766b7e',
                    '#014833', '#ff8347', '#7a8eba', '#274740', '#946444', '#ebd8e6', '#646241', '#373917',
                    '#6ad450', '#81817b', '#d499e3', '#979440', '#011a12', '#526554', '#b5885c', '#a499a5',
                    '#03ad89', '#b3008b', '#e3c4b5', '#96531f', '#867175', '#74569e', '#617d9f', '#e70452',
                    '#067eaf', '#a697b6', '#b787a8', '#9cff93', '#311d19', '#3a9459', '#6e746e', '#b0c5ae',
                    '#84edf7', '#ed3488', '#754c78', '#384644', '#c7847b', '#00b6c5', '#7fa670', '#c1af9e',
                    '#2a7fff', '#72a58c', '#ffc07f', '#9debdd', '#d97c8e', '#7e7c93', '#62e674', '#b5639e',
                    '#ffa861', '#c2a580', '#8d9c83', '#b70546', '#372b2e', '#0098ff', '#985975', '#20204c',
                    '#ff6c60', '#445083', '#8502aa', '#72361f', '#9676a3', '#484449', '#ced6c2', '#3b164a',
                    '#cca763', '#2c7f77', '#02227b', '#a37e6f', '#cde6dc', '#cdfffb', '#be811a', '#f77183',
                    '#ede6e2', '#cdc6b4', '#ffe09e', '#3a7271', '#ff7b59', '#4e4e01', '#4ac684', '#8bc891',
                    '#bc8a96', '#cf6353', '#dcde5c', '#5eaadd', '#f6a0ad', '#e269aa', '#a3dae4', '#436e83',
                    '#002e17', '#ecfbff', '#a1c2b6', '#50003f', '#71695b', '#67c4bb', '#536eff', '#5d5a48',
                    '#890039', '#969381', '#371521', '#5e4665', '#aa62c3', '#8d6f81', '#2c6135', '#410601',
                    '#564620', '#e69034', '#6da6bd', '#e58e56', '#e3a68b', '#48b176', '#d27d67', '#b5b268',
                    '#7f8427', '#ff84e6', '#435740', '#eae408', '#f4f5ff', '#325800', '#4b6ba5', '#adceff',
                    '#9b8acc', '#885138', '#5875c1', '#7e7311', '#fea5ca', '#9f8b5b', '#a55b54', '#89006a',
                    '#af756f', '#2a2000', '#576e4a', '#7f9eff', '#7499a1', '#ffb550', '#00011e', '#d1511c',
                    '#688151', '#bc908a', '#78c8eb', '#8502ff', '#483d30', '#c42221', '#5ea7ff', '#785715',
                    '#0cea91', '#fffaed', '#b3af9d', '#3e3d52', '#5a9bc2', '#9c2f90', '#8d5700', '#add79c',
                    '#00768b', '#337d00', '#c59700', '#3156dc', '#944575', '#ecffdc', '#d24cb2', '#97703c',
                    '#4c257f', '#9e0366', '#88ffec', '#b56481', '#396d2b', '#56735f', '#988376', '#9bb195',
                    '#a9795c', '#e4c5d3', '#9f4f67', '#1e2b39', '#664327', '#afce78', '#322edf', '#86b487',
                    '#c23000', '#abe86b', '#96656d', '#250e35', '#a60019', '#0080cf', '#caefff', '#323f61',
                    '#a449dc', '#6a9d3b', '#ff5ae4', '#636a01', '#d16cda', '#736060', '#ffbaad', '#d369b4',
                    '#ffded6', '#6c6d74', '#927d5e', '#845d70', '#5b62c1', '#2f4a36', '#e45f35', '#ff3b53',
                    '#ac84dd', '#762988', '#70ec98', '#408543', '#2c3533', '#2e182d', '#323925', '#19181b',
                    '#2f2e2c', '#023c32', '#9b9ee2', '#58afad', '#5c424d', '#7ac5a6', '#685d75', '#b9bcbd',
                    '#834357', '#1a7b42', '#2e57aa', '#e55199', '#316e47', '#cd00c5', '#6a004d', '#7fbbec',
                    '#f35691', '#d7c54a', '#62acb7', '#cba1bc', '#a28a9a', '#6c3f3b', '#ffe47d', '#dcbae3',
                    '#5f816d', '#3a404a', '#7dbf32', '#e6ecdc', '#852c19', '#285366', '#b8cb9c', '#0e0d00',
                    '#4b5d56', '#6b543f', '#e27172', '#0568ec', '#2eb500', '#d21656', '#efafff', '#682021',
                    '#2d2011', '#da4cff', '#70968e', '#ff7b7d', '#4a1930', '#e8c282', '#e7dbbc', '#a68486',
                    '#1f263c', '#36574e', '#52ce79', '#adaaa9', '#8a9f45', '#6542d2', '#00fb8c', '#5d697b',
                    '#ccd27f', '#94a5a1', '#790229', '#e383e6', '#7ea4c1', '#4e4452', '#4b2c00', '#620b70',
                    '#314c1e', '#874aa6', '#e30091', '#66460a', '#eb9a8b', '#eac3a3', '#98eab3', '#ab9180',
                    '#b8552f', '#1a2b2f', '#94ddc5', '#9d8c76', '#9c8333', '#94a9c9', '#392935', '#8c675e',
                    '#cce93a', '#917100', '#01400b', '#449896', '#1ca370', '#e08da7', '#8b4a4e', '#667776',
                    '#4692ad', '#67bda8', '#69255c', '#d3bfff', '#4a5132', '#7e9285', '#77733c', '#e7a0cc',
                    '#51a288', '#2c656a', '#4d5c5e', '#c9403a', '#ddd7f3', '#005844', '#b4a200', '#488f69',
                    '#858182', '#d4e9b9', '#3d7397', '#cae8ce', '#d60034', '#aa6746', '#9e5585', '#ba6200',
                    '#dee3E9', '#ebbaB5', '#fef3c7', '#a6e3d7', '#cbb4d5', '#808b96', '#f7dc6f', '#48c9b0',
                    '#af7ac5', '#ec7063', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53',
                    '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16',
                    '#d85679', '#12e193', '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80',
                    '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'
                  ]

    ################################################################################

    # Function: Load AHP Parameters
    def run_ahp(self):
        self.weights, self.rc = ahp_method(self.dataset, self.wd)
        for i in range(0, self.weights.shape[0]):
          print('w(g'+str(i+1)+'): ', round(self.weights[i], 3))
        print("")
        if (self.rc > 0.10):
          print('RC: ' + str(round(self.rc, 3)), ' The solution is inconsistent, the pairwise comparisons must be reviewed')
        else:
          print('RC: ' + str(round(self.rc, 3)), ' The solution is consistent') 
        self.plot_environment(self.dataset)
        return

    # Function: Load Fuzzy AHP Parameters
    def run_fuzzy_ahp(self):
        self.weights, self.rc = fuzzy_ahp_method(self.dataset)
        for i in range(0, self.weights.shape[0]):
          print('w(g'+str(i+1)+'): ', round(self.weights[i], 3))
        print("")
        if (self.rc > 0.10):
          print('RC: ' + str(round(self.rc, 3)), ' The solution is inconsistent, the pairwise comparisons must be reviewed')
        else:
          print('RC: ' + str(round(self.rc, 3)), ' The solution is consistent') 
        self.plot_environment(self.dataset)
        return
    
    # Function: Load U-NSGA-III Parameters
    def run_unsga3(self, references, mutation_rate, list_of_functions, generations, mu, eta, k):
        self.size = references
        self.m_r  = mutation_rate
        self.gen  = generations
        self.mu   = mu
        self.eta  = eta
        self.k    = k
        self.lof  = []
        self.qidx = [item for item in list_of_functions]
        for i in range(0, len(list_of_functions)):
            if   (list_of_functions[i] == 'f0' and self.f0 not in self.lof):
                self.lof.append(self.f0)
            elif (list_of_functions[i] == 'f1' and self.f1 not in self.lof):
                self.lof.append(self.f1)
            elif (list_of_functions[i] == 'f2' and self.f2 not in self.lof):
                self.lof.append(self.f2)
            elif (list_of_functions[i] == 'f3' and self.f3 not in self.lof):
                self.lof.append(self.f3)
            elif (list_of_functions[i] == 'f4' and self.f4 not in self.lof):
                self.lof.append(self.f4)
        self.run_()
        return
    
    ################################################################################
    
    # Function: Plot Environment
    def plot_environment(self, environment, plot_size = 10, policy = []):
      keys_j   = self.saaty_scale
      values_j = self.saaty_strg
      dict_j   = dict(zip(keys_j, values_j))
      fig      = plt.figure(figsize = [plot_size, plot_size], facecolor = 'w')
      ax       = fig.add_subplot(111, xticks = range(environment.shape[1] + 1), yticks = range(environment.shape[0] + 1), position = [0.1, 0.1, 0.8, 0.8])
      pairs    = [ (i,j) for (i,j), x in np.ndenumerate(environment) if i < j ]
      plt.gca().invert_yaxis()
      ax.grid(color = 'k', linestyle = '-', linewidth = 1)
      ax.xaxis.set_tick_params(bottom = 'off', top   = 'off', labelbottom = 'off')
      ax.yaxis.set_tick_params(left   = 'off', right = 'off', labelleft   = 'off')
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      for i in range(0, environment.shape[0]):
        for j in range(0, environment.shape[1]):
          ax.annotate(dict_j[environment[i, j]] , xy = (0.4 + j, 0.55 + i), fontsize = 10, fontweight = 'bold')
          if (i == j):
            ax.annotate(dict_j[environment[i, j]] , xy = (0.4 + j, 0.55 + i), fontsize = 10, fontweight = 'bold', color = 'w')
            black_stone = mpatches.Rectangle( (j, i), 1, 1, linewidth = 1, edgecolor = 'k', facecolor = 'k', clip_on = False)
            ax.add_patch(black_stone)
      if (len(policy ) > 0):
        for i in range(0, len(policy)):
          if (policy[i] < 0):
            i1, j1     = pairs[i]
            red_stone  = mpatches.Rectangle( (j1, i1), 1, 1, linewidth = 1, edgecolor = 'k', facecolor = 'orangered', clip_on = False)
            ax.add_patch(red_stone)
          elif (policy[i] > 0):
            i1, j1     = pairs[i]
            blue_stone = mpatches.Rectangle( (j1, i1), 1, 1, linewidth = 1, edgecolor = 'k', facecolor = 'lightblue', clip_on = False)
            ax.add_patch(blue_stone)
      return

    # Function: Plot Solutions
    def plot_con_solutions(self, view = 'browser'):
        dict_lst = []
        if ('f0' in self.qidx):
          dict_lst.append(dict(range = [self.ind_con['f0(MI)'].min()*1.00, self.ind_con['f0(MI)'].max()*1.00], label = 'f0(MI)', values = self.ind_con['f0(MI)']))
        if ('f1' in self.qidx):
          dict_lst.append(dict(range = [self.ind_con['f1(NC)'].min()*1.00, self.ind_con['f1(NC)'].max()*1.00], label = 'f1(NC)', values = self.ind_con['f1(NC)']))
        if ('f2' in self.qidx):
          dict_lst.append(dict(range = [self.ind_con['f2(KT)'].min()*1.00, self.ind_con['f2(KT)'].max()*1.00], label = 'f2(KT)', values = self.ind_con['f2(KT)']))
        if ('f3' in self.qidx):
          dict_lst.append(dict(range = [self.ind_con['f3(WA)'].min()*1.00, self.ind_con['f3(WA)'].max()*1.00], label = 'f3(WA)', values = self.ind_con['f3(WA)']))
        if ('f4' in self.qidx):
          dict_lst.append(dict(range = [self.ind_con['f4(LM)'].min()*1.00, self.ind_con['f4(LM)'].max()*1.00], label = 'f4(LM)', values = self.ind_con['f4(LM)']))
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        par_plot = go.Figure(data = go.Parcoords(
                                                  line       = dict(color = pd.Categorical(self.ind_con['Consistency']).codes, colorscale = [[0,'purple'], [0.5,'purple'], [1,'gold']]),
                                                  dimensions = dict_lst
                                                )
                            )
        par_plot.update_layout(font = dict(family = 'Arial Black', size = 25, color = 'black'))
        par_plot.show()
        return

    # Function: Complex Radar Plot
    def complex_radar_plot(self, idx, alpha = 0.5, size = 7):
        levels = 5
        names  = []
        if ('f0' in self.qidx):
          names.append('f0(MI)')
        if ('f1' in self.qidx):
          names.append('f1(NC)')
        if ('f2' in self.qidx):
          names.append('f2(KT)')
        if ('f3' in self.qidx):
          names.append('f3(WA)')
        if ('f4' in self.qidx):
          names.append('f4(LM)')
        if (len(idx) == 1):
          data = self.indicators.iloc[idx, :-1].values.reshape(len(idx), len(names))
        else:
          data = self.indicators.iloc[idx, :-1].values
        def scale_data(data, ranges):
            def invert(x, limits):
                return limits[1] - (x - limits[0])
            for d, (y1, y2) in zip(data[1:], ranges[1:]):
                assert (y1 <= d <= y2) or (y2 <= d <= y1)
            x1, x2 = ranges[0]
            d      = data[0]
            if x1 > x2:
                d      = invert(d, (x1, x2))
                x1, x2 = x2, x1
            sdata = [d]
            for d, (y1, y2) in zip(data[1:], ranges[1:]):
                if y1 > y2:
                    d      = invert(d, (y1, y2))
                    y1, y2 = y2, y1
                sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
            return sdata
        ranges  = []
        for j in range(0, len(names)):
          ranges.append( ( np.min(self.ind_con.iloc[:,j]), np.max(self.ind_con.iloc[:,j]) ) ) 
        angles  = np.arange(0, 360, 360./len(names))
        fig     = plt.figure(figsize = (size, size))
        axes    = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar = True, label = 'axes{}'.format(i)) for i in range(0, len(names))]
        _, text = axes[0].set_thetagrids(angles, labels = names)
        axes[0].xaxis.set_tick_params(pad = 40)
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid('off')
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid      = np.linspace(*ranges[i], num = levels+1)
            gridlabel = ['{}'.format(round(x, 2)) for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid     = grid[::-1] 
            gridlabel[0] = ''
            ax.set_rgrids(grid, labels = gridlabel, angle = angles[i])
            ax.set_ylim(*ranges[i])
        angle  = np.deg2rad(np.r_[angles, angles[0]])
        ranges = ranges
        ax     = axes[0]
        for i in range(0, data.shape[0]):
            sdata  = scale_data(data[i,:], ranges)
            ax.plot(angle, np.r_[sdata, sdata[0]], color = self.cols[i], alpha = alpha, linewidth = 0.1)
            sdata  = scale_data(data[i,:], ranges)
            ax.fill(angle, np.r_[sdata, sdata[0]], color = self.cols[i], alpha = alpha)
        return

    # Function: Scatter Plot 
    def plot_scatter(self, proj_view = '2D', view = 'browser'):
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        data     = []
        front    = np.array(self.ind_incon.iloc[:,:-1], dtype = np.dtype('float'))
        inc_list = [ 'Index: '+str(item)+'<br>'+'Solution: Inconsistent' for item in list(self.ind_incon.index)] 
        con_list = [ 'Index: '+str(item)+'<br>'+'Solution: Consistent'   for item in list(self.ind_con.index  )] 
        if (proj_view == '2D' or proj_view == '2d'):
            if (len(con_list) > 0):
              s_trace = go.Scatter(
                                  x         = self.ind_con.iloc[:, 0], # ~self.ind_con.index.isin(self.rank_idx)
                                  y         = self.ind_con.iloc[:, 1], # ~self.ind_con.index.isin(self.rank_idx)
                                  opacity   = 0.85,
                                  mode      = 'markers+text',
                                  marker    = dict(symbol = 'circle-dot', size = 8, color = 'red'),
                                  hovertext = con_list,
                                  name      = ''
                                  )
              data.append(s_trace)
            if (len(inc_list) > 0):
              n_trace = go.Scatter(
                                  x         = front[:, 0],
                                  y         = front[:, 1],
                                  opacity   = 0.5,
                                  mode      = 'markers+text',
                                  marker    = dict(symbol = 'circle-dot', size = 10, color = 'purple'),
                                  hovertext = inc_list,
                                  name      = ''
                                  )
              data.append(n_trace)
            layout  = go.Layout(showlegend   = False,
                                hovermode    = 'closest',
                                margin       = dict(b = 10, l = 5, r = 5, t = 10),
                                plot_bgcolor = 'white',
                                xaxis        = dict(  showgrid       = True, 
                                                      zeroline       = False, 
                                                      showticklabels = True, 
                                                      title          = self.indicators.columns[0],
                                                      tickmode       = 'array', 
                                                      gridcolor      = 'grey',
                                                      spikedash      = 'solid',
                                                      spikecolor     = 'blue',
                                                      spikethickness = 2
                                                  ),
                                yaxis        = dict(  showgrid       = True, 
                                                      zeroline       = False, 
                                                      showticklabels = True,
                                                      title          = self.indicators.columns[1],
                                                      tickmode       = 'array', 
                                                      gridcolor      = 'grey',
                                                      spikedash      = 'solid',
                                                      spikecolor     = 'blue',
                                                      spikethickness = 2
                                                    )
                                )
            fig_aut = go.Figure(data = data, layout = layout)
            fig_aut.update_traces(textfont_size = 10, textfont_color = 'white') 
            fig_aut.show() 
        elif (proj_view == '3D' or proj_view == '3d'):
            if (len(con_list) > 0):
              s_trace = go.Scatter3d(
                                    x       = self.ind_con.iloc[:, 0], # ~self.ind_con.index.isin(self.rank_idx)
                                    y       = self.ind_con.iloc[:, 1], # ~self.ind_con.index.isin(self.rank_idx)
                                    z       = self.ind_con.iloc[:, 2], # ~self.ind_con.index.isin(self.rank_idx)
                                    opacity = 0.85,
                                    mode    = 'markers',
                                    marker  = dict(size = 10, color = 'red'),
                                    name    = ''
                                    )
              data.append(s_trace)
            if (len(inc_list) > 0):
              n_trace = go.Scatter3d(
                                    x       = front[:, 0],
                                    y       = front[:, 1],
                                    z       = front[:, 2],
                                    opacity = 0.5,
                                    mode    = 'markers',
                                    marker  = dict(size = 10, color = 'purple'),
                                    name    = ''
                                    )
              data.append(n_trace)
            layout  = go.Layout(showlegend   = False,
                                hovermode    = 'closest',
                                margin       = dict(b = 10, l = 5, r = 5, t = 10),
                                plot_bgcolor = 'white',
                                )
            fig_aut = go.Figure(data = data, layout = layout)
            fig_aut.update_traces(textfont_size = 10, textfont_color = 'white') 
            fig_aut.update_scenes(xaxis_visible = True, 
                                  yaxis_visible = True, 
                                  zaxis_visible = True,
                                  xaxis_title   = self.indicators.columns[0],
                                  yaxis_title   = self.indicators.columns[1],
                                  zaxis_title   = self.indicators.columns[2]
                                  )
            fig_aut.show() 
        return

    ################################################################################
    
    # Functions: Adjust Comparison
    def judgement_adjustment(self, value, deviation):
        idx   = self.saaty_scale.index(value)
        idx   = idx + deviation
        idx   = np.clip(idx, 0, len(self.saaty_scale) - 1)
        value = self.saaty_scale[int(idx)]
        return value
    
    # Functions: Decode Solution
    def convert_solution(self, sol):
        data      = np.array(self.dataset, copy = True) 
        policy    = [0]*len(sol[:-1])
        for i in range(0, len(sol[:-1])):
            for j in range(0, len(self.ranges)-1):
                lower = self.ranges[j]
                upper = self.ranges[j+1]
                if (sol[i] >= lower and sol[i] < upper):
                    policy[i] = self.deviation[j]
        count = 0
        for i in range(0, data.shape[0]):
            for j in range(i, data.shape[1]):
                if (i != j):
                    data[i, j] = self.judgement_adjustment(data[i, j], policy[count])
                    if (self.f_flag == False):
                      data[j, i] =  1/data[i, j]
                    else:
                      a, b, c   = data[i, j]
                      data[j,i] = (1/c, 1/b, 1/a)
                    count = count + 1  
        return policy, data

    ################################################################################
    # Function: Rank Decending (Adapted from: https://stackoverflow.com/questions/39059371/can-numpys-argsort-give-equal-element-the-same-rank)
    def rank_descending(self, x):
        _, inv = np.unique(-x, return_inverse = True, return_counts = False)
        return inv 
    
    # Functions: Objective Function 0 - Consistency Ratio (MI) 
    def f0(self, variables):
        data   = np.array(self.dataset, copy = True)  
        in_dev = [0]*len(variables)
        for i in range(0, len(variables)):
            for j in range(0, len(self.ranges)-1):
                lower = self.ranges[j]
                upper = self.ranges[j+1]
                if (variables[i] >= lower and variables[i] < upper):
                    in_dev[i] = self.deviation[j]
        count = 0
        for i in range(0, data.shape[0]):
            for j in range(i, data.shape[1]):
                if (i != j):
                    data[i, j] = self.judgement_adjustment(data[i, j], in_dev[count])
                    if (self.f_flag == False):
                      data[j, i] =  1/data[i, j]
                    else:
                      a, b, c   = data[i, j]
                      data[j,i] = (1/c, 1/b, 1/a)
                    count = count + 1        
        if (self.f_flag == False):
          _, adj_rc = ahp_method(data, self.wd)
        else:
          _, adj_rc = fuzzy_ahp_method(data)
        return adj_rc 
    
    # Functions: Objective Function 1 - Adjusted Comparisons (NC)
    def f1(self, variables):
        data   = np.array(self.dataset, copy = True)  
        ac_dev = 0
        in_dev = [0]*len(variables)
        for i in range(0, len(variables)):
            for j in range(0, len(self.ranges)-1):
                lower = self.ranges[j]
                upper = self.ranges[j+1]
                if (variables[i] >= lower and variables[i] < upper):
                    in_dev[i] = self.deviation[j]
        count = 0
        for i in range(0, data.shape[0]):
            for j in range(i, data.shape[1]):
                if (i != j):
                    data[i, j] = self.judgement_adjustment(data[i, j], in_dev[count])
                    if (self.f_flag == False):
                      data[j, i] =  1/data[i, j]
                    else:
                      a, b, c   = data[i, j]
                      data[j,i] = (1/c, 1/b, 1/a)
                    if (abs(in_dev[count]) > 0):
                        ac_dev = ac_dev + 1 
                    count = count + 1        
        return ac_dev
    
    # Functions: Objective Function 2 - Kendall Tau (KT)
    def f2(self, variables):
        data   = np.array(self.dataset, copy = True)  
        in_dev = [0]*len(variables)
        for i in range(0, len(variables)):
            for j in range(0, len(self.ranges)-1):
                lower = self.ranges[j]
                upper = self.ranges[j+1]
                if (variables[i] >= lower and variables[i] < upper):
                    in_dev[i] = self.deviation[j]
        count = 0
        for i in range(0, data.shape[0]):
            for j in range(i, data.shape[1]):
                if (i != j):
                    data[i, j] = self.judgement_adjustment(data[i, j], in_dev[count])
                    if (self.f_flag == False):
                      data[j, i] =  1/data[i, j]
                    else:
                      a, b, c   = data[i, j]
                      data[j,i] = (1/c, 1/b, 1/a)
                    count = count + 1        
        if (self.f_flag == False):
          w1, _ = ahp_method(self.dataset, self.wd)
          w2, _ = ahp_method(data, self.wd)
        else:
          w1, _ = fuzzy_ahp_method(self.dataset)
          w2, _ = fuzzy_ahp_method(data)
        w1             = self.rank_descending(w1)
        w2             = self.rank_descending(w2)
        kendall_tau, _ = stats.kendalltau(w1, w2)
        if (math.isnan(kendall_tau)):
            kendall_tau = -1
        return -kendall_tau
    
    # Functions: Objective Function 3 - Average Change in Weights (WA) 
    def f3(self, variables):
        data   = np.array(self.dataset, copy = True)  
        in_dev = [0]*len(variables)
        for i in range(0, len(variables)):
            for j in range(0, len(self.ranges)-1):
                lower = self.ranges[j]
                upper = self.ranges[j+1]
                if (variables[i] >= lower and variables[i] < upper):
                    in_dev[i] = self.deviation[j]
        count = 0
        for i in range(0, data.shape[0]):
            for j in range(i, data.shape[1]):
                if (i != j):
                    data[i, j] = self.judgement_adjustment(data[i, j], in_dev[count])
                    if (self.f_flag == False):
                      data[j, i] =  1/data[i, j]
                    else:
                      a, b, c   = data[i, j]
                      data[j,i] = (1/c, 1/b, 1/a)
                    count      = count + 1        
        if (self.f_flag == False):
          w1, _ = ahp_method(self.dataset, self.wd)
          w2, _ = ahp_method(data, self.wd)
        else:
          w1, _ = fuzzy_ahp_method(self.dataset)
          w2, _ = fuzzy_ahp_method(data)
        total = sum(abs(w1 - w2)/w1.shape[0])
        return total
    
    # Functions: Objective Function 4 - L1 Distance (LM)
    def f4(self, variables):
        data   = np.array(self.dataset, copy = True)  
        in_dev = [0]*len(variables)
        for i in range(0, len(variables)):
            for j in range(0, len(self.ranges)-1):
                lower = self.ranges[j]
                upper = self.ranges[j+1]
                if (variables[i] >= lower and variables[i] < upper):
                    in_dev[i] = self.deviation[j]
        count = 0
        for i in range(0, data.shape[0]):
            for j in range(i, data.shape[1]):
                if (i != j):
                    data[i, j] = self.judgement_adjustment(data[i, j], in_dev[count])
                    if (self.f_flag == False):
                      data[j, i] =  1/data[i, j]
                    else:
                      a, b, c   = data[i, j]
                      data[j,i] = (1/c, 1/b, 1/a)
                    count      = count + 1        
        if (self.f_flag == False):
          total = np.sum(abs(data - self.dataset))
        else:
          total = 0
          for i in range(0, data.shape[0]):
            for j in range(0, data.shape[1]):
              a1, b1, c1 = data[i,j]
              a2, b2, c2 = self.dataset[i,j]
              total      = total + (1/3)*( abs(a1 - a2) + abs(b1 - b2) + abs(c1 - c2))
        return total

    ################################################################################
    
    # Function: U-NSGA III - Algorithm
    def run_(self):
        self.solution = unified_non_dominated_sorting_genetic_algorithm_III(references        = self.size, 
                                                                            mutation_rate     = self.m_r, 
                                                                            min_values        = self.minv, 
                                                                            max_values        = self.maxv, 
                                                                            list_of_functions = self.lof, 
                                                                            generations       = self.gen, 
                                                                            mu                = self.mu, 
                                                                            eta               = self.eta, 
                                                                            k                 = self.k,
                                                                            rp                = 1
                                                                            )
        return self.solution

    ################################################################################

    # Function: Get Solutions
    def get_solution(self):
        self.indicators = []
        self.solutions  = []
        category        = 'inconsistent'
        count_con       = 0
        count_inc       = 0
        for idx in range(0, self.solution.shape[0]):
          sol = self.solution[idx, 0:int( ( (self.dataset.shape[0]**2 - self.dataset.shape[0])/2 ) + 1)]
          f0_ = round(self.f0(sol),  3)
          if ('f1' in self.qidx):
            f1_ = int(self.f1(sol))
          else:
            f1_ = '-//-'
          if ('f2' in self.qidx):
            f2_ = round(-self.f2(sol),  3)
          else:
            f2_ = '-//-'
          if ('f3' in self.qidx):
            f3_ = round( self.f3(sol),  5)
          else:
            f3_ = '-//-'
          if ('f4' in self.qidx):
            f4_ = round( self.f4(sol),  3)
          else:
            f4_ = '-//-'
          if (f0_ > 0.1):
            category  = 'inconsistent'
            count_inc = count_inc + 1
          else:
            category  = 'consistent'
            count_con = count_con + 1
          self.indicators.append((f0_, f1_, f2_, f3_, f4_, category))
          self.solutions.append(sol)
        self.indicators = pd.DataFrame(self.indicators, columns = ['f0(MI)', 'f1(NC)', 'f2(KT)', 'f3(WA)', 'f4(LM)', 'Consistency'])
        self.ind_con    = self.indicators[self.indicators['Consistency'] =='consistent']
        self.ind_incon  = self.indicators[self.indicators['Consistency'] =='inconsistent']
        if (len(self.ind_con) > 0):
          self.ind_con    = self.ind_con.drop_duplicates()
          self.ind_con    = self.ind_con.drop(columns = self.ind_con.columns[(self.ind_con == '-//-').any()])
        if (len(self.ind_incon) > 0):
          self.ind_incon  = self.ind_incon.drop_duplicates()
          self.ind_incon  = self.ind_incon.drop(columns = self.ind_incon.columns[(self.ind_incon == '-//-').any()])
        self.indicators = self.indicators.drop(columns = self.indicators.columns[(self.indicators == '-//-').any()])
        print('Total Number of Inconsistent Solutions: ', count_inc)
        print('Total Number of Consistent Solutions: '  , count_con)
        print('Total Number of Unique Consistent Solutions: ', self.ind_con.shape[0])
        return

    ################################################################################
    
    # Function: Check PCM
    def check_adj_pcm(self, idx, plot_size = 10):
        self.policy, self.dataset_adj = self.convert_solution(self.solutions[idx])
        self.plot_environment(self.dataset_adj, plot_size = plot_size, policy = self.policy)
        if (self.f_flag == False):
          self.weights_adj, self.rc_adj = ahp_method(self.dataset_adj, wd = self.wd)
        else:
          self.weights_adj, self.rc_adj = fuzzy_ahp_method(self.dataset_adj)
        for i in range(0, self.weights_adj.shape[0]):
          print('w_adj(g'+str(i+1)+'): ', round(self.weights_adj[i], 3),' w(g'+str(i+1)+'): ', round(self.weights[i], 3))
        print('')
        if (self.rc_adj > 0.100000):
          print('RC: ' + str(round(self.rc_adj, 3)), ' The solution is inconsistent, the pairwise comparisons must be reviewed')
        else:
          print('RC: ' + str(round(self.rc_adj, 3)), ' The solution is consistent')
        return

    ################################################################################
