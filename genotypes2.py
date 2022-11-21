from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# PRIMITIVES = [
#     'non_bottleneck_1d_1',
#     'non_bottleneck_1d_2',
#     'non_bottleneck_1d_4',
#     'non_bottleneck_1d_8',
#     'non_bottleneck_1d_16',
# ]

PRIMITIVES = [
    'non_bottleneck_1d_r1',
    'non_bottleneck_1d_r2',
    'non_bottleneck_1d_r3',
    'non_bottleneck_1d_r4',
    'non_bottleneck_1d_r5',
    'non_bottleneck_1d_r1_p2',
    'non_bottleneck_1d_r2_p2',
    'non_bottleneck_1d_r3_p2',
    'non_bottleneck_1d_r4_p2',
    'non_bottleneck_1d_r5_p2',
    # 'non_bottleneck_1d_r1_p3',
    # 'non_bottleneck_1d_r2_p3',
    # 'non_bottleneck_1d_r3_p3',
    # 'non_bottleneck_1d_r4_p3',
    # 'non_bottleneck_1d_r5_p3',
    # 'non_bottleneck_1d_r1_p4',
    # 'non_bottleneck_1d_r2_p4',
    # 'non_bottleneck_1d_r3_p4',
    # 'non_bottleneck_1d_r4_p4',
    # 'non_bottleneck_1d_r5_p4',
]
# 124(63.95)
# 123(66.70)(65.59/7.6G)(66.47/8.35G)(65.55/7.9G+merge)
# 1234(64.37)(65.35/7.49G)

# 12(67.34/8.16G+30/9_3)(66.93/7.28G/9_4)-39325
# 12(66.40/7.82G+20) (64.95/6.97G+40)

# +spatial(12)  30+30+1+11G(68.06/10.55G-search9_6)(8-1)
# 12(65.96/10.7G/30+floss1)
# 12(65.08/8.15G/30+floss1+7.8G)
# 12(66.75/7.8G/30+merge30+floss1+7.8G)
# 12(65.28/7.8G/30+merge30+floss0.5+7.8G)
# 12(66.35/7.8G/15+merge30+floss1+7.8G)
# 12(65.82/7.4G/40+merge30+floss1+7.8G)
# 12(63.79/7.8G/25+merge30+floss1+7.8G)
# 12(65.71/7.8G/30+merge20+floss1+7.8G)
# 12(65.39/7.7G/30+merge40+floss1+7.8G)

# 12(64.27/7.5G/30+merge30+floss1(> 1)+7.8G)
# 12(65.15/7.2G/30+30+floss1(e_flops < target_flops)+7.8G)
# 12(65.71/7.8G/30+merge30+floss1(2*)+7.8G)
# 12(64.09/7.7G/30+merge30+floss1(l1)+7.8G)
# 12(66.92/9.1G/30+merge30+floss1(l2/0.7)+7.8G)
# 12(64.93/8.6G/30+merge30+floss1(l2/0.6)+7.8G)

# (66.86) +bn+diff(66.05)
# CHANNELS = [8/16, 9/16, 10/16, 11/16, 12/16, 13/16, 14/16, 15/16, 16/16,]

# # (65.32)
# CHANNELS = [4/8, 5/8, 6/8, 7/8, 8/8]

# # (66.87) +bn(67.01) +bn+mid(66.32) +bn+mid+last(66.23)
# CHANNELS = [32, 24, 16, 8, 0]

# max：64to96, 128to160(68.39)  diff(68.96)(68.04)
# max：64, 128   diff(66.22)
# CHANNELS = [64, 56, 48, 40, 32, 24, 16, 8, 0]
# max：64, 128   diff(66.50/66.51)
CHANNELS = [32, 28, 24, 20, 16, 12, 8, 4, 0]

# # (65.55)
# CHANNELS = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8]

# # (67.05)  max：64 to 96, 128 to 192
# CHANNELS = [4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 12/12]
#
# # (68.13)  max：64 to 96, 128 to 192
# CHANNELS = [8/24, 9/24, 10/24, 11/24, 12/24, 13/24, 14/24, 15/24, 16/24,
#             17/24, 18/24, 19/24, 20/24, 21/24, 22/24, 23/24, 24/24]
