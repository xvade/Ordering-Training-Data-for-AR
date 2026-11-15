# Per-checkpoint summary

step                          1404      2808       4914
epoch                            4         8         14
lr                         0.04699   0.03504    0.01133
eta                        0.01175  0.008761   0.002832
ref_step_len               0.01825   0.01311   0.005384
val_clean_acc               0.5713     0.695       0.78
val_adv_acc@1               0.4623    0.4717     0.4763
val_adv_acc@2               0.3493    0.2763     0.2193
val_rgc@1                   0.8092    0.6787     0.6107
val_rgc@2                   0.6114    0.3976     0.2812
val_lin_radius               2.191     1.407      1.209
val_gap                     0.7103     1.514      2.465
norm_g_clean                 1.208     1.029     0.9246
norm_g_adv                   2.032     2.783      3.718
cos_clean_adv               0.8791    0.7208     0.5196
cos_probe_val_adv            0.986    0.9764     0.9517
cos_probe_val_clean         0.9746    0.9293      0.806
sin_clean_adv               0.4765    0.6931     0.8544
cos_r_probe_val             0.9786    0.9633      0.932
beta_val                     1.549     2.088      2.035
kappa_acc                    0.634   -0.8469    -0.1002
frac_hurt_adv               0.4264    0.4644     0.4758
frac_hurt_clean             0.4278    0.4906      0.526
frac_help_clean_hurt_adv   0.04678   0.06571    0.05887
frac_widen_gap              0.4773    0.4756     0.4814
spearman_gain_adv_clean     0.9218    0.8263     0.7565
mean_gain_adv                1.827     1.705      1.547
mean_gain_clean              1.234    0.8329     0.6445
mean_gain_gap               0.5932     0.872     0.9025
class_eta2_clean            0.3069    0.3331     0.3336
class_eta2_rob              0.3765     0.288     0.1865
class_eta2_linr_rob         0.3743    0.3324     0.3381
class_eta2_serr_rob         0.3118    0.2865     0.2594
harm_share_top5pct          0.4953    0.5134     0.5742
gap_harm_share_top5pct      0.4243    0.5025     0.5767
r2_adv_x0.25                0.9942    0.9764     0.8456
r2_adv_x1                   0.9913    0.9734      0.927
r2_adv_x4                   0.5961    0.6666     0.8743
slope_adv_x0.25              1.064      1.14      1.104
slope_adv_x1                0.9006     1.007      1.196
slope_adv_x4                0.4631    0.6486     0.9599
r2_clean_x1                 0.9718    0.9869     0.9963
r2_adv_probe_to_val          0.984    0.9394     0.8272
rand_mean_d_clean_loss    -0.01181 -0.006224   -0.00214
rand_mean_d_adv_loss@2    -0.01957  -0.01463  -0.005813
rand_mean_d_gap          -0.007768 -0.008406  -0.003672
rand_mean_d_clean_acc     0.005983  0.001756     0.0012
rand_mean_d_adv_acc@2     0.004261 -0.002017 -0.0002111
rand_mean_d_lin_radius    0.002371 -0.000447 -2.773e-05
rand_std_d_clean_loss     0.002797  0.002341   0.001059
rand_std_d_adv_loss@2     0.006167  0.006548   0.003243
rand_std_d_gap            0.004044  0.004765   0.002586
rand_std_d_clean_acc      0.002398  0.001452  0.0006532
rand_std_d_adv_acc@2      0.001499  0.001443  0.0008299
rand_std_d_lin_radius     0.005068  0.002674    0.00133
noise_seed_std_adv_loss  0.0001324 0.0002519  0.0004219
noise_batch_std_adv_loss  0.006527  0.006553   0.003761
noise_seed_std_adv_acc   0.0003956  0.000866  0.0009134


# Filter trials: mean paired difference vs random filtering of the same pool
(`random` rows show the absolute effect of a random batch. z = mean / standard error.)


## step length equalised (direction only): change in `rob_loss` (rows: rule, columns: epoch) and z-scores

epoch                  4         8          14
rule                                          
cos_rob         -0.008772  -0.01094  -0.005953
gain_adv        -0.004412 -0.003454  -0.002703
gain_clean      -0.002301 0.0004862 -0.0003736
gain_gap        -0.007222 -0.006798  -0.004325
gain_linr        0.004523  0.006003   0.002629
gain_linr_rob     0.01065  0.007717   0.002708
gain_rob        -0.008929  -0.01105  -0.006038
gain_serr_adv   -0.004596 -0.006589  -0.004636
gain_serr_rob   -0.008184  -0.01011  -0.005722
high_adv_margin  0.007532   0.00738   0.003147
high_lin_radius  0.007282  0.007846   0.004433
high_loss       -0.002017 -0.001499 -0.0008743
low_loss         0.009074   0.01024   0.005367
random            -0.0027 -0.003098  -0.002385

z:
epoch              4     8     14
rule                             
cos_rob         -11.5 -15.5   -13
gain_adv         -6.3  -5.8  -6.1
gain_clean       -3.8   0.8  -0.8
gain_gap         -9.4  -8.5    -9
gain_linr         6.2   6.7   4.7
gain_linr_rob    13.8   8.9   4.7
gain_rob        -11.5 -15.7   -13
gain_serr_adv    -6.4  -8.2 -10.1
gain_serr_rob   -10.7 -14.9 -14.1
high_adv_margin   6.4   6.8   5.3
high_lin_radius   5.6   5.9   5.1
high_loss        -3.4  -2.7  -1.8
low_loss          9.1   9.5   7.1
random           -3.4  -4.2  -5.2

## step length equalised (direction only): change in `rob_acc` (rows: rule, columns: epoch) and z-scores

epoch                   4          8          14
rule                                            
cos_rob          0.0003775  0.0003653 -0.0007934
gain_adv         -0.001332 -0.0001723 -0.0002311
gain_clean       -0.001509  -0.000252 -2.681e-05
gain_gap        -0.0004206 -0.0003043   -0.00048
gain_linr        -0.001198   0.001135 -0.0004113
gain_linr_rob       0.0024    0.00115 -0.0003202
gain_rob         0.0003456  0.0001567 -0.0007645
gain_serr_adv    -0.001864 -0.0004106 -5.785e-05
gain_serr_rob    0.0005126 -0.0002984  0.0005332
high_adv_margin   0.002248 -0.0008103  0.0008487
high_lin_radius   0.002388 -0.0001655  0.0005131
high_loss        -0.001106 -0.0003161   -0.00024
low_loss           0.00268  0.0001869  0.0007375
random           0.0009744 -0.0001713  5.362e-05

z:
epoch             4    8    14
rule                          
cos_rob          0.7  0.8 -3.3
gain_adv        -2.3 -0.4 -0.8
gain_clean      -2.6 -0.5 -0.1
gain_gap        -0.7 -0.7 -1.5
gain_linr       -2.1  2.2 -1.5
gain_linr_rob      4  2.1 -1.2
gain_rob         0.7  0.4 -3.2
gain_serr_adv   -3.2   -1 -0.2
gain_serr_rob    0.9 -0.8  1.8
high_adv_margin  2.8 -1.3  3.1
high_lin_radius  2.6 -0.3  1.8
high_loss       -1.9 -0.8   -1
low_loss         4.4  0.3  2.9
random           1.9 -0.5  0.3

## step length equalised (direction only): change in `lin_radius` (rows: rule, columns: epoch) and z-scores

epoch                   4          8          14
rule                                            
cos_rob          0.0006723   0.002156   0.003729
gain_adv         -0.002668  -0.000146  0.0009085
gain_clean       -0.003766  -0.001332 -0.0005016
gain_gap         -0.002371  0.0003207   0.001937
gain_linr         0.008905   0.004009   0.003326
gain_linr_rob     0.008682   0.005728   0.003259
gain_rob         0.0007908   0.002316   0.003708
gain_serr_adv     0.004383   0.001399   0.004882
gain_serr_rob   -0.0009474   0.002181   0.005889
high_adv_margin   0.006998   0.004064   0.001525
high_lin_radius   0.009342   0.005019   0.001933
high_loss        -0.002702 -0.0007795  2.448e-06
low_loss          0.009771   0.006126   0.002646
random           0.0001127 -0.0003908 -0.0002083

z:
epoch             4    8    14
rule                          
cos_rob          0.4  3.8 12.2
gain_adv        -1.7 -0.2  2.7
gain_clean      -2.7 -1.6 -1.8
gain_gap        -1.4  0.4  6.6
gain_linr        5.5  5.7 12.5
gain_linr_rob    4.8  6.9 13.1
gain_rob         0.5  4.3 11.9
gain_serr_adv    2.4  2.1 13.2
gain_serr_rob   -0.6  4.3 15.8
high_adv_margin  2.7  4.3  4.5
high_lin_radius  4.4  5.6  5.2
high_loss       -1.3 -1.2    0
low_loss         5.2  6.3    9
random           0.1 -0.5 -0.9

## step length equalised (direction only): change in `clean_loss` (rows: rule, columns: epoch) and z-scores

epoch                  4          8          14
rule                                           
cos_rob          0.002601   0.001414  0.0006889
gain_adv        -0.001622  -0.002463   -0.00138
gain_clean      -0.002538  -0.002639  -0.001746
gain_gap        0.0005707  -0.001582 -0.0007278
gain_linr        0.000544   0.003407      0.003
gain_linr_rob     0.01481   0.006416   0.002806
gain_rob          0.00274   0.001609  0.0007146
gain_serr_adv   -0.000739 -0.0008372   0.001019
gain_serr_rob    0.002048   0.001232   0.002191
high_adv_margin   0.01468   0.008523   0.003015
high_lin_radius   0.01269    0.00848   0.003131
high_loss       -0.001864  -0.001459 -0.0007803
low_loss          0.01314   0.008034   0.003973
random           -0.01184  -0.006468  -0.002442

z:
epoch              4     8    14
rule                            
cos_rob           5.3   3.4    3
gain_adv         -3.2  -5.4 -6.1
gain_clean       -4.8  -5.3 -6.8
gain_gap          1.1  -3.8 -3.6
gain_linr         1.1   6.7 12.3
gain_linr_rob    18.9  10.8 11.5
gain_rob          5.6   3.7  3.1
gain_serr_adv    -1.5  -2.1  3.8
gain_serr_rob     4.1   2.9  9.5
high_adv_margin  17.2  13.6  8.7
high_lin_radius  10.9  13.2 10.2
high_loss        -4.1  -3.4 -3.6
low_loss         11.7  13.9 10.9
random          -20.4 -12.3 -9.3

## step length equalised (direction only): change in `clean_acc` (rows: rule, columns: epoch) and z-scores

epoch                   4          8          14
rule                                            
cos_rob          -0.001822 -0.0006444 -0.0003778
gain_adv         0.0004889     0.0004 -8.889e-05
gain_clean       0.0007333   0.001356 -0.0007111
gain_gap        -0.0001778 -4.444e-05  8.889e-05
gain_linr        6.667e-05 -0.0001555     -0.001
gain_linr_rob    -0.007044 -0.0006889 -0.0009778
gain_rob         -0.001667 -0.0007333 -0.0003111
gain_serr_adv    0.0009778 -0.0005111 -0.0003555
gain_serr_rob    -0.001089  -0.001244 -0.0008889
high_adv_margin  -0.008733    -0.0024 -0.0008444
high_lin_radius  -0.007622  -0.001822  -0.001089
high_loss         0.001289  0.0001778  4.445e-05
low_loss         -0.007978  -0.001511  -0.001511
random            0.006244   0.001844   0.001422

z:
epoch              4    8    14
rule                           
cos_rob          -2.5   -2   -2
gain_adv          0.7  1.6 -0.5
gain_clean        1.1  4.7 -3.7
gain_gap         -0.2 -0.1  0.5
gain_linr         0.1 -0.5 -3.6
gain_linr_rob    -9.7 -2.2 -3.8
gain_rob         -2.3 -2.6 -1.6
gain_serr_adv     1.4 -1.7 -3.1
gain_serr_rob    -1.4 -4.7 -4.9
high_adv_margin -12.1 -6.3 -3.2
high_lin_radius  -8.4 -6.7   -5
high_loss         2.1  0.7  0.2
low_loss        -10.3 -4.1 -5.3
random            9.3  6.4  7.4

## step length equalised (direction only): change in `adv_loss@2` (rows: rule, columns: epoch) and z-scores

epoch                  4         8         14
rule                                         
cos_rob         -0.004745 -0.007987  -0.00455
gain_adv        -0.006924 -0.008597 -0.005511
gain_clean      -0.006232 -0.005026 -0.003926
gain_gap        -0.006339   -0.0101 -0.005806
gain_linr        0.005366   0.01312  0.008736
gain_linr_rob     0.03358   0.02111  0.008419
gain_rob        -0.004687  -0.00769 -0.004583
gain_serr_adv    -0.00574 -0.008337 -0.002561
gain_serr_rob   -0.005013 -0.007533 -0.001262
high_adv_margin   0.03027   0.02518  0.009284
high_lin_radius   0.02693   0.02555    0.0108
high_loss       -0.004903 -0.004547 -0.002462
low_loss          0.02942   0.02702   0.01345
random           -0.02103  -0.01661 -0.007356

z:
epoch              4     8     14
rule                             
cos_rob            -5 -10.6  -7.1
gain_adv         -6.9  -8.3  -7.5
gain_clean       -6.4  -4.7  -4.9
gain_gap         -6.4 -10.3  -9.1
gain_linr         5.8   8.8  10.6
gain_linr_rob    23.3  12.5  10.6
gain_rob           -5  -9.8    -7
gain_serr_adv      -6  -9.3  -3.8
gain_serr_rob    -5.3  -9.6  -2.1
high_adv_margin  16.1  19.5   9.7
high_lin_radius   9.9  13.6   8.8
high_loss          -6  -4.7  -3.3
low_loss         13.9  15.3    10
random          -20.7 -15.7 -10.1

## step length equalised (direction only): change in `adv_acc@2` (rows: rule, columns: epoch) and z-scores

epoch                   4          8          14
rule                                            
cos_rob         -0.0007778  0.0009111 -0.0007556
gain_adv         -0.001022 -0.0005111 -0.0002222
gain_clean       -0.001044    -0.0014  4.444e-05
gain_gap        -0.0005333 -0.0002667 -0.0004889
gain_linr        -0.001156   0.001267 -0.0003111
gain_linr_rob    -0.002067   0.001733 -0.0002222
gain_rob        -0.0007111  0.0007778 -0.0007333
gain_serr_adv    -0.001244  2.222e-05 -2.222e-05
gain_serr_rob   -0.0001778  0.0007556  0.0006222
high_adv_margin  -0.003289   0.001222  0.0009333
high_lin_radius  -0.002444   0.001378  0.0006222
high_loss       -0.0002889 -0.0004667 -0.0002444
low_loss         -0.002378   0.001467  0.0008889
random            0.004933  -0.001733 -8.889e-05

z:
epoch             4    8    14
rule                          
cos_rob           -3  2.7 -3.1
gain_adv        -4.1 -1.8 -0.8
gain_clean      -3.7 -4.1  0.2
gain_gap        -1.6 -0.9 -1.5
gain_linr       -2.7  3.6 -1.2
gain_linr_rob   -4.8  4.7 -0.8
gain_rob        -3.3    3   -3
gain_serr_adv   -4.3  0.1 -0.1
gain_serr_rob     -1  2.5  2.1
high_adv_margin   -5  2.8  3.4
high_lin_radius   -3  2.8  2.1
high_loss       -0.9 -1.4   -1
low_loss        -4.6  3.5  3.3
random          20.6 -8.1 -0.5

## raw step (eta * mean gradient): change in `rob_loss` (rows: rule, columns: epoch) and z-scores

epoch                  4         8         14
rule                                         
cos_rob           -0.0253  -0.02364   -0.0115
gain_adv         -0.01728  -0.01358 -0.007314
gain_clean       -0.01172 -0.004335 -0.002518
gain_gap         -0.02387  -0.01979  -0.01005
gain_linr        0.004961  0.006671  0.002084
gain_linr_rob     0.01486  0.009039  0.002241
gain_rob         -0.02531  -0.02379  -0.01166
gain_serr_adv    -0.01823  -0.01939 -0.009502
gain_serr_rob    -0.02407  -0.02312  -0.01014
high_adv_margin  0.004872  0.004698  0.002342
high_lin_radius  0.005007  0.004885  0.002502
high_loss       -0.006055 -0.004926 -0.002723
low_loss         0.005924  0.005704    0.0028
random           -0.00273 -0.003057 -0.002094

z:
epoch              4     8     14
rule                             
cos_rob         -35.2 -32.3 -29.5
gain_adv        -21.5 -19.2 -18.2
gain_clean      -16.2  -4.6  -4.4
gain_gap        -35.1 -23.2 -18.7
gain_linr           6   5.5   3.3
gain_linr_rob    15.9   8.5   3.3
gain_rob        -35.7 -29.7 -27.4
gain_serr_adv   -25.7 -23.1   -18
gain_serr_rob   -35.8 -30.2 -23.5
high_adv_margin   5.9   5.8   5.6
high_lin_radius   5.2   5.4   4.9
high_loss          -6  -6.2    -5
low_loss          7.5   6.8   5.1
random           -3.5    -4  -4.4

## raw step (eta * mean gradient): change in `rob_acc` (rows: rule, columns: epoch) and z-scores

epoch                   4          8          14
rule                                            
cos_rob          0.0001648  0.0008487    -0.0017
gain_adv         -0.003143 -0.0001095  -0.003011
gain_clean       -0.003994    0.00137   -0.00212
gain_gap         -0.000914  -0.001629  -0.003009
gain_linr       -0.0006768   0.001088 -0.0002291
gain_linr_rob     0.002384  0.0005488 -5.571e-05
gain_rob         0.0007577  0.0005442  -0.001467
gain_serr_adv    -0.001963  -0.001952  -0.001309
gain_serr_rob   -0.0006897 -0.0005427  0.0008954
high_adv_margin  0.0009416 -0.0004616  0.0001086
high_lin_radius   0.000707 -0.0006378  4.195e-05
high_loss        -0.002485  0.0003305 -0.0006177
low_loss         0.0006213 -0.0004019  0.0001108
random           0.0007115 -0.0002311   4.72e-06

z:
epoch             4    8    14
rule                          
cos_rob          0.2  1.4 -5.6
gain_adv        -6.4 -0.2 -8.5
gain_clean      -7.5  2.6 -7.5
gain_gap        -1.5 -2.8  -10
gain_linr       -1.1  2.1 -0.8
gain_linr_rob    4.3    1 -0.2
gain_rob           1  1.1 -5.1
gain_serr_adv   -3.1 -3.2   -4
gain_serr_rob   -1.3 -0.9  4.1
high_adv_margin  1.5 -1.3  0.5
high_lin_radius  1.1 -1.4  0.2
high_loss       -3.9  0.7 -2.3
low_loss         1.3 -0.9  0.5
random           1.3 -0.6    0

## raw step (eta * mean gradient): change in `lin_radius` (rows: rule, columns: epoch) and z-scores

epoch                   4          8          14
rule                                            
cos_rob           -0.03052   0.005461   0.004702
gain_adv          -0.04878  -0.003023 -0.0005236
gain_clean        -0.04854  -0.008421   -0.00363
gain_gap          -0.04623  0.0001234   0.001451
gain_linr          0.01083   0.006042   0.004513
gain_linr_rob      0.01546   0.006695   0.004416
gain_rob          -0.02981   0.005308   0.004917
gain_serr_adv     -0.03646   0.004034   0.006628
gain_serr_rob     -0.03228    0.00645   0.007564
high_adv_margin    0.00344    0.00117  0.0004508
high_lin_radius   0.005921   0.001546  0.0004049
high_loss         -0.01174  -0.001966 -0.0002773
low_loss          0.005361   0.002089  0.0005783
random          -0.0001018 -0.0006191 -0.0002474

z:
epoch              4    8    14
rule                           
cos_rob          -6.3    9 16.6
gain_adv        -13.8 -1.9 -1.3
gain_clean      -15.2 -5.8 -8.1
gain_gap        -10.8  0.1  3.5
gain_linr         5.9  7.9 11.7
gain_linr_rob     7.7    9   12
gain_rob         -6.3  6.9 18.6
gain_serr_adv    -9.4  3.8   23
gain_serr_rob    -7.5  8.2 23.3
high_adv_margin   1.9  1.8  2.1
high_lin_radius   4.2  2.3    2
high_loss        -4.7 -2.2 -0.6
low_loss          3.4  3.3  2.7
random           -0.1 -0.9 -1.2

## raw step (eta * mean gradient): change in `clean_loss` (rows: rule, columns: epoch) and z-scores

epoch                  4          8          14
rule                                           
cos_rob           0.00149 -0.0009243 -0.0001298
gain_adv        -0.008352   -0.01001  -0.004541
gain_clean       -0.01166   -0.01058  -0.005242
gain_gap        -0.002291  -0.007675  -0.002993
gain_linr       -0.006328   0.002669   0.003239
gain_linr_rob     0.01845   0.007087   0.002985
gain_rob         0.001684 -0.0005058 -0.0001005
gain_serr_adv    -0.00591  -0.005887  0.0004066
gain_serr_rob    0.000253  -0.001318   0.002221
high_adv_margin   0.01244   0.007166   0.002497
high_lin_radius   0.01159   0.007142    0.00249
high_loss       -0.008006   -0.00549  -0.002374
low_loss          0.01189   0.006917   0.002625
random           -0.01148  -0.006522  -0.002394

z:
epoch              4     8     14
rule                             
cos_rob           1.7  -2.1  -0.4
gain_adv         -9.8 -19.3 -19.7
gain_clean      -13.3 -19.9 -22.5
gain_gap         -2.6   -19 -13.9
gain_linr        -9.3   4.3  12.6
gain_linr_rob    14.9  11.5  11.3
gain_rob            2  -1.1  -0.4
gain_serr_adv    -6.7 -13.4   1.2
gain_serr_rob     0.3  -2.8   7.3
high_adv_margin  20.1  12.7   8.3
high_lin_radius  17.1  12.2   8.6
high_loss       -14.3  -9.8  -8.4
low_loss           18  11.8   8.6
random          -14.3 -10.4    -8

## raw step (eta * mean gradient): change in `clean_acc` (rows: rule, columns: epoch) and z-scores

epoch                  4         8          14
rule                                          
cos_rob           -0.0038  0.001133  0.0003333
gain_adv         0.002644  0.004489      0.001
gain_clean       0.004267  0.006289 -8.888e-05
gain_gap        -0.001222  0.002222     0.0008
gain_linr        0.005133  0.001022 -0.0009556
gain_linr_rob   -0.005933 4.445e-05 -0.0007778
gain_rob        -0.004244    0.0008  0.0002222
gain_serr_adv    0.002711  0.001867  0.0002444
gain_serr_rob   -0.002067  0.001222    -0.0006
high_adv_margin -0.006778 -0.002356  -0.001356
high_lin_radius -0.006022 -0.002511  -0.001356
high_loss        0.002622  0.003067  0.0002667
low_loss        -0.006378   -0.0026  -0.001333
random           0.006133    0.0018   0.001378

z:
epoch              4    8    14
rule                           
cos_rob          -3.4  2.6  1.9
gain_adv          4.9 11.5  4.3
gain_clean        6.3 11.6 -0.4
gain_gap         -1.3  4.9  2.8
gain_linr         6.6  2.7 -3.3
gain_linr_rob    -8.4  0.1 -2.7
gain_rob           -4    2  1.3
gain_serr_adv     4.7  4.3  1.3
gain_serr_rob    -2.3  2.6 -2.5
high_adv_margin -10.2 -7.2 -8.4
high_lin_radius  -8.8 -6.9   -9
high_loss         3.2  8.7  1.2
low_loss         -9.2 -7.5 -9.4
random            8.7  4.9  8.5

## raw step (eta * mean gradient): change in `adv_loss@2` (rows: rule, columns: epoch) and z-scores

epoch                  4        8         14
rule                                        
cos_rob          -0.02299 -0.02557  -0.01177
gain_adv         -0.03022 -0.03448  -0.01656
gain_clean       -0.02978 -0.02642  -0.01319
gain_gap         -0.02741 -0.03582  -0.01614
gain_linr       -0.004838  0.01225  0.008677
gain_linr_rob     0.04343  0.02384  0.008316
gain_rob          -0.0227 -0.02485  -0.01186
gain_serr_adv    -0.02738 -0.03168 -0.008674
gain_serr_rob    -0.02368 -0.02587  -0.00562
high_adv_margin   0.02414  0.01966  0.007424
high_lin_radius   0.02295   0.0198  0.007569
high_loss        -0.01845 -0.01639 -0.007556
low_loss          0.02433  0.02015  0.008143
random           -0.02051 -0.01668 -0.006967

z:
epoch              4     8     14
rule                             
cos_rob         -16.4   -25 -15.2
gain_adv        -21.5 -30.7 -24.5
gain_clean      -20.5 -20.2 -15.1
gain_gap        -19.6 -41.4 -23.1
gain_linr        -3.3     6  10.4
gain_linr_rob    17.5  12.4   9.8
gain_rob        -16.4 -25.2 -15.1
gain_serr_adv     -18 -35.4  -9.1
gain_serr_rob   -16.9 -23.9    -7
high_adv_margin    18  18.7   9.3
high_lin_radius  13.4  14.7   9.4
high_loss       -12.1 -10.4  -8.2
low_loss         17.7  14.2   8.7
random          -12.9 -13.2  -8.4

## raw step (eta * mean gradient): change in `adv_acc@2` (rows: rule, columns: epoch) and z-scores

epoch                   4          8          14
rule                                            
cos_rob          -0.002244 -0.0001111  -0.001733
gain_adv         -0.001467  -0.003911  -0.003111
gain_clean       -0.001289  -0.003956  -0.002111
gain_gap         -0.001689  -0.003511  -0.003089
gain_linr         0.002578  0.0002222 -0.0001333
gain_linr_rob    -0.001378  0.0005111  2.222e-05
gain_rob         -0.001933 -0.0001333  -0.001489
gain_serr_adv   -0.0002444  -0.003533  -0.001333
gain_serr_rob       -0.002  -0.001578  0.0009556
high_adv_margin  -0.003356   0.001533  0.0002444
high_lin_radius  -0.003111   0.001489  0.0001778
high_loss       -0.0008222  -0.002267 -0.0006444
low_loss         -0.003422     0.0018  0.0002444
random              0.0046  -0.001756 -0.0001333

z:
epoch             4     8     14
rule                            
cos_rob         -5.4  -0.3  -5.8
gain_adv        -3.9  -8.4  -8.8
gain_clean      -2.9 -11.2  -7.5
gain_gap        -4.4    -8 -10.6
gain_linr        5.1   0.7  -0.5
gain_linr_rob     -3   1.7   0.1
gain_rob        -4.6  -0.5  -5.3
gain_serr_adv   -0.4  -7.1  -4.1
gain_serr_rob     -4  -4.4   4.4
high_adv_margin -6.8     5   1.2
high_lin_radius -5.7   4.7   0.9
high_loss         -2  -5.2  -2.4
low_loss        -6.5     5   1.1
random          15.6  -7.1  -0.7


# Spearman(feature, first-order gain), rows: feature, columns: step


## target: gain `adv`
step             1404     2808     4914
feature                                
adv_grad_norm  0.3094   0.2475   0.1556
adv_margin@1  -0.3799  -0.3146  -0.2314
adv_margin@2   -0.401  -0.3217  -0.2338
adv_margin@4  -0.4179  -0.3268  -0.2409
entropy        0.2796   0.2561   0.2112
grad_norm      0.3209    0.289   0.2171
input_grad_l1  0.3445   0.3058   0.2315
lin_radius    -0.3601  -0.3037   -0.226
loss           0.3801   0.3117   0.2302
margin        -0.3516  -0.2944  -0.2211
p_true        -0.3801  -0.3117  -0.2302
self_cos      -0.1691 -0.08591 -0.07037

## target: gain `rob`
step              1404     2808     4914
feature                                 
adv_grad_norm    0.031  0.02085  0.03642
adv_margin@1   -0.1118 -0.07004 -0.02991
adv_margin@2   -0.1039 -0.05694 -0.02204
adv_margin@4   -0.1051 -0.05888 -0.03016
entropy        0.03286  0.01837  0.04106
grad_norm      0.00695  0.02144  0.01958
input_grad_l1 0.006357 0.007642  0.01253
lin_radius    -0.09111  -0.0781 -0.03697
loss            0.1084  0.07868   0.0481
margin         -0.1258 -0.09541 -0.05268
p_true         -0.1084 -0.07868  -0.0481
self_cos      -0.06769 -0.01663 -0.02323

## target: gain `cos_rob`
step             1404     2808     4914
feature                                
adv_grad_norm  0.1072   0.1184  0.08033
adv_margin@1  -0.2149  -0.1877 -0.08826
adv_margin@2  -0.2022  -0.1726 -0.08018
adv_margin@4  -0.1904  -0.1651 -0.09187
entropy        0.1438   0.1456  0.09707
grad_norm      0.1044   0.1413  0.07744
input_grad_l1 0.09609   0.1285  0.07341
lin_radius    -0.1925  -0.1903 -0.08761
loss           0.2135   0.1956    0.103
margin        -0.2324  -0.2105  -0.1089
p_true        -0.2135  -0.1956   -0.103
self_cos       -0.083 -0.04994 -0.05049

## target: gain `serr_rob`
step              1404     2808      4914
feature                                  
adv_grad_norm   0.0796  0.05335 -0.007235
adv_margin@1   -0.1618 -0.09536   0.04436
adv_margin@2   -0.1574 -0.08585   0.05346
adv_margin@4   -0.1587 -0.09024   0.03841
entropy        0.08138  0.05082  -0.01571
grad_norm       0.0572  0.04896  -0.04713
input_grad_l1  0.06027  0.03991  -0.05887
lin_radius     -0.1552  -0.1043   0.03735
loss            0.1605   0.1024  -0.02159
margin          -0.171  -0.1151   0.01183
p_true         -0.1605  -0.1024   0.02159
self_cos      -0.09017  -0.0174  0.009519

## target: gain `linr`
step              1404     2808     4914
feature                                 
adv_grad_norm   0.1709  0.08069  0.06575
adv_margin@1  -0.04139  0.08553  0.05479
adv_margin@2  -0.07504  0.05893  0.04591
adv_margin@4   -0.1122  0.01522 0.006749
entropy        0.06172 -0.04418  -0.0241
grad_norm       0.1534 -0.02527 -0.02839
input_grad_l1    0.175 -0.01937 -0.03487
lin_radius    -0.01617   0.1082    0.063
loss           0.02958  -0.1105 -0.05754
margin        0.004388   0.1201   0.0537
p_true        -0.02958   0.1105  0.05754
self_cos       0.03256  0.09996   0.0696

## target: gain `linr_rob`
step              1404     2808      4914
feature                                  
adv_grad_norm -0.01663  0.02502   0.07304
adv_margin@1    0.2178   0.1593   0.04132
adv_margin@2    0.1973   0.1348   0.03282
adv_margin@4    0.1659  0.08999 -0.005395
entropy       -0.09492  -0.1093   -0.0105
grad_norm     -0.05723  -0.0961  -0.01468
input_grad_l1 -0.05881 -0.09824  -0.02037
lin_radius      0.2211   0.1783   0.04994
loss           -0.2243   -0.183   -0.0439
margin          0.2418   0.1862   0.04087
p_true          0.2243    0.183    0.0439
self_cos        0.1538   0.1194   0.06661


# Rank stability of per-sample gains between checkpoints
 step_a  step_b    adv  clean    gap    rob  cos_gap  cos_rob  serr_adv  serr_rob   linr  linr_rob
   1404    2808 0.7789 0.8135 0.7315 0.7049   0.7603   0.7432     0.719    0.6636 0.7125     0.774
   1404    4914 0.6106 0.6268 0.5383 0.4646   0.5761   0.4851    0.3038    0.3832 0.3448    0.5254
   2808    4914 0.8294 0.8089 0.7904 0.7429   0.8461   0.7911     0.546     0.648 0.5948    0.6549