require("data.table")
require("dplyr")

d1 = fread("results_with.csv")
setnames(d1, c("path", "mean", "sd", "size", "speed"))

d1$SDMeanRatio = d1$sd / d1$mean
d1[,.(AvgSDMeanRatio = mean(SDMeanRatio)), by = .(size, speed)]
#    size speed AvgSDMeanRatio
# 1:   50 0.060      0.3422847
# 2:   50 0.045      0.2654232
# 3:   50 0.075      0.2549663
# 4:  100 0.180      0.3192082
# 5:  100 0.240      0.4014214
# 6:  100 0.300      0.4323073
# 7:  200 0.720      0.4194786
# 8:  200 0.960      0.4056542
# 9:  200 1.200      0.4954646


d2 = fread("results_without.csv")
setnames(d2, c("path", "mean", "sd", "size", "speed"))

d2$SDMeanRatio = d2$sd / d2$mean
d2[,.(AvgSDMeanRatio = mean(SDMeanRatio)), by = .(size, speed)]
#    size speed AvgSDMeanRatio
# 1:   50 0.060      0.4389157
# 2:   50 0.045      0.4888785
# 3:   50 0.075      0.3681829
# 4:  100 0.180      0.5295387
# 5:  100 0.240      0.4633469
# 6:  100 0.300      0.5064373
# 7:  200 0.720      0.4496567
# 8:  200 0.960      0.6785399
# 9:  200 1.200      0.6473749