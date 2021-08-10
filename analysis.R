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