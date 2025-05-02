library(dndscv)
data <- read.csv('./dndscv_inp.csv')
dndsout <- dndscv(data)
write.csv(dndsout$sel_cv, 'dndscv_out.csv')