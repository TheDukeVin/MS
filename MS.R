
data <- read.delim("/Users/kevindu/Desktop/Coding/Game Theory/MS/session4.out", sep=' ', header = FALSE)
N = length(data[,1])

plot(1:N, data[,4])
