value_data <- read.delim("/Users/kevindu/Desktop/Coding/Game Theory/MS/value0.out", sep=',', header = FALSE)
policy_data <- read.delim("/Users/kevindu/Desktop/Coding/Game Theory/MS/policy0.out", sep=',', header = FALSE)
N = length(value_data)

consec_avg <- function(lst, N){
  avgs <- integer(length(lst) / N)
  for(i in 0:(length(lst) / N - 1)){
    x = 0
    count = 0
    for(j in 1:N){
      num = as.numeric(lst[i*N + j])
      x = x + num
      count = count + 1
    }
    avgs[i+1] = x / count
  }
  avgs
}

avgs = consec_avg(value_data, 10)
plot(1:length(avgs), avgs)

policy_avgs = consec_avg(policy_data, 10)
plot(1:length(policy_avgs), policy_avgs)