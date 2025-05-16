# install.packages("chron")
library(tidyverse)
library(chron)
library(fitdistrplus)

#Options from Stakahama
Sys.setlocale("LC_TIME","C")
options(stringsAsFactors=FALSE)
options(chron.year.abb=FALSE)
theme_set(theme_bw()) # just my preference for plots

wdirAW <- "C:/Users/aliwa/AirPollution/ENV-409_Project"
wdirLC <- ""
setwd(wdirAW)

# Function from STakahama
ReadTSeries <- function(filename, timecolumn="datetime", timeformat="%d.%m.%Y %H:%M") {
  ## read the table, strip units in column names, rename time column
  ##   and change data type of time column from a string of characters to
  ##   a numeric type so that we can perform operations on it
  data <- read.table(filename, skip=5, header=TRUE, sep=";", check.names=FALSE)
  names(data) <- sub("[ ].*$","",names(data)) # strip units for simplification
  names(data) <- sub("Date/time", timecolumn, names(data), fixed=TRUE)
  data[,timecolumn] <- as.chron(data[,timecolumn], timeformat) - 1/24 # end time -> start time
  ## extract additional variables from the time column
  data[,"year"] <- years(data[,timecolumn])
  data[,"month"] <- months(data[,timecolumn])
  data[,"day"] <- days(data[,timecolumn])
  data[,"hour"] <- hours(data[,timecolumn])
  data[,"dayofwk"] <- weekdays(data[,timecolumn])
  data[,"daytype"] <- ifelse(data[,"dayofwk"] %in% c("Sat","Sun"), "Weekend", "Weekday")
  data[,"season"] <- Month2Season(unclass(data[,"month"]))
  ## return value
  data
}

Month2Season <- function(month) {
  ## month is an integer (1-12)
  ## a factor with levels {"DJF", "MAM", "JJA", "SON"} is returned
  seasons <- c("DJF", "MAM", "JJA", "SON")
  index <- findInterval(month %% 12, seq(0, 12, 3))
  factor(seasons[index], seasons)
}

LUG<-ReadTSeries("./data/LUG_2023-04-16.csv")
RIG<-ReadTSeries("./data/RIG_2023-04-16.csv")


df <- full_join(cbind(site="LUG", LUG),
                cbind(site="RIG", RIG))

# This operation cuts out the unnecessary columns
reg_df <- df[, !names(df) %in% c("CPC", "EC", "PREC", "RAD", "TEMP")]

# Make the df a long format
lf <- reg_df %>%
  gather(variable, value,
         -c(site, datetime, season, year, month, day, hour, dayofwk, daytype))


daily <- lf %>%
  mutate(date = dates(datetime)) %>% # creates a date column
  group_by(site, date, variable)%>% # groups the data
  summarize(percent.recovery = length(na.omit(value))/length(value)*1e2,
            value = mean(value, na.rm=TRUE)) %>%
  ungroup()

threshold <- 75

daily %>%
  filter(percent.recovery < threshold) %>%
  count(site, variable)


ComputeStats <- function(x) {
  x <- na.omit(x)
  m <- mean(x)
  s <- sd(x)
  n <- length(x)
  t <- qt(.975, n - 1)
  data.frame(mean=m,
             conf.lower=m-t*s/sqrt(n),
             conf.upper=m+t*s/sqrt(n),
             sd.lower=m-s,
             sd.upper=m+s)
}


july.hour <- lf %>%
        filter(month =="Jul")

july.day <- daily %>% mutate(month = months(date))%>%
        filter( month =="Jul")

ReplaceNonpositive <- function(x) {
  x <- na.omit(x)
  min.positive.value <- min(x[x>0])
  replace(x, x < min.positive.value, min.positive.value)
}

ReplaceNonpositive.log <- function(x) {
  x <- na.omit(x)
  min.value <- min(x)
  if (min.value<0){
    ret <- x-min.value+0.00001
  } else{
    ret <-x
  }
  ret
}




pollutants <- unique(july.hour$variable)
pollutants <- pollutants[pollutants != "NMVOC"]

# concvec <- c(na.omit(filter(july.hour,variable == "SO2")[["value"]]))
# fit <- fitdist(ReplaceNonpositive(concvec), "lnorm")
# fit_stats <- gofstat(fit)
# print(fit_stats)
# print(fit_stats[["chisqpvalue"]])
# print(min(fit_stats[["chisqtable"]], na.rm = TRUE))
# print(fit_stats[["chisqtable"]][,1])

hourly <- data.frame(site = "0", variable = "0", chi_p = 0, min_chi = 0, df = 0, ks = 0, ad=0, samples = 0)
day <- data.frame(site = "0", variable = "0", chi_p = 0, min_chi = 0, df=0, ad = 0, samples = 0, adjust=0, alpha_adjust=0)

i <- 0
for (s in c("LUG", "RIG")){
    path <- paste('./',s,'_hourly.pdf', sep='')
    i <- i+1
    pdf(path, width=5, height=10)
    par(mfrow = c(7,2))
    for (pol in pollutants){
      df1 <- july.hour %>% filter(variable == pol & site == s)
      concvec <- c(na.omit(df1)[["value"]])
      fit <- fitdist(ReplaceNonpositive(concvec), "lnorm")
      fit_stats <- gofstat(fit)
      hourly <- hourly %>% add_row(variable = pol, ks=fit_stats[["ks"]], chi_p = fit_stats[["chisqpvalue"]],
                                     min_chi = min(fit_stats[["chisqtable"]][,1], na.rm = TRUE),
                                   site=s, samples = nrow(df1), ad = fit_stats[["ad"]], df = fit_stats[["chisqdf"]])
      
      # path <- paste("./QQ_plots/hourly_",s,"_",pol,".pdf", sep="")
      # pdf(file=path)
      #   mytitle = paste(pol,"at",s)
      #   plot(fit)
      # dev.off()
      denscomp(fit,main=paste('Emp. and Theo. Density\nFor', pol))
      qqcomp(fit, main = paste('Q-Q Plot For', pol))
      
      
      # concvec <- c(na.omit(df2)[["value"]])
      # fit <- fitdist(ReplaceNonpositive(concvec), "lnorm")
      # fit_stats <- gofstat(fit)
      # samps = nrow(df2)
      # k = 1+4/samps - 25/(samps^2)
      
      # path <- paste("./QQ_plots/daily_",s,"_",pol,".pdf", sep="")
      #   pdf(file=path)
      #   mytitle = paste(pol,"at",s)
      #   plot(fit)
      # dev.off()
      
      # day <- day %>% add_row(variable = pol, ad=fit_stats[["ad"]], chi_p = fit_stats[["chisqpvalue"]],
      #                                min_chi = min(fit_stats[["chisqtable"]][,1], na.rm = TRUE),
      #                        site = s, samples = samps, adjust=k, alpha_adjust = k*0.05, df = fit_stats[["chisqdf"]])
      # 
  }
  dev.off()
  
}
write.csv(hourly, './hourly_dist.csv')
write.csv(day, './daily_dist.csv')

# now to log the data

july.hour.log <- july.hour %>% filter(value !=0) %>% mutate(value = log(value))
july.day.log <- july.day %>% filter(value !=0) %>% mutate(value = log(value))


log.results <- data.frame(site = "0", variable = "0", chi_p = 0, min_chi = 0, df=0, 
                          ad = 0, ks = 0, samples = 0, adjust=0, alpha_adjust=0, timeframe="0")
for (s in c("LUG", "RIG")){
  for (pol in pollutants){
    df1 <- july.hour.log %>% filter(variable == pol & site == s)
    
    concvec <- c(na.omit(df1)[["value"]])
    print(concvec[concvec<0])
    temp <- ReplaceNonpositive.log(concvec)
    print(temp[temp<0])
    fit <- fitdist(ReplaceNonpositive.log(concvec), "lnorm")
    fit_stats <- gofstat(fit)
    log.results <- log.results %>% add_row(variable = pol, ks=fit_stats[["ks"]], chi_p = fit_stats[["chisqpvalue"]],
                                 min_chi = min(fit_stats[["chisqtable"]][,1], na.rm = TRUE),
                                 site=s, samples = nrow(df1), ad = fit_stats[["ad"]], df = fit_stats[["chisqdf"]],
                                 timeframe="hourly")
    
    path <- paste("./QQ_plots/hourly_log_",s,"_",pol,".pdf", sep="")
    pdf(file=path)
    mytitle = paste(pol,"at",s)
    plot(fit)
    dev.off()
    
    
  }
  
  df2 <- july.day.log %>% filter(variable == "O3" & site == s)
  concvec <- c(na.omit(df2)[["value"]])
  fit <- fitdist(ReplaceNonpositive.log(concvec), "lnorm")
  fit_stats <- gofstat(fit)
  samps = nrow(df2)
  k = 1+4/samps - 25/(samps^2)
  
  path <- paste("./QQ_plots/log_daily_",s,"_O3.pdf", sep="")
  pdf(file=path)
  mytitle = paste(pol,"at",s)
  plot(fit)
  dev.off()
  
  log.results <- log.results %>% add_row(variable = "O3", ad=fit_stats[["ad"]], chi_p = fit_stats[["chisqpvalue"]],
                         min_chi = min(fit_stats[["chisqtable"]][,1], na.rm = TRUE),
                         site = s, samples = samps, adjust=k, alpha_adjust = k*0.05, 
                         df = fit_stats[["chisqdf"]], timeframe="daily")
}
write.csv(log.results, './log_results.csv')

