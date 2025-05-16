# install.packages("chron")
library(tidyverse)
library(chron)

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

ReadMet <- function(filename) {
  data <- read.table(filename, skip=15, col.names=c("year", "month", "day", "hour", "minute", "WIRI", "WIGE"))
  data %>%
    mutate(datetime = as.chron(paste(year, month, day, hour, minute), "%Y %m %d %H %M"),
           year     = years(datetime),
           month    = months(datetime),
           day      = days(datetime),
           hour     = hours(datetime),
           minute   = minutes(datetime),
           WIRI     = ifelse(WIRI <= -9999, NA, WIRI),
           WIGE     = ifelse(WIGE <= -9999, NA, WIGE))
}

datapath <- "./data/"
met <- full_join(cbind(site="LUG", ReadMet(file.path(datapath, "LUG_Wind_MM10_22.txt"))),
                 cbind(site="RIG", ReadMet(file.path(datapath, "RIG_Wind_MM10_22.txt"))))
met[,"season"] <- Month2Season(unclass(met[,"month"]))


df <- full_join(cbind(site="LUG", LUG),
                cbind(site="RIG", RIG))

# This operation cuts out the unnecessary columns
# reg_df <- df[, !names(df) %in% c("CPC", "EC", "PREC", "RAD", "TEMP")]

variables <- c("O3", "NO2", "CO", "PM10", "SO2", "NMVOC", "EC", "TEMP", "PREC", "RAD", "PM2.5", "CPC", "NOX")





lf <- gather(df, variable, value, all_of(variables))

variables <- c("WIRI","WIGE")
mlf <- gather(met, variable, value, all_of(variables)) %>%
        filter(year=="2022") %>% mutate(year=factor(year))

msf.hourly <- mlf %>% group_by(site, year, month, day, hour, season, variable) %>%
  summarize(value=mean(value, na.rm=TRUE)) %>%
  spread(variable, value)

msf.hourly.2 <- mutate(msf.hourly, 
       datetime = as.chron(paste(day, month, year, hour, sep = "."), 
                                 "%d.%b.%Y.%H"))

mlf.hourly <- gather(msf.hourly, variable, value, all_of(variables))

# first take hourly average of windspeed

alf <- full_join(lf, mlf.hourly)

alf <-alf %>% mutate(datetime = as.chron(paste(year, month, day, hour), "%Y %b %d %H",),
                     dayofwk = weekdays(datetime),
                     daytype = ifelse(dayofwk %in% c("Sat","Sun"), "Weekend", "Weekday"))

daily.mean <- lf %>%
  group_by(site, year, month, day, season, variable) %>%
  summarize(value=mean(value, na.rm=TRUE)) %>%
  spread(variable, value)

daily.max <- lf %>%
  group_by(site, year, month, day, season, variable) %>%
  summarize(value=max(value, na.rm=TRUE)) %>%
  spread(variable, value)

daily.med.alf <- alf %>%
  group_by(site, year, month, day, season, variable) %>%
  summarize(value=median(value, na.rm=TRUE)) %>%
  spread(variable, value)

asf <- alf %>% spread(variable, value)

ggplot(daily.max)+
  facet_grid(site~season)+
  geom_point(aes(RAD, O3))

#Windspeed is WIGE
ggplot(daily.med.alf)+
  facet_grid(site~season)+
  geom_point(aes(WIGE, PM2.5))


(cor.values.B <- daily.max %>% group_by(site, season) %>%
    summarize(correlation=cor(RAD, O3, use="pairwise.complete.obs")))

(cor.values.A <- daily.med.alf %>% group_by(site, season) %>%
    summarize(correlation=cor(WIGE, PM2.5, use="pairwise.complete.obs")))

Lag <- function(pair, k) {
  out <- data.frame(lag=k, head(pair[,1],-k), tail(pair[,2],-k))
  names(out)[2:3] <- colnames(pair)
  out
}

# lagged <- df %>%
#   group_by(site, season) %>%
#   do(rbind(Lag(.[,c("RAD","O3")], 1),
#            Lag(.[,c("RAD","O3")], 2),
#            Lag(.[,c("RAD","O3")], 3),
#            Lag(.[,c("RAD","O3")], 4),
#            Lag(.[,c("RAD","O3")], 5),
#            Lag(.[,c("RAD","O3")], 6)))
# 
# ggplot(lagged) +
#   geom_point(aes(RAD, O3, group=site, color=site), shape=4)+
#   facet_grid(lag~season)

LaggedCorrelation <- function(pair, ...) {
  out <- ccf(pair[,1], pair[,2], ..., na.action=na.pass, plot=FALSE)
  data.frame(lag=out[["lag"]], value=out[["acf"]])
}

lagged.values <- df %>% group_by(site, season) %>%
  do(LaggedCorrelation(.[,c("RAD","O3")], lag.max=6))

ggplot(lagged.values)+
  geom_segment(aes(x=lag,xend=lag,y=0,yend=value))+
  facet_grid(site~season)+
  xlab("lag (hours)")+
  ylab("Cross correlation coefficient")

NO2_LUG <-mean(LUG$NO2, na.rm = TRUE)
NO2_RIG <-mean(RIG$NO2, na.rm = TRUE)

NOX_LUG <-mean(LUG$NOX, na.rm = TRUE)
NOX_RIG <-mean(RIG$NOX, na.rm = TRUE)
