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


df <- full_join(cbind(site="LUG", LUG),
                cbind(site="RIG", RIG))

# This operation cuts out the unnecessary columns
reg_df <- df[, !names(df) %in% c("CPC", "EC", "PREC", "RAD", "TEMP")]

# Make the df a long format
lf <- df %>%
  gather(variable, value,
         -c(site, datetime, season, year, month, day, hour, dayofwk, daytype))

# 
# daily <- lf %>%
#   mutate(date = dates(datetime)) %>% # creates a date column
#   group_by(site, date, variable)%>% # groups the data
#   summarize(percent.recovery = length(na.omit(value))/length(value)*1e2,
#             value = mean(value, na.rm=TRUE)) %>%
#   ungroup()
# 
# threshold <- 75
# 
# daily %>%
#   filter(percent.recovery < threshold) %>%
#   count(site, variable)

s = "LUG"
pol = "CO"

ix <- s==df[["site"]] & "Jul"==df[["month"]]
out <- approx(df[ix,"datetime"], df[ix,pol], df[ix,"datetime"])
PercentRecovery <- function(x) {
  length(na.omit(x))/length(x)*1e2
}

PercentRecovery(df[ix,pol])

spec <- spectrum(out[["y"]])
hrs <- c("4-hr"=4, "6-hr"=6, "8-hr"=8, "12-hr"=12, "daily"=24, "weekly"=24*7, "monthly"=24*30)
abline(v=1/hrs, col=seq(hrs)+1, lty=2)
legend("topright", names(hrs), col=seq(hrs)+1, lty=2, bg="white")
