library(tidyverse)
library(chron)

#Options from Stakahama
Sys.setlocale("LC_TIME","C")
options(stringsAsFactors=FALSE)
options(chron.year.abb=FALSE)
theme_set(theme_bw()) # just my preference for plots

wdirLC <- "C:/Users/loren/Desktop/MA2/Air Pollution/ENV-409_Project"
setwd(wdirLC)

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
  data[,"year"] <- chron::years(data[,timecolumn])
  data[,"month"] <- months(data[,timecolumn])
  data[,"day"] <- chron::days(data[,timecolumn])
  data[,"hour"] <- chron::hours(data[,timecolumn])
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

df <- full_join(cbind(site="LUG", ReadTSeries("./data/LUG_2023-04-16.csv")),
                cbind(site="RIG", ReadTSeries("./data/RIG_2023-04-16.csv")))




lf <- df %>%
  gather(variable, value,
         -c(site, datetime, season, year, month, day, hour, dayofwk, daytype))


ggplot(lf)+                                        # `lf` is the data frame
facet_grid(variable~site, scale="free_y")+         # panels created out of these variables
geom_line(aes(datetime, value, color=site))+       # plot `value` vs. `time` as lines
scale_x_chron()+                                   # format x-axis labels (time units)
theme(axis.text.x=element_text(angle=30, hjust=1)) # rotate x-axis labels

ggplot(lf) +
  facet_grid(variable ~ site, scale = "free_y") +
  geom_boxplot(aes(month, value), outlier.size = 0.5, outlier.shape = 3)

ggplot(subset(lf, variable %in% c("EC", "CPC", "TEMP", "PREC", "RAD"))) +
  facet_grid(variable ~ site, scale = "free_y") +
  geom_boxplot(aes(season, value), outlier.size = 0.5, outlier.shape = 3)

lf %>%
  filter(site=="LUG" & !is.na(value)) %>%
  ggplot +
  facet_grid(variable ~ season, scale = "free_y") +
  geom_boxplot(aes(daytype, value), outlier.size = 0.5, outlier.shape = 3)

cat(encodeString(readLines("data/LUG_Wind_MM10_22.txt", n=20)), sep="\n")
cat(encodeString(readLines("data/RIG_Wind_MM10_22.txt", n=20)), sep="\n")

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



mean.angle <- function(theta, r=1, ...) {
  ## Function for averaging angles
  ## Polar coordinates -> Cartesian coordinates -> polar coordinates
  ##   'theta' is in degrees
  ##   'r=1' for unit circle
  ##   returns value is mean theta in degrees
  theta.rad <- theta * pi/180
  x <- mean(r * cos(theta.rad), ...)
  y <- mean(r * sin(theta.rad), ...)
  theta.deg <- atan2(y, x) * 180/pi
  ifelse(sign(theta.deg) < 0, (theta.deg + 360) %% 360, theta.deg) # -179--180 to 0--359
}


mean(c(359, 1)) # arithmetic mean
mean.angle(c(359, 1)) # mean in Cartesian coordinates

df1 <- left_join(df,
                met %>% select(-datetime) %>% mutate(year=as.character(year)))
# conc[["year"]] is otherwise incompatible with met[["year"]]
tail(df1)

df1 %>%
  mutate(hour = factor(hour)) %>%
  ggplot+
  facet_grid(site~season)+
  geom_boxplot(aes(hour, WIGE))+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

source("Windrose.R")

df1 %>%
  filter(site=="LUG") %>%
  plotWindrose(spd = "WIGE", dir = "WIRI", decreasing=FALSE)

df1 %>%
  filter(site=="RIG") %>%
  plotWindrose(spd = "WIGE", dir = "WIRI", decreasing=FALSE)

plotWindrose(df1, spd = "WIGE", dir = "WIRI") +
  facet_grid(site~season)
