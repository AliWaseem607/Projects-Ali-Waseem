# install.packages("chron")
library(tidyverse)
library(chron)
library(lubridate) # for working with dates
library(ggplot2)  # for creating graphs
library(scales)   # to access breaks/formatting functions

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

df <- full_join(cbind(site="LUG", LUG),
                cbind(site="RIG", RIG))

# This operation cuts out the unnecessary columns
reg_df <- df[, !names(df) %in% c("CPC", "EC", "PREC", "RAD", "TEMP")]

# Set up limit values for 24h limits
daily.limits <- data.frame(value = c(100, 80, 8, 50),
                          variable=c("SO2","NO2", "CO", "PM10"))

df7 <- df[, !names(df) %in% c("CPC", "EC", "PREC", "RAD", "TEMP", "CO", "NMVOC", "NOX", "O3","SO2", "NO2")] %>%
        filter(month %in% c("Jan", "Feb", "Mar", "Apr", "May", "Jun"))

df7_all_month <- df[, !names(df) %in% c("CPC", "EC", "PREC", "RAD", "TEMP", "CO", "NMVOC", "NOX", "O3","SO2", "NO2")]

# Make the df a long format
lf <- df7 %>%
  gather(variable, value,
         -c(site, datetime, season, year, month, day, hour, dayofwk, daytype))

ggplot(lf)+                                        # `lf` is the data frame
  facet_grid(variable~site, scale="free_y")+         # panels created out of these variables
  geom_line(aes(datetime, value, color=site))+       # plot `value` vs. `time` as lines
  scale_x_chron(n=6)+                                   # format x-axis labels (time units)
  theme(axis.text.x=element_text(angle=30, hjust=1))+ # rotate x-axis labels
  geom_vline(xintercept = c(as.chron( "01.03.2022", "%d.%m.%Y"), as.chron( "01.04.2022", "%d.%m.%Y")))



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
source("Windrose.R")

mean(c(359, 1)) # arithmetic mean
mean.angle(c(359, 1)) # mean in Cartesian coordinates

df1 <- left_join(df,
                 met %>% select(-datetime) %>% mutate(year=as.character(year)))
# conc[["year"]] is otherwise incompatible with met[["year"]]
tail(df1)

df1_7 <- df1 %>% filter(datetime <= as.chron("01.04.2022", "%d.%m.%Y")) %>%
          filter(datetime >= as.chron("01.03.2022", "%d.%m.%Y"))

df1_else <- df1%>% filter(datetime > as.chron("01.04.2022", "%d.%m.%Y")| datetime < as.chron("01.03.2022", "%d.%m.%Y")) 

df1_7 %>%
  filter(site=="LUG") %>%
  plotWindrose(spd = "WIGE", dir = "WIRI", decreasing=FALSE)

df1_7 %>%
  filter(site=="RIG") %>%
  plotWindrose(spd = "WIGE", dir = "WIRI", decreasing=FALSE)

plotWindrose(df1_7, spd = "WIGE", dir = "WIRI") +
  facet_grid(.~site)

df1_else %>%
  filter(site=="LUG") %>%
  plotWindrose(spd = "WIGE", dir = "WIRI", decreasing=FALSE)

df1_else %>%
  filter(site=="RIG") %>%
  plotWindrose(spd = "WIGE", dir = "WIRI", decreasing=FALSE)

plotWindrose(df1_else, spd = "WIGE", dir = "WIRI") +
  facet_grid(.~site)

df1_before <- df1 %>% filter(datetime < as.chron("01.03.2022", "%d.%m.%Y")) %>%
  filter(datetime >= as.chron("01.02.2022", "%d.%m.%Y"))

plotWindrose(df1_before, spd = "WIGE", dir = "WIRI") +
  facet_grid(.~site)


df7_ttest_PM10 <- df7_all_month %>% gather(variable, value, c(PM10, PM2.5)) %>% filter(variable == "PM10")
df7_ttest_PM2.5 <- df7_all_month %>% gather(variable, value, c(PM10, PM2.5)) %>% filter(variable == "PM2.5")


(out_LUG_PM10 <- t.test(filter(df7_ttest_PM10, site=="LUG" & (datetime <= as.chron("01.04.2022", "%d.%m.%Y")& datetime >= as.chron("01.03.2022", "%d.%m.%Y")))[["value"]],
               filter(df7_ttest_PM10, site=="LUG" & (datetime > as.chron("01.04.2022", "%d.%m.%Y")| datetime < as.chron("01.03.2022", "%d.%m.%Y")))[["value"]],
               alternative="greater"))

(out_RIG_PM10 <- t.test(filter(df7_ttest_PM10, site=="RIG" & (datetime <= as.chron("01.04.2022", "%d.%m.%Y")& datetime >= as.chron("01.03.2022", "%d.%m.%Y")))[["value"]],
               filter(df7_ttest_PM10, site=="RIG" & (datetime > as.chron("01.04.2022", "%d.%m.%Y")| datetime < as.chron("01.03.2022", "%d.%m.%Y")))[["value"]],
               alternative="greater"))

(out_LUG_PM2.5 <- t.test(filter(df7_ttest_PM2.5, site=="LUG" & (datetime <= as.chron("01.04.2022", "%d.%m.%Y")& datetime >= as.chron("01.03.2022", "%d.%m.%Y")))[["value"]],
                        filter(df7_ttest_PM2.5, site=="LUG" & (datetime > as.chron("01.04.2022", "%d.%m.%Y")| datetime < as.chron("01.03.2022", "%d.%m.%Y")))[["value"]],
                        alternative="greater"))

(out_RIG_PM2.5 <- t.test(filter(df7_ttest_PM2.5, site=="RIG" & (datetime <= as.chron("01.04.2022", "%d.%m.%Y")& datetime >= as.chron("01.03.2022", "%d.%m.%Y")))[["value"]],
                        filter(df7_ttest_PM2.5, site=="RIG" & (datetime > as.chron("01.04.2022", "%d.%m.%Y")| datetime < as.chron("01.03.2022", "%d.%m.%Y")))[["value"]],
                        alternative="greater"))

library(lattice)

CorrelationValue <- function(x, y, ...) {
  correlation <- cor(x, y, use="pairwise.complete.obs") 
  if(is.finite(correlation)) {
    cpl <- current.panel.limits()
    panel.text(mean(cpl$xlim), mean(cpl$ylim),
               bquote(italic(r)==.(sprintf("%.2f", correlation))),
               adj=c(0.5,0.5), col="blue")
  }
}

df_apit <- df%>% filter(datetime <= as.chron("01.04.2022", "%d.%m.%Y")& datetime >= as.chron("01.03.2022", "%d.%m.%Y"))
ix <- "RIG" == df_apit[["site"]]

df_apot <- df%>% filter(datetime > as.chron("01.04.2022", "%d.%m.%Y")| datetime < as.chron("01.03.2022", "%d.%m.%Y"))
ix <- "LUG" == df_apot[["site"]]

pdf("rplot.pdf", width = 10, height = 10)
# 2. Create the plot
  splom(~df_apot[ix,c("O3","NO2","CO","PM10","PM2.5","TEMP", "SO2")],
        upper.panel = CorrelationValue,
        pch=4)
# 3. Close the file
dev.off()

png("rplot.png", width = 700, height = 700)
# 2. Create the plot
splom(~df_apot[ix,c("O3","NO2","CO","PM10","PM2.5","TEMP", "SO2")],
      upper.panel = CorrelationValue,
      pch=4)
# 3. Close the file
dev.off()



