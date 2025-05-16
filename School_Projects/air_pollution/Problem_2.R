# install.packages("chron")
library(chron)
library(tidyverse)

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

# Make the df a long format
lf <- df %>%
  gather(variable, value,
         -c(site, datetime, season, year, month, day, hour, dayofwk, daytype))

ggplot(lf)+                                        # `lf` is the data frame
  facet_grid(variable~site, scale="free_y")+         # panels created out of these variables
  geom_line(aes(datetime, value, color=site))+       # plot `value` vs. `time` as lines
  scale_x_chron()+                                   # format x-axis labels (time units)
  theme(axis.text.x=element_text(angle=30, hjust=1)) # rotate x-axis labels

# Let's first plot out the general variables
ggplot(filter(lf,variable %in% c("PREC", "RAD", "TEMP")))+                                        # `lf` is the data frame
  facet_grid(variable~site, scale="free_y")+         # panels created out of these variables
  geom_line(aes(datetime, value, color=site))+       # plot `value` vs. `time` as lines
  scale_x_chron()+                                   # format x-axis labels (time units)
  theme(axis.text.x=element_text(angle=30, hjust=1))

# Then let's plot out the rest
ggplot(filter(lf,!variable %in% c("PREC", "RAD", "TEMP")))+                                        # `lf` is the data frame
  facet_grid(variable~site, scale="free_y")+         # panels created out of these variables
  geom_line(aes(datetime, value, color=site))+       # plot `value` vs. `time` as lines
  scale_x_chron()+                                   # format x-axis labels (time units)
  theme(axis.text.x=element_text(angle=30, hjust=1))

daily <- lf %>%
  filter(variable %in% daily.limits[["variable"]])%>% # Grabs only the needed variables
  mutate(date = dates(datetime)) %>% # creates a date column
  group_by(site, date, variable)%>% # groups the data
  summarize(percent.recovery = length(na.omit(value))/length(value)*1e2,
            value = mean(value, na.rm=TRUE)) %>%
  ungroup()

threshold <- 75

daily %>%
  filter(percent.recovery < threshold) %>%
  count(site, variable)

daily %>%
  filter(percent.recovery >= threshold) %>%
  ggplot+
  facet_grid(variable~site, scale="free_y")+  
  geom_line(aes(x=date, y=value, color=site))+
  geom_hline(data=daily.limits, mapping=aes(yintercept=value), linetype=2)+
  scale_x_chron(format="%d.%m")+
  theme(axis.text.x=element_text(angle=30, hjust=1))

O3 <- lf %>%
  filter(variable == "O3")

O3 %>%
  ggplot+
  facet_grid(variable~site, scale="free_y")+
  geom_line(aes(datetime, value, color=site))+
  scale_x_chron()+   # format x-axis labels (time units)
  theme(axis.text.x=element_text(angle=30, hjust=1))+
  geom_hline(aes(yintercept=120), linetype=2)


(limits.vec <- with(daily.limits, setNames(value, variable)))
exceedances <- daily %>%
  filter(percent.recovery >= threshold &
           value > limits.vec[as.character(variable)])  

exceedances %>%
  count(site, variable)

O3.exceedance <- O3 %>%
  filter(value > 120) 

# Counts the number of O3 excedances by site
O3.exceedance %>%
  count(site, variable)

# Prints the dates for the excedances to be reported
print(exceedances)

# Get the mean of the PM10 and PM2.5

PM10 <- lf %>% # With the long format
  filter(variable == "PM10") %>% #get the rows where the variable is PM10
  drop_na() # Get rid of NA values
mean(PM10$value)

PM2.5 <- lf %>% # With the long format
  filter(variable == "PM2.5") %>% #get the rows where the variable is PM10
  drop_na() # Get rid of NA values
mean(PM2.5$value)

RIG.numeric <- RIG[,sapply(RIG, is.numeric)]
LUG.numeric <- LUG[,sapply(LUG, is.numeric)]

print(colMeans(RIG.numeric, na.rm=T))
print(colMeans(LUG.numeric, na.rm=T))
