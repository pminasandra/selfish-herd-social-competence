## Pranav Minasandra and Cecilia Baldoni 

# Libraries ####
library(tidyverse)
library(here)
library(emmeans)

# Data loading and tidying ####

directory <- here(".cw")
directory_path <- sub("/$", "", readLines(directory, warn = FALSE))

## Load group data from Data/groups folder ####
group_list <- list.files(path = file.path(directory_path, "Data/groups"), pattern = "*.csv", full.names = TRUE)

group_list <- lapply(group_list, function(file) {
  data <- read.csv(file)
  if (nrow(data) == 0) {
    message("Empty file: ", file, ", skipping")
    return(NULL)
  }
  data$type <- gsub("\\.csv$", "", basename(file))
  split_values <- strsplit(data$type, "-")
  data$pop_size <- sapply(split_values, `[`, 1)
  data$reasoning <- sapply(split_values, `[`, 2)
  
  return(data)
})

group_list <- group_list[!sapply(group_list, is.null)]

group_data <- bind_rows(group_list)
group_data <- group_data %>%
  select(-type) %>% 
  relocate(pop_size, .after = uname) %>%
  relocate(reasoning, .after = pop_size)
#Optional, save csv file
#write.csv(group_data, "group_data.csv", row.names = FALSE)

### Convert into long format ####
group_longdata <- group_data %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "time",
               values_to = "group_size") %>%
  mutate(time = as.numeric(gsub("t", "", time)))

### EDA ####

ggplot(group_longdata, aes(x = time, y = group_size, color = pop_size, linetype = reasoning)) +
  #geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.7) +
  geom_point(size = 0.5, alpha = 0.8) +
  theme_bw() +
  facet_wrap(~pop_size)
ggplot(group_longdata, aes(x = time, y = group_size, color = pop_size, linetype = reasoning)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.7) +
  #geom_point(size = 0.5, alpha = 0.8) +
  theme_bw() +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
  facet_wrap(~pop_size)

ggplot(group_longdata, aes(x = time, y = group_size, color = pop_size)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.1) +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
  facet_wrap(~reasoning) 

plot <- ggplot(group_longdata, aes(x = time, y = group_size, color = reasoning)) +
  #geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.1) +
  geom_smooth(aes(group = interaction(pop_size, reasoning)), method = "loess", fill = 'orange') +
  facet_wrap(~pop_size, scales = "free") +
  theme_bw() +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

ggsave(file.path(file.path(directory_path, "Figures"), "group_size_plot.pdf"), plot = plot, width = 8, height = 6, units = "cm")

### Data analysis ####
average_group_size <- group_longdata %>%
  group_by(time, pop_size, reasoning) %>%
  summarize(mean_group_size = mean(group_size, na.rm = TRUE),
            median_group_size = median(group_size, na.rm = TRUE),
            sd_group_size = sd(group_size, na.rm = TRUE),
            .groups = "drop")
ggplot(average_group_size, aes(x = time, y = mean_group_size, color = pop_size)) +
  geom_line(size = 1) +
  facet_wrap(~reasoning) +
  theme_bw()

average_group_size %>%
  group_by(pop_size, reasoning) %>%
  filter(mean_group_size == max(mean_group_size)) %>%
  arrange(pop_size, reasoning)

average_group_size <- average_group_size %>%
  mutate(reasoning = factor(reasoning, levels = c("d0", "d1", "d2", "d3")))

# Run a two-way ANOVA, pairwise comparison and plot emmeans
anova_result <- aov(mean_group_size ~ reasoning * pop_size, data = average_group_size)
summary(anova_result)

emmeans(anova_result, ~ reasoning | pop_size)
pairwise_df <- as.data.frame(emmeans(anova_result, ~ reasoning | pop_size))

emmeans <- ggplot(pairwise_df, aes(x = reasoning, y = emmean, color = pop_size, group = pop_size)) +
  geom_line(size = 1) + geom_point(size = 2) +
  theme_bw()

ggsave(file.path(file.path(directory_path, "Figures"), "emmeans.pdf"), plot = emmeans, width = 8, height = 6, units = "cm")


## Load area file from Data/median-area folder ####
area_list <- list.files(path = file.path(directory_path, "Data/median-area"), pattern = "*.csv", full.names = TRUE)

area_list <- lapply(area_list, function(file) {
  data <- read.csv(file)
  if (nrow(data) == 0) {
    message("Empty file: ", file, ", skipping")
    return(NULL)
  }
  data$type <- gsub("^areas-|\\.csv$", "", basename(file))
  split_values <- strsplit(data$type, "-")
  data$pop_size <- sapply(split_values, `[`, 1)
  data$reasoning <- sapply(split_values, `[`, 2)
  
  return(data)
})

area_list <- area_list[!sapply(area_list, is.null)]

area_data <- bind_rows(area_list)
area_data <- area_data %>%
  select(-type) %>% 
  relocate(pop_size, .after = uname) %>%
  relocate(reasoning, .after = pop_size)

### Convert into long format ####
area_longdata <- area_data %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "time",
               values_to = "area_size") %>%
  mutate(time = as.numeric(gsub("t", "", time)))

### EDA ####
ggplot(area_longdata, aes(x = time, y = area_size, color = pop_size)) +
  #geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.7) +
  geom_point(size = 0.5, alpha = 0.8) +
  theme_bw() +
  facet_wrap(~pop_size)
ggplot(area_longdata, aes(x = time, y = area_size, color = pop_size)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.7) +
  #geom_point(size = 0.5, alpha = 0.8) +
  theme_bw() +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
  facet_wrap(~pop_size, scales = "free")

ggplot(area_longdata, aes(x = time, y = area_size, color = reasoning)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.1) +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
  facet_wrap(~reasoning, scales = "free") 

plot <- ggplot(area_longdata, aes(x = time, y = area_size, color = reasoning)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), colour = "lightblue", alpha = 0.05) +
  geom_smooth(aes(group = interaction(pop_size, reasoning)), method = "loess", fill = 'orange') +
  facet_wrap(~pop_size, scales = "free") +
  theme_bw() +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))
