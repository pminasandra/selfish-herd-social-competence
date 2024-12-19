## Pranav Minasandra and Cecilia Baldoni

### Libraries
library(tidyverse)
library(here)
library(emmeans)

## Data tidying

directory <- here(".cw")
directory_path <- sub("/$", "", readLines(directory, warn = FALSE))

file_list <- list.files(path = file.path(directory_path, "Data/Results"), pattern = "*.csv", full.names = TRUE)

data_list <- lapply(file_list, function(file) {
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

data_list <- data_list[!sapply(data_list, is.null)]

combined_data <- bind_rows(data_list)
combined_data <- combined_data %>%
  select(-type) %>% 
  relocate(pop_size, .after = uname) %>%
  relocate(reasoning, .after = pop_size)
#Optional, save csv file
#write.csv(combined_data, "combined_data.csv", row.names = FALSE)

# Convert into long format
long_data <- combined_data %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "time",
               values_to = "group_size") %>%
  mutate(time = as.numeric(gsub("t", "", time)))

## Plotting

ggplot(long_data, aes(x = time, y = group_size, color = pop_size, linetype = reasoning)) +
  #geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.7) +
  geom_point(size = 0.5, alpha = 0.8) +
  theme_bw() +
  facet_wrap(~pop_size)
ggplot(long_data, aes(x = time, y = group_size, color = pop_size, linetype = reasoning)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.7) +
  #geom_point(size = 0.5, alpha = 0.8) +
  theme_bw() +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
  facet_wrap(~pop_size)

ggplot(long_data, aes(x = time, y = group_size, color = pop_size)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.1) +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
  facet_wrap(~reasoning) 

plot <- ggplot(long_data, aes(x = time, y = group_size, color = reasoning)) +
  #geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.1) +
  geom_smooth(aes(group = interaction(pop_size, reasoning)), method = "loess", fill = 'orange') +
  facet_wrap(~pop_size, scales = "free") +
  theme_bw() +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

ggsave(file.path(file.path(directory_path, "Figures"), "group_size_plot.pdf"), plot = plot, width = 8, height = 6, units = "cm")

## Data investigation

average_group_size <- long_data %>%
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
