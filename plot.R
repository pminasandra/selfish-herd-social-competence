## Pranav Minasandra and Cecilia Baldoni
### Libraries
library(tidyverse)


csv_dir <- "~/Downloads/results-typical-group-size"
file_list <- list.files(path = csv_dir, pattern = "*.csv", full.names = TRUE)

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
write.csv(combined_data, "combined_data.csv", row.names = FALSE)

## GGplots

long_data <- combined_data %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "time",
               values_to = "group_size") %>%
  mutate(time = as.numeric(gsub("t", "", time)))

ggplot(long_data, aes(x = time, y = group_size, color = pop_size, linetype = reasoning)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.7) +
  geom_point(size = 0.5, alpha = 0.8) +
  labs(title = "Group Size Over Time", x = "Time", y = "Group Size", color = "Population Size",  linetype = "Degree of Reasoning") +
  facet_wrap(~pop_size)

ggplot(long_data, aes(x = time, y = group_size, color = reasoning)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.1) +
  geom_point(size = 0.5, alpha = 0.8) +
  facet_wrap(~reasoning)

ggplot(long_data, aes(x = time, y = group_size, color = reasoning)) +
  #geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.1) +
  geom_smooth(aes(group = interaction(pop_size, reasoning)), method = "loess", fill = 'orange') +
  facet_wrap(~pop_size, scales = "free") +
  theme_bw()

