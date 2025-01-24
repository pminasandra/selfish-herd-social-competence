## Pranav Minasandra and Cecilia Baldoni 

#  Setup and Libraries ####

# Ensure renv is installed, otherwise install it: 
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Restore the R environment with required libraries:
renv::restore()

# Libraries:
library(tidyverse)
library(here)
library(mgcv)
library(emmeans)
library(gratia)

# Data loading and processing ####

directory <- here(".cw")
directory_path <- sub("/$", "", readLines(directory, warn = FALSE))

load_and_process_data <- function(path, pattern, additional_processing = NULL) {
  files <- list.files(path = path, pattern = pattern, full.names = TRUE)
  
  data_list <- lapply(files, function(file) {
    data <- read.csv(file)
    if (nrow(data) == 0) {
      message("Empty file: ", file, ", skipping")
      return(NULL)
    }
    # Determine file type based on pattern
    base_name <- gsub("\\.csv$", "", basename(file))
    if (grepl("^[0-9].*\\.csv$", basename(file))) {
      # Process group data format: pop_size-reasoning
      split_values <- strsplit(base_name, "-")[[1]]
      data$pop_size <- split_values[1]
      data$reasoning <- split_values[2]
    } else if (grepl("^areas.*\\.csv$", basename(file))) {
      # Process area data format: areas-pop_size-reasoning
      split_values <- strsplit(base_name, "-")[[1]]
      data$pop_size <- split_values[2]
      data$reasoning <- split_values[3]
    } else {
      stop("Unknown file naming format: ", file)
    }
    
    if (!is.null(additional_processing)) {
      data <- additional_processing(data)
    }
    return(data)
  })
  
  data_list <- data_list[!sapply(data_list, is.null)]
  return(bind_rows(data_list))
}

# Load group data
group_data <- load_and_process_data(
  path = file.path(directory_path, "Data/Results"),
  pattern = "^[0-9].*\\.csv$") %>%
  relocate(pop_size, .after = uname) %>%
  relocate(reasoning, .after = pop_size)

# Load area data
area_data <- load_and_process_data(
  path = file.path(directory_path, "Data/Results"),
  pattern = "^areas.*\\.csv$") %>%
  relocate(pop_size, .after = uname) %>%
  relocate(reasoning, .after = pop_size)

## Group Size ####

# Convert into long format
group_longdata <- group_data %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "time",
               values_to = "group_size") %>%
  mutate(time = as.numeric(gsub("t", "", time)),
         pop_size = factor(pop_size, levels = sort(unique(as.numeric(pop_size)))),
         reasoning = as.factor(reasoning))

### EDA ####
ggplot(group_longdata, aes(x = time, y = group_size, color = pop_size, linetype = reasoning)) +
  geom_point(size = 0.5, alpha = 0.8) +
  labs(y= "Group Size", x = "Time (s)", color = "Population Size") +
  facet_wrap(~pop_size) +
  theme_bw()

ggplot(group_longdata, aes(x = time, y = group_size, color = pop_size)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.1) +
  labs(y= "Group Size", x = "Time (s)", color = "Population Size") +
  facet_wrap(~reasoning) +
  theme_bw() 

plot_groupsize <- ggplot(group_longdata, aes(x = time, y = group_size, color = reasoning)) +
  geom_smooth(aes(group = interaction(pop_size, reasoning)), method = "loess", fill = 'orange') +
  labs(y= "Group Size", x = "Time (s)", color = "Reasoning") +
  facet_wrap(~pop_size, scales = "free") +
  theme_bw()

ggsave(file.path(file.path(directory_path, "Figures"), "group_size_plot.pdf"), 
       plot = plot_groupsize, width = 20, height = 15, units = "cm")

### Data analysis ####
average_group_size <- group_longdata %>%
  group_by(time, pop_size, reasoning) %>%
  mutate(reasoning = factor(reasoning, levels = c("d0", "d1", "d2", "d3"))) %>% 
  summarize(mean_group_size = mean(group_size, na.rm = TRUE),
            median_group_size = median(group_size, na.rm = TRUE),
            sd_group_size = sd(group_size, na.rm = TRUE),
            .groups = "drop")
ggplot(average_group_size, aes(x = time, y = mean_group_size, color = pop_size)) +
  geom_line(linewidth = 1) +
  facet_wrap(~reasoning) +
  labs(y= "Mean Group Size", x = "Time (s)", color = "Population Size") +
  theme_bw()

gam_group <- gam(mean_group_size ~ reasoning + pop_size + s(time, bs = "cs"), 
                 family = Gamma(link = "log"), 
                 data = average_group_size)
summary(gam_group)
plot(gam_group, residuals = TRUE, pch = 16, cex = 0.5)

gam_group_interaction <- gam(mean_group_size ~ reasoning + s(time, by = pop_size, bs = "cs") + pop_size, 
                             family = Gamma(link = "log"), 
                             data = average_group_size)
summary(gam_group_interaction)
plot(gam_group_interaction, residuals = TRUE, pch = 16, cex = 0.5)

# Add predictions and confidence intervals
average_group_size$pop_size <- factor(average_group_size$pop_size)
newdata <- expand.grid(time = seq(0, 500, by = 1),
                       reasoning = unique(average_group_size$reasoning),
                       pop_size = unique(average_group_size$pop_size))

# Generate predictions and confidence intervals
predictions <- predict(gam_group_interaction, newdata = newdata, se.fit = TRUE, type = "response")

# Add predictions to the newdata
newdata$fit <- predictions$fit
newdata$lower <- predictions$fit - 1.96 * predictions$se.fit
newdata$upper <- predictions$fit + 1.96 * predictions$se.fit
newdata$pop_size <- factor(newdata$pop_size, levels = levels(average_group_size$pop_size))

ggplot(newdata, aes(x = time, y = fit, color = reasoning, group = reasoning)) +
  geom_line(linewidth = 1) + 
  # geom_ribbon(aes(ymin = lower, ymax = upper, fill = reasoning), alpha = 0.2) + 
  facet_wrap(~pop_size, scales = "free") +
  labs(y = "Predicted Group Size", x = "Time (s)", color = "Reasoning", fill = "Reasoning") +
  theme_bw()

# Compute derivatives for the smooth term and plot
derivatives <- derivatives(gam_group_interaction, 
                           select = "time", interval = "confidence", partial_match = TRUE,
                           data = expand_grid(time = seq(0,500, by = 1),
                                              pop_size = unique(average_group_size$pop_size),
                                              reasoning = unique(average_group_size$reasoning)))

ggplot(derivatives, aes(x = time, y = .derivative, color = pop_size, group = pop_size)) +
  geom_line(size = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Time (s)", y = "Derivatives of Group Size by Time", color = "Population Size") +
  theme_bw() +
  facet_wrap(~pop_size)

# Filter for near-zero derivatives
inflection_points <- derivatives %>%
  filter(abs(.derivative) < 0.005) %>%
  group_by(pop_size) %>%
  summarize(inflection_time = min(time))

ggplot(inflection_points, aes(x = pop_size, y = inflection_time)) +
  geom_point(size = 3) +
  geom_line(group = 1, color = "blue") +
  labs(x = "Population Size", y = "Inflection Time (s)") +
  theme_bw()

# Get estimated marginal means
time_points <- seq(0, 500, by = 50)
emmeans_group <- emmeans(gam_group_interaction, ~ reasoning | pop_size + time, 
                         at = list(time = time_points))

# Convert EMMeans to a data frame
emmeans_df <- as.data.frame(emmeans_group)

# Plot EMMeans across time
ggplot(emmeans_df, aes(x = time, y = emmean, color = reasoning, group = reasoning)) +
  geom_line(size = 1) + geom_point(size = 2) +
  geom_ribbon(aes(ymin = lower.CL, ymax = upper.CL, fill = reasoning), alpha = 0.2) + 
  facet_wrap(~ pop_size, scales = "free") +  # Facet by population size
  labs(y = "Estimated Marginal Mean Group Size", x = "Time (s)", color = "Reasoning", fill = "Reasoning") +
  theme_bw()

emmeans_gam_avg <- emmeans(gam_group_interaction, ~ reasoning | pop_size)

# Normalize EMMeans as before
pairwise_norm_avg <- as.data.frame(emmeans_gam_avg) %>%
  group_by(pop_size) %>%
  mutate(normalized_emmean = emmean / emmean[reasoning == "d0"],
         normalized_lower.CL = lower.CL / emmean[reasoning == "d0"],
         normalized_upper.CL = upper.CL / emmean[reasoning == "d0"])

# Plot normalized EMMeans
emmeans_norm <- ggplot(pairwise_norm_avg, aes(x = reasoning, y = normalized_emmean, color = pop_size, group = pop_size)) +
  geom_line(linewidth = 1) + geom_point(size = 1.5) +
  geom_hline(yintercept = 1, color = "grey", linetype = "dashed", linewidth = 0.5) + 
  labs(y = "Normalized EMMean Group Size", x = "Depth of Reasoning", color = "Population Size", fill = "Population Size") +
  geom_ribbon(aes(ymin = normalized_lower.CL, ymax = normalized_upper.CL, fill = pop_size), alpha = 0.1, color = NA) + theme_bw()
                  
ggsave(file.path(file.path(directory_path, "Figures"), "emmeans_group_size.pdf"), 
       plot = emmeans_norm, width = 20, height = 15, units = "cm")

## Area ####
# Convert into long format
area_longdata <- area_data %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "time",
               values_to = "area_size") %>%
  mutate(time = as.numeric(gsub("t", "", time)),
         pop_size = factor(pop_size, levels = sort(unique(as.numeric(pop_size)))),
         reasoning = as.factor(reasoning))

area_longdata <- area_longdata %>%
  mutate(pop_size_numeric = as.numeric(as.character(pop_size)),
         area_scaled = area_size * pop_size_numeric) %>% 
  select(- pop_size_numeric)

### EDA ####
ggplot(area_longdata, aes(x = time, y = area_size, color = pop_size)) +
  geom_point(size = 0.5, alpha = 0.8) +
  labs(y= "Area Size", x = "Time (s)", color = "Population Size") +
  facet_wrap(~pop_size) +
  theme_bw()

ggplot(area_longdata, aes(x = time, y = area_size, color = pop_size)) +
  geom_line(aes(group = interaction(uname, pop_size, reasoning)), alpha = 0.7) +
  labs(y= "Area Size", x = "Time (s)", color = "Population Size") +
  facet_wrap(~pop_size, scales = "free") +
  theme_bw() 

plot_area <- ggplot(area_longdata, aes(x = time, y = area_scaled, color = reasoning)) +
  geom_smooth(aes(group = interaction(pop_size, reasoning)), method = "loess", fill = 'orange') +
  facet_wrap(~pop_size) +
  labs(y= "Area Size", x = "Time (s)") +
  theme_bw() 

ggsave(file.path(file.path(directory_path, "Figures"), "area_size_plot.pdf"), 
       plot = plot_area, width = 20, height = 15, units = "cm")

### Data analysis ####
average_area_size <- area_longdata %>%
  group_by(time, pop_size, reasoning) %>%
  mutate(reasoning = factor(reasoning, levels = c("d0", "d1", "d2", "d3"))) %>% 
  summarize(mean_area_size = mean(area_size, na.rm = TRUE),
            median_area_size = median(area_size, na.rm = TRUE),
            sd_area_size = sd(area_size, na.rm = TRUE),
            .groups = "drop")
ggplot(average_area_size, aes(x = time, y = mean_area_size, color = pop_size)) +
  geom_line(linewidth = 1) +
  facet_wrap(~reasoning) +
  labs(y= "Mean Area Size", x = "Time (s)", color = "Population Size") +
  theme_bw()

gam_area <- gam(mean_area_size ~ reasoning + pop_size + s(time, bs = "cs"), 
    family = Gamma(link = "log"), 
    data = average_area_size)
summary(gam_area)
plot(gam_area, residuals = TRUE, pch = 16, cex = 0.5)

gam_area_interaction <- gam(mean_area_size ~ reasoning + s(time, by = pop_size, bs = "cs") + pop_size, 
                             family = Gamma(link = "log"), 
                             data = average_area_size)
summary(gam_area_interaction)
plot(gam_area_interaction, residuals = TRUE, pch = 16, cex = 0.5)

# Add predictions and confidence intervals
average_area_size$pop_size <- factor(average_area_size$pop_size)

# Create a grid for predictions
prediction_grid_area <- expand.grid(time = seq(0, 500, by = 10),
                                    reasoning = unique(average_area_size$reasoning),
                                    pop_size = unique(average_area_size$pop_size))

# Generate predictions and confidence intervals
predictions <- predict(gam_area_interaction, newdata = prediction_grid_area, se.fit = TRUE, type = "response")

# Add predictions to the grid
prediction_grid_area$fit <- predictions$fit
prediction_grid_area$lower <- predictions$fit - 1.96 * predictions$se.fit
prediction_grid_area$upper <- predictions$fit + 1.96 * predictions$se.fit
prediction_grid_area$pop_size <- factor(prediction_grid_area$pop_size, levels = levels(average_area_size$pop_size))

# Plot predicted values from GAM interaction model
ggplot(prediction_grid_area, aes(x = time, y = fit, color = reasoning, group = reasoning)) +
  geom_line(linewidth = 1) + 
  # geom_ribbon(aes(ymin = lower, ymax = upper, fill = reasoning), alpha = 0.2) + 
  facet_wrap(~pop_size, scales = "free") +
  labs(y = "Predicted Area Size", x = "Time (s)", color = "Reasoning", fill = "Reasoning") +
  theme_bw()

# Compute derivatives for the smooth term and plot
derivatives_area <- derivatives(gam_area_interaction, select = "time", interval = "confidence", partial_match = TRUE)

ggplot(derivatives_area, aes(x = time, y = .derivative, color = pop_size, group = pop_size)) +
  geom_line(size = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +  # Highlight zero derivative
  labs(x = "Time (s)", y = "Derivatives of Area Size by Time", color = "Population Size") +
  theme_bw() +
  facet_wrap(~pop_size)

# Filter for near-zero derivatives
inflection_area <- derivatives_area %>%
  filter(abs(.derivative) < 0.01) %>%
  group_by(pop_size) %>%
  summarize(inflection_time = min(time))

ggplot(inflection_area, aes(x = pop_size, y = inflection_time)) +
  geom_point(size = 3) +
  geom_line(group = 1, color = "blue") +
  labs(
    x = "Population Size", 
    y = "Inflection Time (s)") +
  theme_bw()

# Get estimated marginal means
emmeans_area <- emmeans(gam_area_interaction, ~ reasoning | pop_size + time, 
                        at = list(time = time_points))

# Convert EMMeans to a data frame
emmeans_area_df <- as.data.frame(emmeans_area)

# Plot EMMeans across time
ggplot(emmeans_area_df, aes(x = time, y = emmean, color = reasoning, group = reasoning)) +
  geom_line(size = 1) + geom_point(size = 2) +
  geom_ribbon(aes(ymin = lower.CL, ymax = upper.CL, fill = reasoning), alpha = 0.2) + 
  facet_wrap(~ pop_size, scales = "free") +  # Facet by population size
  labs(y = "Estimated Marginal Mean Area Size", x = "Time (s)", color = "Reasoning", fill = "Reasoning") +
  theme_bw()

emmeans_area_avg <- emmeans(gam_area_interaction, ~ reasoning | pop_size)

# Normalize EMMeans as before
pairwise_norm_area_avg <- as.data.frame(emmeans_area_avg) %>%
  group_by(pop_size) %>%
  mutate(normalized_emmean = emmean / emmean[reasoning == "d0"],
         normalized_lower.CL = lower.CL / emmean[reasoning == "d0"],
         normalized_upper.CL = upper.CL / emmean[reasoning == "d0"])

# Plot normalized EMMeans
emmeans_area_norm <- ggplot(pairwise_norm_area_avg, aes(x = reasoning, y = normalized_emmean, color = pop_size, group = pop_size)) +
  geom_line(linewidth = 1) + geom_point(size = 1.5) +
  geom_hline(yintercept = 1, color = "grey", linetype = "dashed", linewidth = 0.5) + 
  labs(y = "Normalized EMMean Area Size", x = "Depth of Reasoning", color = "Population Size", fill = "Population Size") +
  geom_ribbon(aes(ymin = normalized_lower.CL, ymax = normalized_upper.CL, fill = pop_size), alpha = 0.1, color = NA) + theme_bw()

# Save the plot
ggsave(file.path(file.path(directory_path, "Figures"), "emmeans_area_size.pdf"), 
       plot = emmeans_area_norm, width = 20, height = 15, units = "cm")
