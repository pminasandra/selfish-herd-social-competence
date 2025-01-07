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
library(emmeans)

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

## Optional, save csv file
#write.csv(group_data, "group_data.csv", row.names = FALSE)

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

# Run a two-way ANOVA, pairwise comparison and plot emmeans
anova_group <- aov(mean_group_size ~ reasoning * pop_size, data = average_group_size)
summary(anova_group)

# Residual diagnostics for Group Data ANOVA
residuals_group <- residuals(anova_group)
shapiro.test(residuals_group)
bartlett.test(mean_group_size ~ interaction(reasoning, pop_size), data = average_group_size)

# Run GLM with a log link to account for non-normality
glm_group <- glm(mean_group_size ~ reasoning * pop_size, data = average_group_size, family = gaussian(link = "log"))
summary(glm_group)

# Standardized residuals
residuals <- residuals(glm_group, type = "deviance")
fitted_values <- fitted(glm_group)

# Plot residuals vs. fitted values
plot(fitted_values, residuals, main = "Residuals vs. Fitted Values", 
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")

# Run a Gamma GLM with a log link to handle non-normal, positive data
gamma_group <- glm(mean_group_size ~ reasoning * pop_size, 
                 family = Gamma(link = "log"), 
                 data = average_group_size)
summary(gamma_group)
residuals_gamma <- residuals(gamma_group, type = "deviance")
plot(fitted(gamma_group), residuals_gamma, 
     main = "Residuals vs Fitted (Gamma GLM)", 
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")

# Get estimated marginal means
emmeans_group <- emmeans(gamma_group, ~ reasoning | pop_size)

# Normalize the emmean values relative to D0 for each population size
pairwise_norm <- as.data.frame(emmeans_group) %>%
  group_by(pop_size) %>%
  mutate(normalized_emmean = emmean / emmean[reasoning == "d0"],
         normalized_lower.CL = lower.CL / emmean[reasoning == "d0"],
         normalized_upper.CL = upper.CL / emmean[reasoning == "d0"])

# Create the plot with normalized values
emmeans_norm <- ggplot(pairwise_norm, aes(x = reasoning, y = normalized_emmean, color = pop_size, group = pop_size)) +
  geom_line(linewidth = 1.5) + geom_point(size = 2) +
  geom_hline(yintercept = 1, color = "grey", linetype = "dashed", linewidth = 0.8) + # Add horizontal grey line
  labs(y= "Normalized EMMean Group Size", x = "Depth of Reasoning", color = "Population Size", fill = "Population Size") +
  geom_ribbon(aes(ymin = normalized_lower.CL, ymax = normalized_upper.CL, 
                  fill = pop_size), alpha = 0.1, color = NA) +
  theme_bw()

ggsave(file.path(file.path(directory_path, "Figures"), "emmeans_group_size.pdf"), plot = emmeans_norm, width = 10, height = 8, units = "cm")

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

ggsave(file.path(file.path(directory_path, "Figures"), "area_size_plot.pdf"), plot = plot_area, width = 10, height = 8, units = "cm")

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

# Run a two-way ANOVA, pairwise comparison and plot emmeans
anova_area <- aov(mean_area_size ~ reasoning * pop_size, data = average_area_size)
summary(anova_area)

# Residual diagnostics for Group Data ANOVA
residuals_area <- residuals(anova_area)
shapiro.test(residuals_area)
bartlett.test(mean_area_size ~ interaction(reasoning, pop_size), data = average_area_size)

glm_area <- glm(mean_area_size ~ reasoning * pop_size, 
                data = average_area_size, 
                family = gaussian(link = "log"))
summary(glm_area)

# Residual diagnostics for Gaussian GLM
# Plot residuals vs fitted values to check for patterns or heteroscedasticity
residuals_glm <- residuals(glm_area, type = "deviance")
fitted_glm <- fitted(glm_area)

plot(fitted_glm, residuals_glm, 
     main = "Residuals vs. Fitted (Gaussian GLM - Area)", 
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")

emmeans(glm_area, ~ reasoning | pop_size)
emmeans_area_df <- as.data.frame(emmeans(glm_area, ~ reasoning | pop_size))

# Normalize the EMMeans relative to reasoning = d0
pairwise_norm_area <- emmeans_area_df %>%
  group_by(pop_size) %>%
  mutate(normalized_emmean = emmean / emmean[reasoning == "d0"],
    normalized_lower.CL = lower.CL / emmean[reasoning == "d0"],
    normalized_upper.CL = upper.CL / emmean[reasoning == "d0"])

# Create the plot with normalized values
emmeans_area_norm <- ggplot(pairwise_norm_area, aes(x = reasoning, y = normalized_emmean, color = pop_size, group = pop_size)) +
  geom_line(linewidth = 1.5) + geom_point(size = 2) +
  geom_hline(yintercept = 1, color = "grey", linetype = "dashed", linewidth = 0.8) + # Add horizontal grey line
  labs(y= "Normalized EMMean Area", x = "Depth of Reasoning", color = "Population Size", fill = "Population Size") +
  geom_ribbon(aes(ymin = normalized_lower.CL, ymax = normalized_upper.CL, 
                  fill = pop_size), alpha = 0.05, color = NA) +
  theme_bw()

ggsave(file.path(file.path(directory_path, "Figures"), "emmeans.pdf"), plot = emmeans_area_norm, width = 10, height = 8, units = "cm")
