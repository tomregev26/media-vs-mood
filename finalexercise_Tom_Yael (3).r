# Load necessary libraries
library(rstan)
library(bayesplot)
library(ggplot2)
library(dplyr)

# Load and prepare social_data
setwd("~/Desktop/Baysian Modeling/Final Exercise")
social_data <- read.csv("social_networks.csv")
social_data <- na.omit(social_data) # the original social data set was structured with one line separation between observations
colnames(social_data)[colnames(social_data) == "Daily_Usage_Time..minutes."] <- "Daily_Usage_Time"

# Question 1-
# Create response variable 'Engagement' that represents the Likes+ Comments that a user received
social_data <- social_data %>%
  mutate(
    Engagement = Likes_Received_Per_Day + Comments_Received_Per_Day
  )

social_data$Dominant_Emotion <- as.numeric(factor(social_data$Dominant_Emotion,
                                                  levels = c("Happiness", "Anger", "Boredom", "Neutral", "Anxiety", "Sadness")))

stan_social_data <- list( # data for stan model 
  N = nrow(social_data),
  X = social_data$Daily_Usage_Time,
  y = social_data$Engagement, emotion = as.integer(social_data$Dominant_Emotion)
)

#Extract the prior predictive samples + plot the prior predictive distribution-
stan_prior_data <- list(
  N = nrow(social_data),
  X = social_data$Daily_Usage_Time
)

fit_prior_check <- stan(file = "Prior_Pred_Check_Q2.stan", data = stan_prior_data,
                       iter = 1000, chains = 4, seed = 123, init = 0)
prior_samples <- extract(prior_check_Q2)
y_prior_rep <- prior_samples$y_rep
emotion_prior_rep <- prior_samples$emotion_rep
ppc_dens_overlay(y = social_data$Engagement, yrep = y_prior_rep[1:100, ])
table(emotion_prior_rep)

# Model fitting and posterior predictive check to assess the fit of the model by comparing predictions to actual data
fit_model_q2 <- stan(file = "stan_code_q2.stan", data = stan_social_data, # fitting the model
            iter = 1000, chains = 4)
# convergence metrics-
print(fit_model_q2, pars = c("alpha", "beta", "sigma", "beta_emotion[1]", "beta_emotion[2]", "beta_emotion[3]",
                             "beta_emotion[4]", "beta_emotion[5]", "beta_emotion[6]", "alpha_emotion[1]", "alpha_emotion[2]",
                             "alpha_emotion[3]", "alpha_emotion[4]", "alpha_emotion[5]", "alpha_emotion[6]",
                             "beta_interaction[1]", "beta_interaction[2]", "beta_interaction[3]",
                             "beta_interaction[4]", "beta_interaction[5]", "beta_interaction[6]"), probs = c(0.025, 0.5, 0.975))

# Posterior predictive check and a residual plot to examine if our model captures well the data (obs. data VS. pred. values)
samples_q2 <- extract(fit_model_q2)
y_rep <- samples_q2$y_rep
emotion_rep <- samples_q2$emotion_rep
ppc_dens_overlay(y = social_data$Engagement, yrep = y_rep[1:100, ])
table(emotion_rep) # show Dominant Emotion posterior distribution
hist(emotion_rep)
posterior_residuals <- social_data$Engagement - colMeans(y_rep)
plot(posterior_residuals, main= "Residuals plot of Posterior Predictive Check", ylab= "residuals")

###############
# Question 2-
#create a new column that mapping into binary values the platforms under Platform column, conditioned by whether the platform is text-based (0) or visual-based(1)
social_data$Platform_Type <- case_when(social_data$Platform %in% c("Twitter", "Telegram", "Whatsapp", "LinkedIn") ~ 0,
                                       social_data$Platform %in% c("Facebook", "Instagram", "Snapchat") ~ 1, TRUE~ NA_real_)

library(loo)
platforms_time_data_list <- list( #data to be used in the stan modeling ahead in the code
  N = nrow(social_data),
  Daily_Usage_Time = social_data$Daily_Usage_Time,
  Platform_Type = social_data$Platform_Type
)

# fitting the null model (representing H0 which assumes that Daily usage time isn't affected by type of the platform)
fit_null_model <- stan(file= "null_model_q2.stan", data = platforms_time_data_list, iter = 1000, chains = 4)

# fitting the alternative model (H1= the platform's type is affecting the Daily usage time)
fit_alternative_model <- stan(file= "alternative_model_q2.stan", data = platforms_time_data_list, iter=1000, chains = 4)

# Examine the convergence metrics of both models:
print(summary(fit_null_model)$summary[, c("mean", "sd", "2.5%", "97.5%", "n_eff", "Rhat")])
print(summary(fit_alternative_model)$summary[, c("mean", "sd", "2.5%", "97.5%", "n_eff", "Rhat")])

#calculation of WAIC and models comparison through these values-
library(loo)
loglikeli_null_model <- extract_log_lik(fit_null_model, parameter_name = "log_lik")
loglikeli_alt_model <- extract_log_lik(fit_alternative_model, parameter_name = "log_lik")
waic_null_model <- waic(loglikeli_null_model)
waic_alt_model <- waic(loglikeli_alt_model)

waic_models_comparison <- loo_compare(waic_null_model, waic_alt_model)

#plot the results received above in a sensible way that shows the difference between the estimates-
posterior_samples_platforms <- extract(fit_alternative_model)
mu_0_samples <- posterior_samples_platforms$mu_0
mu_1_samples <- posterior_samples_platforms$mu_1

posterior_df_platforms <- data.frame(
  mu_value = c(mu_0_samples, mu_1_samples),
  Platform_Type = rep(c("Text-based", "Visual-based"), each = length(mu_0_samples))
)

ggplot(posterior_df_platforms, aes(x = mu_value, fill = Platform_Type)) +
  geom_density(alpha = 0.6) + 
  geom_vline(xintercept = mean(mu_0_samples), linetype = "solid", color = "blue", linewidth = 0.8) +
  geom_vline(xintercept = mean(mu_1_samples), linetype = "solid", color = "red", linewidth = 0.8) +
  labs(title = "Posterior Distributions of Daily Usage Time by Platform Type",
       x = "Daily Usage Time (min)", y = "Density") +
  scale_fill_manual(values = c("Text-based" = "blue", "Visual-based" = "red")) +
  theme_minimal()

##########
#Question 3-
social_data_filtered <- subset(social_data, Gender %in% c("Female", "Male")) #create new df to contain only Males and Females (324 users omitted)
social_data_filtered$Gender <- ifelse(social_data_filtered$Gender == "Female", 1, 0) #convert the Gender's values into dummy where 0 indicates Male and 1 females

#prepare the data lists for the stan models fitting ahead-
stan_data_no_diff <- list(
  N = nrow(social_data_filtered),
  Daily_Usage_Time = social_data_filtered$Daily_Usage_Time,
  Posts_Per_Day = social_data_filtered$Posts_Per_Day,
  Messages_Sent_Per_Day = social_data_filtered$Messages_Sent_Per_Day)

stan_data_gender_diff <- list(
  N = nrow(social_data_filtered),
  Daily_Usage_Time = social_data_filtered$Daily_Usage_Time,
  Posts_Per_Day = social_data_filtered$Posts_Per_Day,
  Messages_Sent_Per_Day = social_data_filtered$Messages_Sent_Per_Day,
  Gender = social_data_filtered$Gender)

#fitting the models to the observed filtered data
no_gendiff_model <- stan(file = "nullmod_genderdif.stan", data = stan_data_no_diff, iter = 1000, chains = 4)
gender_differences_model <- stan(file = "h1model_genderdif.stan", data = stan_data_gender_diff, iter = 1000, chains = 4)
print(summary(no_gendiff_model)$summary[, c("mean", "sd", "n_eff", "Rhat")])
print(summary(gender_differences_model)$summary[, c("mean", "sd", "n_eff", "Rhat")])


posterior_gendiff_samples <- extract(gender_differences_model)
#calculating the probability of the hypothesis that posts are more common activity of females rather then for males
probability_posting_isgreater_females <- mean(posterior_gendiff_samples$beta_posts_female > posterior_gendiff_samples$beta_posts_male)
probability_msging_isgreater_males <- mean(posterior_gendiff_samples$beta_messages_male > posterior_gendiff_samples$beta_messages_female)
cat("P(tendency of posting among female is greater then among males:", probability_posting_isgreater_females, "\n")
cat("P(tendency of messaging is greater among men them among females:", probability_msging_isgreater_males, "\n")
#calculate also BF to compare the whole models representing the hypothesis-
install.packages("bridgesampling")
library(bridgesampling)
bridge_no_diff_model <- bridge_sampler(no_gendiff_model)
bridge_gendiff_model <- bridge_sampler(gender_differences_model)
bayes_factor_gendiff <- bf(bridge_gendiff_model, bridge_no_diff_model)
print(bayes_factor_gendiff)

