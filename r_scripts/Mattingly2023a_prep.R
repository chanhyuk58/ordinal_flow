rm(list = ls())

# Import survey data
load("../replication_data/clean.RData")

covs = ~ country + gender + age + education + national_pride + leftright

clean <- clean[, c("political_model", "econ_model", "treatment", "world_leader", "country", "gender", "age", "education", "national_pride", "leftright")]
clean <- na.omit(clean)
write.csv(clean, "../replication_data/Mattingly2023a.csv", row.names=FALSE)
