graphics.off()
rm(list = ls())

set.seed(42)
N <- 8760
hours <- 1:N
day_of_year <- ((hours-1)%/%24)+1
hour_of_day <- ((hours-1)%%24)+1

rain_rate <- abs(arima.sim(list(ar=0.98), n=N))*2 * (1 + 0.4*sin(2*pi*day_of_year/365 - pi/2))
temp <- 25 + 8*sin(2*pi*day_of_year/365 - pi/2) + 2*sin(2*pi*hour_of_day/24 - pi/3) + rnorm(N,0,0.5)
wind <- abs(arima.sim(list(ar=0.95), n=N)) + 3

signal_loss <- 20 + (0.00454*(rain_rate^1.353)*10) + (0.0003*temp) + (0.05*wind) + rnorm(N,0,0.3)

mw_data <- data.frame(Hour=hours, Rain=rain_rate, Temp=temp, Wind=wind, Loss=signal_loss)

mw_data$Loss_Lag1 <- c(NA, mw_data$Loss[-N])
mw_data$Rain_Lag1 <- c(NA, mw_data$Rain[-N])
mw_data$Next_Loss <- c(mw_data$Loss[-1], NA)
mw_data <- na.omit(mw_data)

split_idx <- floor(0.8 * nrow(mw_data))
train_data <- mw_data[1:split_idx,]
test_data  <- mw_data[(split_idx+1):nrow(mw_data),]

model <- lm(Next_Loss ~ Rain + Temp + Wind + Rain_Lag1 + Loss_Lag1, data = train_data)

Actual <- test_data$Next_Loss
Predicted <- predict(model, test_data)
Residuals <- Predicted - Actual

MAE <- mean(abs(Residuals))
RMSE <- sqrt(mean(Residuals^2))
MAPE <- mean(abs(Residuals / Actual)) * 100
R2 <- 1 - (sum(Residuals^2) / sum((Actual - mean(Actual))^2))

cat(sprintf("R-squared (%%) : %.2f %%\n", R2 * 100))
cat(sprintf("MAE (Unit)    : %.4f\n", MAE))
cat(sprintf("RMSE (Unit)   : %.4f\n", RMSE))
cat(sprintf("MAPE (%%)      : %.2f %%\n", MAPE))
cat("===========================================\n\n")

par(mfrow=c(1,1), bg="white", col.axis="black", col.lab="black", col.main="black", mar=c(5, 5, 4, 2) + 0.1)

plot(Actual, Predicted, main="1. Actual vs Predicted Loss",
     xlab="Actual Loss (dB)", ylab="Predicted Loss (dB)",
     pch=16, col=rgb(0.2, 0.4, 0.8, 0.4), cex=1.2)
abline(a=0, b=1, col="red", lwd=3, lty=2)

hist(Residuals, breaks=50, col="darkgray", border="white",
     main="2. Residual Distribution", xlab="Residual Error (dB)", ylab="Frequency")
abline(v=0, col="red", lwd=3, lty=2)

qqnorm(Residuals, main="3. Normal Q-Q Plot", col=rgb(0.2, 0.2, 0.2, 0.5), pch=16)
qqline(Residuals, col="red", lwd=3, lty=2)

plot(Residuals, type="l", col=rgb(0, 0, 0, 0.4),
     main="4. Residuals Over Time (Test Set)", xlab="Time Index", ylab="Residual (dB)")
abline(h=0, col="red", lwd=3)

coefs <- summary(model)$coefficients[-1, "Estimate"]
par(mar=c(5, 8, 4, 2) + 0.1)

barplot(sort(coefs), horiz=TRUE, col="steelblue",
        main="5. Feature Coefficients (Importance)", xlab="Estimate Value", las=1)
abline(v=0, col="red", lwd=3, lty=2)