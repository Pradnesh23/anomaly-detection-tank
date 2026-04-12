/*
 * ==========================================================================
 * prophet_seasonal.h -- Prophet's Learned Patterns for ESP32
 * ==========================================================================
 * Auto-generated from trained Prophet model.
 * Contains daily seasonality as a 24-value lookup table.
 * RAM usage: 96 bytes (24 floats) + 8 bytes (trend) = 104 bytes
 * ==========================================================================
 */

#ifndef PROPHET_SEASONAL_H
#define PROPHET_SEASONAL_H

/* Daily hourly effect (additive, relative to mean) */
const float DAILY_SEASONAL[24] = {
    -7.5135f, -4.1303f, -8.5978f, -14.4973f, -13.9059f, -6.5059f, -0.7029f, -3.8885f, -12.5750f, -12.7188f, 6.4586f, 39.9577f,
    67.6562f, 69.5689f, 42.5526f, 3.5770f, -23.7043f, -28.0050f, -16.5584f, -5.9923f, -7.6869f, -15.7189f, -20.2331f, -15.9246f
};

/* Linear trend parameters */
#define PROPHET_TREND_SLOPE     0.000359f
#define PROPHET_TREND_INTERCEPT 58.5398f

/* Predict expected reading for a given hour */
static inline float predict_expected(int hour, int minutes_since_start) {
    float trend    = PROPHET_TREND_INTERCEPT +
                     PROPHET_TREND_SLOPE * (float)minutes_since_start;
    float seasonal = DAILY_SEASONAL[hour % 24];
    return trend + seasonal;
}

#endif /* PROPHET_SEASONAL_H */
