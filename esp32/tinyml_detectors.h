/*
 * ==========================================================================
 * tinyml_detectors.h -- TinyML Anomaly Detection for ESP32
 * ==========================================================================
 * Auto-generated from Python training pipeline.
 * Thresholds calibrated on real HC-SR04 sensor data.
 *
 * Total RAM usage: ~558  bytes (~350 bytes)
 * Total inference:  ~30 us per reading (all 3 detectors)
 *
 * Usage:
 *   #include "tinyml_detectors.h"
 *   MASD_State  masd;   masd_init(&masd);
 *   CUSUM_State cusum;  cusum_init(&cusum, first_reading);
 *   float prev = first_reading;
 *
 *   // In sensor loop:
 *   int severity = detect_all(reading, &masd, &cusum, &prev);
 *   // severity: 0=normal, 1=low, 2=medium, 3=high
 * ==========================================================================
 */

#ifndef TINYML_DETECTORS_H
#define TINYML_DETECTORS_H

#include <math.h>

/* -- Calibrated Thresholds ----------------------------------- */
#define MASD_WINDOW     100
#define MASD_N_STD      2.5f
#define ROC_DROP_THR    -1.0f
#define ROC_RISE_THR    1.0f
#define CUSUM_ALPHA     0.15f
#define CUSUM_K         0.5f
#define CUSUM_H         10.0f

/* -- MA+SD Detector ---------------------------------------- */
/* RAM: 100 * 4 bytes (buffer) + 16 bytes (state) = ~416 bytes */
typedef struct {
    float buffer[MASD_WINDOW];
    int   count;
    int   idx;
    float sum;
    float sq_sum;
} MASD_State;

static inline void masd_init(MASD_State *s) {
    s->count  = 0;
    s->idx    = 0;
    s->sum    = 0.0f;
    s->sq_sum = 0.0f;
    for (int i = 0; i < MASD_WINDOW; i++) s->buffer[i] = 0.0f;
}

static inline int masd_detect(MASD_State *s, float reading) {
    /* Remove oldest value if buffer is full */
    if (s->count >= MASD_WINDOW) {
        float old = s->buffer[s->idx];
        s->sum    -= old;
        s->sq_sum -= old * old;
    }

    /* Add new value */
    s->buffer[s->idx] = reading;
    s->sum    += reading;
    s->sq_sum += reading * reading;
    s->idx     = (s->idx + 1) % MASD_WINDOW;
    if (s->count < MASD_WINDOW) s->count++;

    /* Compute mean and std */
    float mean = s->sum / s->count;
    float var  = (s->sq_sum / s->count) - (mean * mean);
    float std  = (var > 0) ? sqrtf(var) : 0.001f;

    /* Flag if deviation exceeds threshold */
    float deviation = fabsf(reading - mean);
    return (deviation > MASD_N_STD * std) ? 1 : 0;
}

/* -- Rate of Change Detector ------------------------------- */
/* RAM: 4 bytes (previous reading) */
static inline int roc_detect(float current, float *prev) {
    float roc = current - *prev;
    *prev = current;
    return (roc < ROC_DROP_THR || roc > ROC_RISE_THR) ? 1 : 0;
}

/* -- Adaptive CUSUM Detector ------------------------------- */
/* RAM: 12 bytes (3 floats) */
typedef struct {
    float baseline;
    float pos;
    float neg;
} CUSUM_State;

static inline void cusum_init(CUSUM_State *s, float first_val) {
    s->baseline = first_val;
    s->pos      = 0.0f;
    s->neg      = 0.0f;
}

static inline int cusum_detect(CUSUM_State *s, float reading) {
    /* Update adaptive baseline (EWMA) */
    s->baseline = CUSUM_ALPHA * reading + (1.0f - CUSUM_ALPHA) * s->baseline;

    /* Accumulate positive and negative drift */
    s->pos = fmaxf(0.0f, s->pos + (reading - s->baseline - CUSUM_K));
    s->neg = fmaxf(0.0f, s->neg + (s->baseline - reading - CUSUM_K));

    /* Check threshold */
    int flag = (s->pos > CUSUM_H || s->neg > CUSUM_H) ? 1 : 0;

    /* Reset on detection */
    if (flag) {
        s->pos = 0.0f;
        s->neg = 0.0f;
    }

    return flag;
}

/* -- Combined Detection ------------------------------------ */
/* Returns severity: 0 (normal), 1 (low), 2 (medium), 3 (high) */
static inline int detect_all(float reading,
                             MASD_State *masd,
                             CUSUM_State *cusum,
                             float *prev_reading) {
    int masd_flag  = masd_detect(masd, reading);
    int roc_flag   = roc_detect(reading, prev_reading);
    int cusum_flag = cusum_detect(cusum, reading);

    return masd_flag + roc_flag + cusum_flag;
}

#endif /* TINYML_DETECTORS_H */
