CREATE SCHEMA IF NOT EXISTS hard_drive;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS predictions;

CREATE TABLE IF NOT EXISTS hard_drive.smart_daily (
    date DATE NOT NULL,
    serial_number VARCHAR(50) NOT NULL,
    model VARCHAR(100),
    capacity_bytes BIGINT,
    failure SMALLINT DEFAULT 0,
    
    smart_1_raw BIGINT,
    smart_2_raw BIGINT,
    smart_3_raw BIGINT,
    smart_4_raw BIGINT,
    smart_5_raw BIGINT,
    smart_7_raw BIGINT,
    smart_8_raw BIGINT,
    smart_9_raw BIGINT,
    smart_10_raw BIGINT,
    smart_12_raw BIGINT,
    smart_183_raw BIGINT,
    smart_184_raw BIGINT,
    smart_187_raw BIGINT,
    smart_188_raw BIGINT,
    smart_189_raw BIGINT,
    smart_190_raw BIGINT,
    smart_191_raw BIGINT,
    smart_192_raw BIGINT,
    smart_193_raw BIGINT,
    smart_194_raw BIGINT,
    smart_195_raw BIGINT,
    smart_196_raw BIGINT,
    smart_197_raw BIGINT,
    smart_198_raw BIGINT,
    smart_199_raw BIGINT,
    smart_200_raw BIGINT,
    smart_220_raw BIGINT,
    smart_222_raw BIGINT,
    smart_223_raw BIGINT,
    smart_224_raw BIGINT,
    smart_225_raw BIGINT,
    smart_226_raw BIGINT,
    smart_240_raw BIGINT,
    smart_241_raw BIGINT,
    smart_242_raw BIGINT,
    smart_250_raw BIGINT,
    smart_251_raw BIGINT,
    smart_252_raw BIGINT,
    smart_254_raw BIGINT,
    smart_255_raw BIGINT,
    
    smart_1_normalized SMALLINT,
    smart_5_normalized SMALLINT,
    smart_9_normalized SMALLINT,
    smart_187_normalized SMALLINT,
    smart_194_normalized SMALLINT,
    smart_197_normalized SMALLINT,
    smart_198_normalized SMALLINT,
    
    PRIMARY KEY (date, serial_number)
) PARTITION BY RANGE (date);

CREATE TABLE IF NOT EXISTS hard_drive.smart_daily_2025_q1 
    PARTITION OF hard_drive.smart_daily
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

CREATE TABLE IF NOT EXISTS hard_drive.smart_daily_2025_q2 
    PARTITION OF hard_drive.smart_daily
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');

CREATE TABLE IF NOT EXISTS hard_drive.smart_daily_2025_q3 
    PARTITION OF hard_drive.smart_daily
    FOR VALUES FROM ('2025-07-01') TO ('2025-10-01');

CREATE TABLE IF NOT EXISTS hard_drive.smart_daily_2025_q4 
    PARTITION OF hard_drive.smart_daily
    FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS hard_drive.drive_registry (
    serial_number VARCHAR(50) PRIMARY KEY,
    model VARCHAR(100),
    capacity_bytes BIGINT,
    first_seen DATE,
    last_seen DATE,
    total_days INT DEFAULT 0,
    failed BOOLEAN DEFAULT FALSE,
    failure_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features.drive_features (
    id BIGSERIAL PRIMARY KEY,
    serial_number VARCHAR(50) NOT NULL,
    feature_date DATE NOT NULL,
    
    smart_5_mean_7d FLOAT,
    smart_5_std_7d FLOAT,
    smart_5_max_7d FLOAT,
    smart_5_delta_7d FLOAT,
    smart_187_mean_7d FLOAT,
    smart_187_delta_7d FLOAT,
    smart_197_mean_7d FLOAT,
    smart_197_delta_7d FLOAT,
    smart_198_mean_7d FLOAT,
    smart_198_delta_7d FLOAT,
    
    smart_5_mean_30d FLOAT,
    smart_5_trend_30d FLOAT,
    smart_187_mean_30d FLOAT,
    smart_197_mean_30d FLOAT,
    
    power_on_hours BIGINT,
    temperature_mean FLOAT,
    age_days INT,
    
    days_to_failure INT,
    will_fail_30d BOOLEAN DEFAULT FALSE,
    will_fail_60d BOOLEAN DEFAULT FALSE,
    
    UNIQUE(serial_number, feature_date)
);

CREATE TABLE IF NOT EXISTS predictions.failure_predictions (
    id BIGSERIAL PRIMARY KEY,
    serial_number VARCHAR(50) NOT NULL,
    prediction_date DATE NOT NULL,
    model_version VARCHAR(50),
    failure_prob_30d FLOAT,
    failure_prob_60d FLOAT,
    risk_category VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_smart_serial ON hard_drive.smart_daily(serial_number);
CREATE INDEX IF NOT EXISTS idx_smart_model ON hard_drive.smart_daily(model);
CREATE INDEX IF NOT EXISTS idx_smart_failure ON hard_drive.smart_daily(failure) WHERE failure = 1;
CREATE INDEX IF NOT EXISTS idx_registry_failed ON hard_drive.drive_registry(failed) WHERE failed = TRUE;
CREATE INDEX IF NOT EXISTS idx_features_serial ON features.drive_features(serial_number);

DO $$
BEGIN
    RAISE NOTICE 'All tables created successfully!';
END $$;