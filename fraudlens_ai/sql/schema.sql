-- Minimal relational model for a transaction-risk analytics project.
-- Keep direct identifiers out of the analytics layer; use a pseudonymous customer key.
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGINT PRIMARY KEY,
    customer_key VARCHAR(128),
    transaction_ts TIMESTAMP,
    amount NUMERIC(14,2) NOT NULL CHECK (amount >= 0),
    merchant_category VARCHAR(100),
    device_type VARCHAR(100),
    is_international BOOLEAN,
    is_card_present BOOLEAN,
    is_fraud BOOLEAN,
    risk_probability NUMERIC(8,7),
    risk_band VARCHAR(16),
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transactions_ts ON transactions(transaction_ts);
CREATE INDEX IF NOT EXISTS idx_transactions_risk ON transactions(risk_band);
