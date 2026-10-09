-- 1. Fraud rate and transaction volume by month
SELECT DATE_TRUNC('month', transaction_ts) AS month,
       COUNT(*) AS transactions,
       SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_transactions,
       ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END), 3) AS fraud_rate_pct
FROM transactions
WHERE is_fraud IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- 2. Review queue size by risk band
SELECT risk_band, COUNT(*) AS transaction_count,
       ROUND(SUM(amount), 2) AS transaction_value
FROM transactions
GROUP BY risk_band
ORDER BY CASE risk_band WHEN 'High' THEN 1 WHEN 'Review' THEN 2 ELSE 3 END;

-- 3. Precision of the high-risk queue (requires adjudicated labels)
SELECT COUNT(*) AS reviewed_transactions,
       SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS confirmed_fraud,
       ROUND(AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END), 4) AS confirmed_fraud_rate
FROM transactions
WHERE risk_band = 'High' AND is_fraud IS NOT NULL;

-- 4. Compare fraud rate across merchant categories
SELECT merchant_category, COUNT(*) AS transactions,
       ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END), 3) AS fraud_rate_pct
FROM transactions
WHERE is_fraud IS NOT NULL
GROUP BY merchant_category
HAVING COUNT(*) >= 50
ORDER BY fraud_rate_pct DESC;
