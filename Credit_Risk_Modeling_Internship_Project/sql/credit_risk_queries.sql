-- Credit Risk Modeling: SQL Analysis
-- Works with SQLite after importing data/credit_risk.csv as credit_risk.

-- 1. View all applicants
SELECT * FROM credit_risk;

-- 2. Applicants with low credit score
SELECT loan_id, age, income, credit_score, default
FROM credit_risk
WHERE credit_score < 650;

-- 3. Default count
SELECT default, COUNT(*) AS applicant_count
FROM credit_risk
GROUP BY default;

-- 4. Average income by default status
SELECT default, ROUND(AVG(income), 2) AS avg_income
FROM credit_risk
GROUP BY default;

-- 5. Average credit score by default status
SELECT default, ROUND(AVG(credit_score), 2) AS avg_credit_score
FROM credit_risk
GROUP BY default;

-- 6. Default rate by home ownership
SELECT home_ownership,
       COUNT(*) AS total_applicants,
       SUM(default) AS defaults,
       ROUND(100.0 * SUM(default) / COUNT(*), 2) AS default_rate_pct
FROM credit_risk
GROUP BY home_ownership
ORDER BY default_rate_pct DESC;

-- 7. High debt-to-income applicants
SELECT loan_id, income, loan_amount, debt_to_income, default
FROM credit_risk
WHERE debt_to_income >= 0.45
ORDER BY debt_to_income DESC;
