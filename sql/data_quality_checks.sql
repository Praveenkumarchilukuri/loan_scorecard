-- sql/data_quality_checks.sql
-- Standalone SQL validation scripts for the Loan Scorecard pipeline.
-- These mirror the DuckDB checks in 01_data_quality.py.
-- Can be run directly against any SQL engine (DuckDB, SQLite, PostgreSQL)
-- that has an 'applicants' table loaded from the CSV.
--
-- Usage with DuckDB CLI:
--   duckdb -c "CREATE TABLE applicants AS SELECT * FROM read_csv_auto('data/cs-training.csv');
--              .read sql/data_quality_checks.sql"

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. NULL CHECKS
-- ─────────────────────────────────────────────────────────────────────────────

-- Critical: target must never be null
SELECT
    'null_target'       AS check_name,
    'CRITICAL'          AS severity,
    COUNT(*)            AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "SeriousDlqin2yrs" IS NULL;

-- Critical: age must never be null
SELECT
    'null_age'          AS check_name,
    'CRITICAL'          AS severity,
    COUNT(*)            AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE age IS NULL;

-- Warning: monthly income nulls are expected (imputed later)
SELECT
    'null_monthly_income' AS check_name,
    'WARNING'             AS severity,
    COUNT(*)              AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "MonthlyIncome" IS NULL;

-- Warning: dependents nulls
SELECT
    'null_dependents'   AS check_name,
    'WARNING'           AS severity,
    COUNT(*)            AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "NumberOfDependents" IS NULL;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. RANGE VALIDATION
-- ─────────────────────────────────────────────────────────────────────────────

-- Age must be between 18 and 110
SELECT
    'age_range'         AS check_name,
    'CRITICAL'          AS severity,
    COUNT(*)            AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE age < 18 OR age > 110;

-- Revolving utilization should be non-negative
-- (very high values are capped in preprocessing)
SELECT
    'revolving_util_range' AS check_name,
    'WARNING'              AS severity,
    COUNT(*)               AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "RevolvingUtilizationOfUnsecuredLines" < 0
   OR "RevolvingUtilizationOfUnsecuredLines" > 50;

-- Debt ratio must be non-negative
SELECT
    'debt_ratio_negative' AS check_name,
    'CRITICAL'            AS severity,
    COUNT(*)              AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "DebtRatio" < 0;

-- Past due counts must be non-negative
SELECT
    'dpd_30_59_negative' AS check_name,
    'CRITICAL'           AS severity,
    COUNT(*)             AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "NumberOfTime30-59DaysPastDueNotWorse" < 0;

SELECT
    'dpd_60_89_negative' AS check_name,
    'CRITICAL'           AS severity,
    COUNT(*)             AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "NumberOfTime60-89DaysPastDueNotWorse" < 0;

SELECT
    'dpd_90_negative'   AS check_name,
    'CRITICAL'          AS severity,
    COUNT(*)            AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "NumberOfTimes90DaysLate" < 0;

-- Monthly income must not be negative
SELECT
    'income_negative'   AS check_name,
    'CRITICAL'          AS severity,
    COUNT(*)            AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "MonthlyIncome" < 0;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. REFERENTIAL INTEGRITY / DOMAIN CHECKS
-- ─────────────────────────────────────────────────────────────────────────────

-- Target must be binary (0 or 1)
SELECT
    'target_binary'     AS check_name,
    'CRITICAL'          AS severity,
    COUNT(*)            AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "SeriousDlqin2yrs" NOT IN (0, 1);

-- Open credit lines must be non-negative
SELECT
    'open_lines_negative' AS check_name,
    'WARNING'             AS severity,
    COUNT(*)              AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "NumberOfOpenCreditLinesAndLoans" < 0;

-- Real estate loans must be non-negative
SELECT
    'real_estate_negative' AS check_name,
    'WARNING'              AS severity,
    COUNT(*)               AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "NumberRealEstateLoansOrLines" < 0;

-- Dependents must be non-negative
SELECT
    'dependents_negative' AS check_name,
    'WARNING'             AS severity,
    COUNT(*)              AS fail_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicants) AS fail_pct
FROM applicants
WHERE "NumberOfDependents" < 0;

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. SUMMARY PROFILE QUERY
-- ─────────────────────────────────────────────────────────────────────────────

-- Full data profile overview
SELECT
    COUNT(*)                                    AS total_rows,
    SUM("SeriousDlqin2yrs")                     AS total_defaults,
    AVG("SeriousDlqin2yrs") * 100               AS default_rate_pct,
    AVG(age)                                    AS avg_age,
    MEDIAN(age)                                 AS median_age,
    AVG("MonthlyIncome")                        AS avg_monthly_income,
    MEDIAN("MonthlyIncome")                     AS median_monthly_income,
    SUM(CASE WHEN "MonthlyIncome" IS NULL THEN 1 ELSE 0 END) AS null_income_count,
    SUM(CASE WHEN "NumberOfDependents" IS NULL THEN 1 ELSE 0 END) AS null_dep_count,
    AVG("RevolvingUtilizationOfUnsecuredLines") AS avg_revolving_util,
    AVG("DebtRatio")                            AS avg_debt_ratio
FROM applicants;
