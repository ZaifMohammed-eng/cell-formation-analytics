-- Recompute the K-value (self-discharge, mV/day) from raw OCV readings and
-- their timestamps, and compare with the value the ETL stored.
-- A disagreement means a calculation problem, not a cell problem.
SELECT
    grade_day,
    COUNT(*) AS cells,
    AVG((ABS(k_stored_mv_day - k_raw) > 1e-6 * GREATEST(ABS(k_raw), 1e-3))::INT)
        AS mismatch_rate,
    MEDIAN(k_stored_mv_day / NULLIF(k_raw, 0)) AS median_stored_over_raw
FROM (
    SELECT
        grade_day, k_stored_mv_day,
        (ocv1_v - ocv2_v) * 1000.0
          / ((epoch(ocv2_ts) - epoch(ocv1_ts)) / 86400.0) AS k_raw
    FROM cells
)
GROUP BY grade_day
ORDER BY grade_day;
