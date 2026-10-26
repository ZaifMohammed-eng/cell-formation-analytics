-- Daily Formation KPIs by grading day: volume, first-pass and final yield,
-- and fail rate by failure mode.
SELECT
    grade_day,
    COUNT(*)                                   AS cells_graded,
    AVG(first_pass::INT)                       AS first_pass_yield,
    AVG(final_pass::INT)                       AS final_yield,
    SUM(fail_k::INT)                           AS n_fail_k,
    SUM(fail_cap::INT)                         AS n_fail_cap,
    SUM(fail_rls::INT)                         AS n_fail_rls,
    AVG((protocol = 'fast')::INT)              AS fast_share
FROM cell_disposition
GROUP BY grade_day
ORDER BY grade_day;
