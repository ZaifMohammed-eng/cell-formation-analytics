-- First-pass grading result per cell against the limits table.
CREATE OR REPLACE VIEW cell_results AS
SELECT
    c.*,
    c.k_stored_mv_day > l.k_usl_mv_day AS fail_k,
    c.cap_meas_ah     < l.cap_lsl_ah   AS fail_cap,
    c.rls_meas_mohm   > l.rls_usl_mohm AS fail_rls,
    NOT (c.k_stored_mv_day > l.k_usl_mv_day
         OR c.cap_meas_ah < l.cap_lsl_ah
         OR c.rls_meas_mohm > l.rls_usl_mohm) AS first_pass
FROM cells c CROSS JOIN limits l;

-- Final disposition after one retest of each failed cell.
CREATE OR REPLACE VIEW cell_disposition AS
SELECT
    r.*,
    t.cell_id IS NOT NULL AS retested,
    CASE
        WHEN r.first_pass THEN TRUE
        WHEN t.cell_id IS NULL THEN FALSE
        ELSE NOT (t.retest_k_mv_day > l.k_usl_mv_day
                  OR t.retest_cap_ah < l.cap_lsl_ah
                  OR t.retest_rls_mohm > l.rls_usl_mohm)
    END AS final_pass
FROM cell_results r
LEFT JOIN retests t USING (cell_id)
CROSS JOIN limits l;
