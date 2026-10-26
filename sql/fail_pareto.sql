-- Pareto of first-pass fails by mode over the whole run.
SELECT mode, n_fail, n_fail / SUM(n_fail) OVER () AS share
FROM (
    SELECT 'K_HIGH' AS mode, SUM(fail_k::INT) AS n_fail FROM cell_results
    UNION ALL SELECT 'CAP_LOW', SUM(fail_cap::INT) FROM cell_results
    UNION ALL SELECT 'RLS_HIGH', SUM(fail_rls::INT) FROM cell_results
)
ORDER BY n_fail DESC;
