-- Weekly output and cycle time (cell build to grading complete) by protocol.
SELECT
    grade_day // 7                                         AS week,
    protocol,
    COUNT(*)                                               AS cells_graded,
    AVG(final_pass::INT)                                   AS final_yield,
    AVG((epoch(grade_ts) - epoch(build_ts)) / 3600.0)      AS cycle_time_h,
    AVG(formation_hours)                                   AS formation_h
FROM cell_disposition
GROUP BY ALL
ORDER BY week, protocol;
