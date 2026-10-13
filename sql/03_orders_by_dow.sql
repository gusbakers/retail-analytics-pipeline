SELECT order_dow,
  CASE order_dow WHEN 0 THEN 'Mon' WHEN 1 THEN 'Tue' WHEN 2 THEN 'Wed'
    WHEN 3 THEN 'Thu' WHEN 4 THEN 'Fri' WHEN 5 THEN 'Sat' WHEN 6 THEN 'Sun' END AS day_name,
  COUNT(DISTINCT order_id) AS total_orders
FROM sales_clean GROUP BY order_dow ORDER BY order_dow;

