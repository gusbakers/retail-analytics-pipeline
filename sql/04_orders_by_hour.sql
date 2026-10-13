SELECT order_hour, COUNT(DISTINCT order_id) AS total_orders
FROM sales_clean GROUP BY order_hour ORDER BY order_hour;
