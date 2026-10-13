SELECT country, SUM(quantity) AS units_sold,
       SUM(gross_sales) AS gross_sales_gbp, COUNT(DISTINCT order_id) AS orders_count
FROM sales_clean GROUP BY country ORDER BY units_sold DESC;
