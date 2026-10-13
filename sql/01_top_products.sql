SELECT product_id, MAX(product_name) AS product_name,
       SUM(quantity) AS units_sold, COUNT(DISTINCT order_id) AS orders_count
FROM sales_clean
GROUP BY product_id
ORDER BY units_sold DESC, product_id
LIMIT 10;
