SELECT transaction_timestamp();

BEGIN;

DROP TABLE IF EXISTS ratings CASCADE;

CREATE TABLE ratings (
    customerid integer ,
    rating numeric,
    date DATE,
    movieid integer
);

\copy ratings FROM '../data/ratings.csv'  WITH (HEADER false, FORMAT csv);

COMMIT;

SELECT transaction_timestamp();