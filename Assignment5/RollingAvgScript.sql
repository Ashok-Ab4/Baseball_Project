USE baseball;

#Rolling average

CREATE TABLE IF NOT EXISTS batters AS
SELECT batter,atBat , Hit , b.game_id , local_date
FROM batter_counts b JOIN game g ON b.game_id = g.game_id 
ORDER BY game_id ,local_date ;

CREATE OR REPLACE TABLE RollingBattingAvg AS
SELECT SUM(IFNULL(HistBatters.Hit,0))/NULLIF(SUM(IFNULL(HistBatters.atBat,0)),0) as RollingAvg, CurrBatters.local_date as Local_date, CurrBatters.batter, CurrBatters.game_id , count(*) as cnt
FROM batters CurrBatters JOIN batters HistBatters
ON CurrBatters.batter = HistBatters.batter AND HistBatters.local_date > DATE_SUB(CurrBatters.local_date,interval 100 DAY) AND HistBatters.local_date < CurrBatters.local_date
WHERE CurrBatters.game_id = 12560 #(remove comment to limit values to a particular game)
GROUP BY CurrBatters.game_id,CurrBatters.batter,CurrBatters.local_date;

#selecting values from the RollingAvg table
SELECT * FROM RollingBattingAvg
Into Outfile './RollingAvg.csv';
