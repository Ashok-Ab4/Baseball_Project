USE baseball;

#Historic batting average
CREATE OR REPLACE TABLE HistBattingAvg(batter int, battingaverage float) ENGINE=MyISAM
SELECT batter,SUM(IFNULL(Hit,0))/NULLIF(SUM(IFNULL(atBat,0)),0) AS battingavg 
FROM batter_counts 
GROUP BY batter;

#select values from historing average table
SELECT * FROM HistBattingAvg;

#annual batting average
CREATE OR REPLACE TABLE AnnualBattingAvg(batter int, battingaverage float, DateYear int) ENGINE=MyISAM
SELECT batter, SUM(IFNULL(Hit,0))/NULLIF(SUM(IFNULL(atBat,0)),0) AS battingavg, YEAR(local_date)
FROM batter_counts bc JOIN game g2 ON bc.game_id = g2.game_id 
GROUP BY batter,YEAR(local_date)
ORDER BY batter;


#select values from the annual average table
SELECT * FROM AnnualBattingAvg;



#Rolling average

CREATE OR REPLACE TABLE batters 
SELECT batter,atBat , Hit , b.game_id , local_date
FROM batter_counts b JOIN game g ON b.game_id = g.game_id 
ORDER BY batter game_id ,local_date ;

CREATE OR REPLACE TABLE RollingBattingAvg 
SELECT SUM(IFNULL(HistBatters.Hit,0))/NULLIF(SUM(IFNULL(HistBatters.atBat,0)),0) as RollingAvg, CurrBatters.local_date as Local_date, CurrBatters.batter, CurrBatters.game_id , count(*) as cnt
FROM batters CurrBatters JOIN batters HistBatters
ON CurrBatters.batter = HistBatters.batter AND HistBatters.local_date > DATE_SUB(CurrBatters.local_date,interval 100 DAY) AND HistBatters.local_date < CurrBatters.local_date
#WHERE CurrBatters.game_id = 10000 #(remove comment to limit values to a particular game)
GROUP BY CurrBatters.game_id,CurrBatters.batter,CurrBatters.local_date;

#selecting values from the RollingAvg table
SELECT * FROM RollingBattingAvg;