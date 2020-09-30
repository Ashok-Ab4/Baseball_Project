#!/usr/bin/env python3
# Setup Spark
import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from RollingAvg import RollingAverage


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball"
    port = "3306"
    user = "root"
    password = ""

    # loading in the batter_counts table joined with game table

    df = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}",
            driver="org.mariadb.jdbc.Driver",
            dbtable="(SELECT Hit ,atBat ,batter, b.game_id , date(local_date) as l_date \
            FROM batter_counts b JOIN game g ON b.game_id = g.game_id)batters",
            user=user,
            password=password,
        )
        .load()
    )

    df.show()

    # Loading the table into a temporary View so that it can be reused for the RolingAvg calculations

    df.createOrReplaceTempView("batters")
    df.persist(StorageLevel.DISK_ONLY)
    # Calculating rolling average

    op = spark.sql(
        """(SELECT SUM(ba.Hit) as Hits, SUM(ba.atbat) as AtBat, b.game_id,b.batter,count(*) \
        FROM batters ba JOIN batters b ON ba.batter = b.batter \
        AND ba.l_date BETWEEN DATE_SUB(b.l_date, 100 ) AND DATE_SUB(b.l_date, 1 ) \
        GROUP BY b.game_id, b.batter \
        ORDER BY b.game_id)"""
    )

    # Applying the transformer
    RollAvg = RollingAverage(inputCols=["Hits", "AtBat"], outputCol="Rolling_Average")
    fin_RollAvg = RollAvg.transform(op)
    fin_RollAvg.show()


if __name__ == "__main__":
    sys.exit(main())
