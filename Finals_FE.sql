use baseball;

drop table if exists RollingAvgTable;
create table RollingAvgTable as
select
		tbc1.team_id
		,tbc1.game_id
		,count(*) as cnt
		,sum(tbc2.plateApperance) as plateAppearance
		,sum(tbc2.atBat) as atBat
		,sum(tbc2.finalScore) as RunsScored
		,sum(tbc2.opponent_finalScore) as RunsAllowed
		,sum(tbc2.Hit)as Hit
		,sum(tbc2.caughtStealing2B)as caughtStealing2B
		,sum(tbc2.caughtStealing3B)as caughtStealing3B 
		,sum(tbc2.caughtStealingHome)as caughtStealingHome 
		,sum(tbc2.stolenBase2B)as stolenBase2B 
		,sum(tbc2.stolenBase3B)as stolenBase3B 
		,sum(tbc2.stolenBaseHome)as stolenBaseHome 
		,sum(tbc2.toBase)as toBase 
		,sum(tbc2.Batter_Interference)as Batter_Interference 
		,sum(tbc2.Bunt_Ground_Out)+sum(tbc2.Bunt_Groundout)as Bunt_Ground_Out 
		,sum(tbc2.Bunt_Pop_Out)as Bunt_Pop_Out 
		,sum(tbc2.Catcher_Interference)as Catcher_Interference 
		,sum(tbc2.`Double`)as `Double` 
		,sum(tbc2.Double_Play)as Double_Play 
		,sum(tbc2.Fan_interference)as Fan_interference 
		,sum(tbc2.Field_Error)as Field_Error 
		,sum(tbc2.Fielders_Choice)as Fielders_Choice 
		,sum(tbc2.Fly_Out)+sum(tbc2.Flyout) as Fly_Out 
		,sum(tbc2.Force_Out)+SUM(tbc2.Forceout)as Force_Out 
		,sum(tbc2.Ground_Out)+SUM(tbc2.Groundout)as Ground_Out 
		,sum(tbc2.Grounded_Into_DP)as Grounded_Into_DP 
		,sum(tbc2.Hit_By_Pitch)as Hit_By_Pitch 
		,sum(tbc2.Home_Run)as Home_Run 
		,sum(tbc2.Intent_Walk)as Intent_Walk 
		,SUM(tbc2.Line_Out)as Line_Out 
		,SUM(tbc2.Pop_Out)as Pop_Out 
		,SUM(tbc2.Runner_Out)as Runner_Out 
		,sum(tbc2.Sac_Bunt)as Sac_Bunt 
		,sum(tbc2.Sac_Fly)as Sac_Fly 
		,sum(tbc2.Sac_Fly_DP)as Sac_Fly_DP 
		,sum(tbc2.Sacrifice_Bunt_DP)as Sacrifice_Bunt_DP 
		,SUM(tbc2.Single)as Single 
		,SUM(tbc2.Strikeout)as Strikeout 
		,SUM(tbc2.`Strikeout_-_DP`)as `Strikeout_-_DP` 
		,SUM(tbc2.`Strikeout_-_TP`)as `Strikeout_-_TP` 
		,sum(tbc2.Triple)as Triple 
		,sum(tbc2.Triple_Play)as Triple_Play 
		,sum(tbc2.Walk)as walk
FROM team_batting_counts tbc1 
join game g1 on tbc1.game_id = g1.game_id and g1.`type` IN ("R")
join team_batting_counts tbc2 on tbc1.team_id = tbc2.team_id 
join game g2 on tbc2.game_id = g2.game_id and g2.`type` in ("R")
and g2.local_date < g1.local_date and 
g2.local_date >= DATE_ADD(g1.local_date,interval -200 day)
group by tbc1.team_id,tbc1.game_id,g1.local_date
order by g1.local_date,tbc1.team_id;
create unique index team_game on RollingAvgTable(team_id,game_id);

#Making a final output table:- 
drop table OutputTable ;
create table OutputTable as
select g.home_team_id
, g.away_team_id
, g.game_id
, (r2dh.Hit/r2dh.atBat) /(r2da.Hit/r2da.atBat) -1.0 as BA_Ratio
, (r2dh.Hit/r2dh.atBat) - (r2da.Hit/r2da.atBat) as BA_diff
, (r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat)-(r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)/(r2da.atBat) as Slug_diff
, ((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/((r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)/(r2da.atBat)) -1.0 as Slug_ratio
, (1/(1+(POWER((r2dh.RunsAllowed/NULLIF(r2dh.RunsScored,0)),2))))-(1/(1+(POWER((r2da.RunsAllowed/NULLIF(r2da.RunsScored,0)),2)))) as pythag_winDiff
, (1/(1+(POWER((r2dh.RunsAllowed/NULLIF(r2dh.RunsScored,0)),2))))/nullif((1/(1+(POWER((r2da.RunsAllowed/NULLIF(r2da.RunsScored,0)),2)))),0)-1.0 as pythag_winRatio
, (r2dh.atBat/NULLIF(r2dh.Home_Run,0))-(r2da.atBat/NULLIF(r2da.Home_Run,0)) as atBatsPerHomeRunDiff
, (r2dh.atBat/NULLIF(r2dh.Home_Run,0))/nullif((r2da.atBat/NULLIF(r2da.Home_Run,0)),0)-1.0 as atBatsPerHomeRunRatio
, ((r2dh.Hit-r2dh.Home_Run)/NULLIF((r2dh.atBat-r2dh.StrikeOut-r2dh.Home_Run+r2dh.Sac_Fly),0))-((r2da.Hit-r2da.Home_Run)/NULLIF((r2da.atBat-r2da.StrikeOut-r2da.Home_Run+r2da.Sac_Fly),0)) as BattingAvgOnBallsInPlay_Diff
, ((r2dh.Hit-r2dh.Home_Run)/NULLIF((r2dh.atBat-r2dh.StrikeOut-r2dh.Home_Run+r2dh.Sac_Fly),0))/nullif(((r2da.Hit-r2da.Home_Run)/NULLIF((r2da.atBat-r2da.StrikeOut-r2da.Home_Run+r2da.Sac_Fly),0)),0)-1.0 as BattingAvgOnBallsInPlay_Ratio
, (r2dh.walk/NULLIF((r2dh.Strikeout+r2dh.`Strikeout_-_DP`+r2dh.`Strikeout_-_TP`),0))-(r2da.Walk/NULLIF((r2da.Strikeout+r2da.`Strikeout_-_DP`+r2da.`Strikeout_-_TP`),0)) as WalkStrkRatio_Diff
, (r2dh.walk/NULLIF((r2dh.Strikeout+r2dh.`Strikeout_-_DP`+r2dh.`Strikeout_-_TP`),0))/nullif((r2da.Walk/NULLIF((r2da.Strikeout+r2da.`Strikeout_-_DP`+r2da.`Strikeout_-_TP`),0)),0)-1.0 as WalkStrkRatio_Ratio
, (r2dh.`double`+r2dh.triple+r2dh.Home_Run)-(r2da.`double`+r2da.triple+r2da.Home_Run)as ExtraBaseHit_Diff
, (r2dh.`double`+r2dh.triple+r2dh.Home_Run)/nullif((r2da.`double`+r2da.triple+r2da.Home_Run),0)-1.0 as ExtraBaseHit_Ratio
, (r2dh.Hit+r2dh.Walk+r2dh.Hit_By_Pitch)-(r2da.Hit+r2da.Walk+r2da.Hit_By_Pitch)as TimesOnBase_Diff
, (r2da.Hit+r2da.Walk+r2da.Hit_By_Pitch)/nullif((r2dh.Hit+r2dh.Walk+r2dh.Hit_By_Pitch),0)-1.0 as TimesOnBase_Ratio
, (r2dh.Single+2*r2dh.`double`+3*r2dh.Triple+4*r2dh.Home_run)-(r2da.Single+2*r2da.`double`+3*r2da.Triple+4*r2da.Home_run) as Total_bases_Diff
, (r2dh.Single+2*r2dh.`double`+3*r2dh.Triple+4*r2dh.Home_run)/nullif((r2da.Single+2*r2da.`double`+3*r2da.Triple+4*r2da.Home_run),0)-1.0 as Total_bases_Ratio
, ((r2dh.Single+2*r2dh.`double`+3*r2dh.Triple+4*r2dh.Home_run+r2dh.Hit_By_Pitch+r2dh.stolenBase2B+r2dh.stolenBase3B+r2dh.stolenBaseHome+r2dh.Walk)/(r2dh.atBat-r2dh.Hit+r2dh.caughtStealing2B+r2dh.caughtStealing3B+r2dh.caughtStealingHome+r2dh.Grounded_Into_DP))-((r2da.Single+2*r2da.`double`+3*r2da.Triple+4*r2da.Home_run+r2da.Hit_By_Pitch+r2da.stolenBase2B+r2da.stolenBase3B+r2da.stolenBaseHome+r2da.Walk)/(r2da.atBat-r2da.Hit+r2da.caughtStealing2B+r2da.caughtStealing3B+r2da.caughtStealingHome+r2da.Grounded_Into_DP))as TotalAvg_Diff
, ((r2dh.Single+2*r2dh.`double`+3*r2dh.Triple+4*r2dh.Home_run+r2dh.Hit_By_Pitch+r2dh.stolenBase2B+r2dh.stolenBase3B+r2dh.stolenBaseHome+r2dh.Walk)/(r2dh.atBat-r2dh.Hit+r2dh.caughtStealing2B+r2dh.caughtStealing3B+r2dh.caughtStealingHome+r2dh.Grounded_Into_DP))/nullif(((r2da.Single+2*r2da.`double`+3*r2da.Triple+4*r2da.Home_run+r2da.Hit_By_Pitch+r2da.stolenBase2B+r2da.stolenBase3B+r2da.stolenBaseHome+r2da.Walk)/(r2da.atBat-r2da.Hit+r2da.caughtStealing2B+r2da.caughtStealing3B+r2da.caughtStealingHome+r2da.Grounded_Into_DP)),0)-1.0as TotalAvg_Ratio
, (r2dh.plateAppearance/(r2dh.Strikeout+r2dh.`Strikeout_-_DP`+r2dh.`Strikeout_-_TP`))-(r2da.plateAppearance/(r2da.Strikeout+r2da.`Strikeout_-_DP`+r2da.`Strikeout_-_TP`))as plateapprperstrikeout_Diff
, (r2dh.plateAppearance/(r2dh.Strikeout+r2dh.`Strikeout_-_DP`+r2dh.`Strikeout_-_TP`))/nullif((r2da.plateAppearance/(r2da.Strikeout+r2da.`Strikeout_-_DP`+r2da.`Strikeout_-_TP`)),0)-1.0 as plateapprperstrikeout_Ratio
, ((r2dh.Hit+r2dh.Walk+r2dh.Hit_By_Pitch)/(r2dh.atBat+r2dh.Walk+r2dh.Hit_By_Pitch+r2dh.Sac_Fly+r2dh.Sac_Fly_DP))-((r2da.Hit+r2da.Walk+r2da.Hit_By_Pitch)/(r2da.atBat+r2da.Walk+r2da.Hit_By_Pitch+r2da.Sac_Fly+r2da.Sac_Fly_DP))as OnBasePercentage_Diff
, ((r2dh.Hit+r2dh.Walk+r2dh.Hit_By_Pitch)/(r2dh.atBat+r2dh.Walk+r2dh.Hit_By_Pitch+r2dh.Sac_Fly+r2dh.Sac_Fly_DP))/nullif(((r2da.Hit+r2da.Walk+r2da.Hit_By_Pitch)/(r2da.atBat+r2da.Walk+r2da.Hit_By_Pitch+r2da.Sac_Fly+r2da.Sac_Fly_DP)),0)-1.0 as OnBasePercentage_Ratio
, (((r2dh.atBat*(r2dh.Hit+r2dh.Walk+r2dh.Hit_By_Pitch))+((r2dh.Single+2*r2dh.`Double`+3*r2dh.Triple+4*r2dh.Home_Run)*(r2dh.atBat+r2dh.Walk+r2dh.Sac_Fly+r2dh.Sac_Fly_DP+r2dh.Hit_By_Pitch)))/(r2dh.atBat*(r2dh.atBat+r2dh.Walk+r2dh.Sac_Fly+r2dh.Sac_Fly_DP+r2dh.Hit_By_Pitch)))-(((r2da.atBat*(r2da.Hit+r2da.Walk+r2da.Hit_By_Pitch))+((r2da.Single+2*r2da.`Double`+3*r2da.Triple+4*r2da.Home_Run)*(r2da.atBat+r2da.Walk+r2da.Sac_Fly+r2da.Sac_Fly_DP+r2da.Hit_By_Pitch)))/(r2da.atBat*(r2da.atBat+r2da.Walk+r2da.Sac_Fly+r2da.Sac_Fly_DP+r2da.Hit_By_Pitch))) as onbaseplusslug_Diff
, (((r2dh.atBat*(r2dh.Hit+r2dh.Walk+r2dh.Hit_By_Pitch))+((r2dh.Single+2*r2dh.`Double`+3*r2dh.Triple+4*r2dh.Home_Run)*(r2dh.atBat+r2dh.Walk+r2dh.Sac_Fly+r2dh.Sac_Fly_DP+r2dh.Hit_By_Pitch)))/(r2dh.atBat*(r2dh.atBat+r2dh.Walk+r2dh.Sac_Fly+r2dh.Sac_Fly_DP+r2dh.Hit_By_Pitch)))/nullif((((r2da.atBat*(r2da.Hit+r2da.Walk+r2da.Hit_By_Pitch))+((r2da.Single+2*r2da.`Double`+3*r2da.Triple+4*r2da.Home_Run)*(r2da.atBat+r2da.Walk+r2da.Sac_Fly+r2da.Sac_Fly_DP+r2da.Hit_By_Pitch)))/(r2da.atBat*(r2da.atBat+r2da.Walk+r2da.Sac_Fly+r2da.Sac_Fly_DP+r2da.Hit_By_Pitch))),0)-1.0 as onbaseplusslug_Ratio
, (((1.8*((r2dh.Hit+r2dh.Walk+r2dh.Hit_By_Pitch)/(r2dh.atBat+r2dh.Walk+r2dh.Hit_By_Pitch+r2dh.Sac_Fly+r2dh.Sac_Fly_DP)))+((r2dh.Single+2*r2dh.`Double`+3*r2dh.Triple+4*r2dh.Home_Run)/r2dh.atBat))/4)-(((1.8*((r2da.Hit+r2da.Walk+r2da.Hit_By_Pitch)/(r2da.atBat+r2da.Walk+r2da.Hit_By_Pitch+r2da.Sac_Fly+r2da.Sac_Fly_DP)))+((r2da.Single+2*r2da.`Double`+3*r2da.Triple+4*r2da.Home_Run)/r2da.atBat))/4) as GPA_Diff
, (((1.8*((r2dh.Hit+r2dh.Walk+r2dh.Hit_By_Pitch)/(r2dh.atBat+r2dh.Walk+r2dh.Hit_By_Pitch+r2dh.Sac_Fly+r2dh.Sac_Fly_DP)))+((r2dh.Single+2*r2dh.`Double`+3*r2dh.Triple+4*r2dh.Home_Run)/r2dh.atBat))/4)/nullif((((1.8*((r2da.Hit+r2da.Walk+r2da.Hit_By_Pitch)/(r2da.atBat+r2da.Walk+r2da.Hit_By_Pitch+r2da.Sac_Fly+r2da.Sac_Fly_DP)))+((r2da.Single+2*r2da.`Double`+3*r2da.Triple+4*r2da.Home_Run)/r2da.atBat))/4),0)-1.0 as GPA_Ratio
, ((r2dh.stolenBase2B+r2dh.stolenBase3B+r2dh.stolenBaseHome)/NULLIF((r2dh.stolenBase2B+r2dh.stolenBase3B+r2dh.stolenBaseHome+r2dh.caughtStealing2B+r2dh.caughtStealing3B+r2dh.caughtStealingHome),0))-((r2da.stolenBase2B+r2da.stolenBase3B+r2da.stolenBaseHome)/NULLIF((r2da.stolenBase2B+r2da.stolenBase3B+r2da.stolenBaseHome+r2da.caughtStealing2B+r2da.caughtStealing3B+r2da.caughtStealingHome),0)) as stolenbasepercentage_Diff
, ((r2dh.stolenBase2B+r2dh.stolenBase3B+r2dh.stolenBaseHome)/NULLIF((r2dh.stolenBase2B+r2dh.stolenBase3B+r2dh.stolenBaseHome+r2dh.caughtStealing2B+r2dh.caughtStealing3B+r2dh.caughtStealingHome),0))/nullif(((r2da.stolenBase2B+r2da.stolenBase3B+r2da.stolenBaseHome)/NULLIF((r2da.stolenBase2B+r2da.stolenBase3B+r2da.stolenBaseHome+r2da.caughtStealing2B+r2da.caughtStealing3B+r2da.caughtStealingHome),0)),0)-1.0 as stolenbasepercentage_Ratio
, ts1.streak as HT_Streak
, ts2.streak as AT_Streak
, case when b.away_runs < b.home_runs then 1
	   when b.away_runs > b.home_runs then 0 
	   else 0 end as Home_team_wins
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join team_streak ts1 on ts1.game_id = g.game_id and ts1.team_id = g.home_team_id 
join team_streak ts2 on ts2.game_id = g.game_id and ts2.team_id = g.away_team_id 
join boxscore b on b.game_id = g.game_id ;


select * from OutputTable;
Into Outfile '/Users/ashok/Desktop/Output_table.csv';