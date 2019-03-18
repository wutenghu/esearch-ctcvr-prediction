set CURRENT_DATE="2019-02-18";
set TRACE_DATE_NUM=6;
set EXPO_THRESH=10;
set KEYWORD_FREQ_THRESH=100;
set KEYWORD_LENGTH_THRESH=50;

--创建搜索下的曝光-->点击-->转化日志流水表
CREATE TABLE if not EXISTS es_common.ctcvr_gearbest_pc_search_log
(user string, keyword string, goodssn string, exposed int, clicked int, purchased int);

--导入数据：利用用户ID追溯用户行为
insert overwrite table es_common.ctcvr_gearbest_pc_search_log
select expo.u user, expo.keyword, expo.goodssn, exposed, COALESCE(clicked, 0) clicked, COALESCE(purchased, 0) purchased from
(
    select keyword, goodssn, 1 as exposed, u from
    base_data.base_exposure_click_daily where
    ubcd=10002 and
    pm="mp" and
    page_type="b02" and
    plf="pc" and
    dc="1301" and
    keyword!="" and keyword!="NILL" and
    u!="NILL" and u!="" and u is not NULL and
    type = "exposure" and
    unix_timestamp(date_sub(${hiveconf:CURRENT_DATE}, ${hiveconf:TRACE_DATE_NUM}),'yyyy-MM-dd') <=unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd') and
    unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd')<=unix_timestamp(${hiveconf:CURRENT_DATE},'yyyy-MM-dd')
) expo
left join
(
    select keyword, goodssn, 1 as clicked, u from
    base_data.base_exposure_click_daily where
    ubcd=10002 and
    pm="mp" and
    page_type="b02" and
    plf="pc" and
    dc="1301" and
    keyword!="" and keyword!="NILL" and
    u!="NILL" and u!="" and u is not NULL and
    type = "click" and
    unix_timestamp(date_sub(${hiveconf:CURRENT_DATE}, ${hiveconf:TRACE_DATE_NUM}),'yyyy-MM-dd') <=unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd') and
    unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd')<=unix_timestamp(${hiveconf:CURRENT_DATE},'yyyy-MM-dd')
) click on expo.keyword=click.keyword and expo.goodssn=click.goodssn and expo.u=click.u
left join
(
    select goodssn, 1 as purchased, u from
    base_data.base_pay_daily where
    ubcd=10002 and
    plf="pc" and
    dc="1301" and
    u!="NILL" and u!="" and u is not NULL and
    unix_timestamp(date_sub(${hiveconf:CURRENT_DATE}, ${hiveconf:TRACE_DATE_NUM}),'yyyy-MM-dd') <=unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd') and
    unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd')<=unix_timestamp(${hiveconf:CURRENT_DATE},'yyyy-MM-dd')
) purchase on click.goodssn=purchase.goodssn and click.u=purchase.u;

--导入数据：利用CookieID追溯用户行为
insert into table es_common.ctcvr_gearbest_pc_search_log
select expo.od user, expo.keyword, expo.goodssn, exposed, COALESCE(clicked, 0) clicked, COALESCE(purchased, 0) purchased from
(
    select keyword, goodssn, 1 as exposed, od from
    base_data.base_exposure_click_daily where
    ubcd=10002 and
    pm="mp" and
    page_type="b02" and
    plf="pc" and
    dc="1301" and
    u="NILL" and
    keyword!="" and keyword!="NILL" and
    od!="NILL" and od!="" and od is not NULL and
    type = "exposure" and
    unix_timestamp(date_sub(${hiveconf:CURRENT_DATE}, ${hiveconf:TRACE_DATE_NUM}),'yyyy-MM-dd') <=unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd') and
    unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd')<=unix_timestamp(${hiveconf:CURRENT_DATE},'yyyy-MM-dd')
) expo
left join
(
    select keyword, goodssn, 1 as clicked, od from
    base_data.base_exposure_click_daily where
    ubcd=10002 and
    pm="mp" and
    page_type="b02" and
    plf="pc" and
    dc="1301" and
    u="NILL" and
    keyword!="" and keyword!="NILL" and
    od!="NILL" and od!="" and od is not NULL and
    type = "click" and
    unix_timestamp(date_sub(${hiveconf:CURRENT_DATE}, ${hiveconf:TRACE_DATE_NUM}),'yyyy-MM-dd') <=unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd') and
    unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd')<=unix_timestamp(${hiveconf:CURRENT_DATE},'yyyy-MM-dd')
) click on expo.keyword=click.keyword and expo.goodssn=click.goodssn and expo.od=click.od
left join
(
    select goodssn, 1 as purchased, od from
    base_data.base_pay_daily where
    ubcd=10002 and
    plf="pc" and
    dc="1301" and
    od!="NILL" and od!="" and od is not NULL and
    unix_timestamp(date_sub(${hiveconf:CURRENT_DATE}, ${hiveconf:TRACE_DATE_NUM}),'yyyy-MM-dd') <=unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd') and
    unix_timestamp(concat(year,"-",month,"-",day),'yyyy-MM-dd')<=unix_timestamp(${hiveconf:CURRENT_DATE},'yyyy-MM-dd')
) purchase on click.goodssn=purchase.goodssn and click.od=purchase.od;

--用户日志流水去重
insert overwrite table es_common.ctcvr_gearbest_pc_search_log
select distinct user, keyword, goodssn, exposed, clicked, purchased from
es_common.ctcvr_gearbest_pc_search_log;


--创建关键词商品维度样本数据
CREATE TABLE if not EXISTS es_common.ctcvr_training_sample
(keyword string, goodssn string, expose_total int, click_total int, purchase_total int, clicked float, purchased float);

--导入数据 过滤曝光过低的样本和异常样本
insert overwrite table es_common.ctcvr_training_sample
select keyword, goodssn, expose_total, click_total, purchase_total, clicked, purchased from (
select keyword, goodssn, expose_total, click_total, purchase_total, clicked, COALESCE(purchased, 0) purchased from (
select keyword, goodssn,
sum(exposed) as expose_total,
sum(clicked) as click_total,
sum(purchased) as purchase_total,
sum(clicked)/sum(exposed) as clicked,
sum(purchased)/sum(clicked) as purchased from es_common.ctcvr_gearbest_pc_search_log group by keyword, goodssn
) v where expose_total > ${hiveconf:EXPO_THRESH} and click_total >= purchase_total ) vv;

--过滤掉出现次数过少的搜索词的关联条目
insert overwrite table es_common.ctcvr_training_sample
select
orig.keyword keyword,
orig.goodssn goodssn,
orig.expose_total expose_total,
orig.click_total click_total,
orig.purchase_total purchase_total,
orig.clicked clicked,
orig.purchased purchased
from es_common.ctcvr_training_sample orig
inner join
(
  select keyword, kw_count from
  (
    select keyword, count(keyword) kw_count from es_common.ctcvr_gearbest_pc_search_log group by keyword) v
  where kw_count >= ${hiveconf:KEYWORD_FREQ_THRESH}
) vv on orig.keyword=vv.keyword;

insert overwrite directory '/esearch/esearch-stat/data/result_data/gearbest_search_train'
select keyword, goodssn, expose_total, click_total, purchase_total, clicked, purchased from es_common.ctcvr_training_sample
where length(keyword)<${hiveconf:KEYWORD_LENGTH_THRESH};

insert overwrite directory '/esearch/esearch-stat/data/result_data/gearbest_search_test'
select log.keyword, log.goodssn from
(select distinct keyword, goodssn from es_common.ctcvr_gearbest_pc_search_log where length(keyword)<${hiveconf:KEYWORD_LENGTH_THRESH}) log
inner join
(select distinct keyword from es_common.ctcvr_training_sample where length(keyword)<${hiveconf:KEYWORD_LENGTH_THRESH}) sample
on log.keyword=sample.keyword;