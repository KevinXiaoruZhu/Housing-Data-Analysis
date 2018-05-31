from pyspark import SparkContext, SparkConf, StorageLevel
import csv
import json
import io

# Configure
data_path = 'Dataset/NewYork2015.csv'
conf = SparkConf().setMaster('local').setAppName('Housing Data App')
sc = SparkContext(conf=conf)

# Initialize RDD
rdd_lines = sc.textFile(data_path).persist(StorageLevel(True, True, False, False, 1))  # MEMORY_AND_DISK
head = rdd_lines.first()
rdd_lines = rdd_lines.filter(lambda ln: ln != head)\
                     .mapPartitions(lambda x: csv.reader(x))\
                     .filter(lambda ln: int(ln[10]) >= 10000)\
                     .filter(lambda ln: int(ln[8]) != 0 and int(ln[9]) != 0)  # remove the first line+resolve csv file+remove zero sale lines

# Attribute list
attr_list = ['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'ZIP CODE', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 'LAND SQUARE FEET,YEAR BUILT', 'SALE PRICE(US D)', 'SALE DATE']


def house_decades(build_year):
    build_year = int(build_year)
    build_time = 2015 - build_year
    if build_time <= 30:
        return '0~30 years'
    elif build_time > 30 and build_time <= 60 :
        return '30~60 years'
    else:
        return 'more than 60 years'


def divide(x, y):
    x, y = int(x), int(y)
    return round(x/y, 2)


def to_CSV_line(list_line):
    return ','.join(str(d) for d in list_line)


# Data analysis
def sales_num():
    return rdd_lines.count()


def neighbor_st():
    rdd_ngh = rdd_lines.map(lambda attr: (attr[0], 1))\
                       .reduceByKey(lambda x, y: x + y)\
                       .sortBy(keyfunc=lambda v: v[1], ascending=False)
    return rdd_ngh.collect()


def building_class_st():
    rdd_bd = rdd_lines.map(lambda attr: (attr[1], 1))\
                      .reduceByKey(lambda x, y: x + y)\
                      .sortBy(keyfunc=lambda v: v[1], ascending=False)
    return rdd_bd.collect()


def unit_num_st():
    rdd_un = rdd_lines.map(lambda attr: (attr[7], 1))\
                      .reduceByKey(lambda x, y: x + y)\
                      .sortBy(keyfunc=lambda v: v[1], ascending=False)
    return rdd_un.collect()


def sale_month_st():
    rdd_sd = rdd_lines.map(lambda attr: (attr[11].split('/')[1], 1))\
                      .reduceByKey(lambda x, y: x + y)\
                      .sortByKey(ascending=True, keyfunc=lambda v: int(v))
    return rdd_sd.collect()


def year_build_st():
    rdd_yb = rdd_lines.map(lambda attr: (house_decades(attr[9]), 1))\
                      .reduceByKey(lambda x, y: x + y)\
                      .sortBy(keyfunc=lambda v: v[1], ascending=False)
    return rdd_yb.collect()


def neighbor_avg_price():
    rdd_nap = rdd_lines.map(lambda attr: (attr[0], divide(attr[10], attr[8])))\
                       .mapValues(lambda vl: (vl, 1))\
                       .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
                       .mapValues(lambda vl: divide(vl[0], vl[1]))\
                       .sortBy(keyfunc=lambda v: v[1], ascending=False)
    return rdd_nap.collect()


def class_avg_price():
    rdd_cap = rdd_lines.map(lambda attr: (attr[1], divide(attr[10], attr[8])))\
                       .mapValues(lambda vl: (vl, 1))\
                       .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
                       .mapValues(lambda vl: divide(vl[0], vl[1]))\
                       .sortBy(keyfunc=lambda v: v[1], ascending=False)
    return rdd_cap.collect()


def year_avg_price():
    rdd_yap = rdd_lines.map(lambda attr: (house_decades(attr[9]), divide(attr[10], attr[8])))\
                       .mapValues(lambda vl: (vl, 1))\
                       .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
                       .mapValues(lambda vl: divide(vl[0], vl[1]))\
                       .sortBy(keyfunc=lambda v: v[1], ascending=False)
    return rdd_yap.collect()


# Write to CSV files
def save_as_csv(head_list, rdd_rst, save_path):
    union_head_rdd = sc.parallelize([head_list])\
                       .union(rdd_rst)
    union_head_rdd.coalesce(1).map(to_CSV_line).saveAsTextFile(str(save_path))
    print('Saving CSV file successfully!\n')


# list convert to json object
def convert_to_json(lst, file_name):
    lst = list(lst)
    file_name = str(file_name)
    jsn_dict = {}
    for k, v in lst:
        jsn_dict[k] = v
    with open('./HousingAnalysis/Results/' + file_name + '.json', 'w') as f:
        json.dump(jsn_dict, f)
    return 'file ' + file_name + ' successfully saved!'


if __name__ == "__main__":
    # save_as_csv(attr_list, rdd_lines, './Dataset/train.csv')
    with open('./HousingAnalysis/Results/total_sale_num.txt', 'w') as f:
        f.write(str(sales_num()))

    print(convert_to_json(neighbor_st(), 'neighbor_st'))
    print(convert_to_json(building_class_st(), 'building_class_st'))
    print(convert_to_json(unit_num_st(), 'unit_num_st'))
    print(convert_to_json(sale_month_st(), 'sale_month_st'))
    print(convert_to_json(year_build_st(), 'year_build_st'))
    print(convert_to_json(neighbor_avg_price(), 'neighbor_avg_price'))
    print(convert_to_json(class_avg_price(), 'class_avg_price'))
    print(convert_to_json(year_avg_price(), 'year_avg_price'))
