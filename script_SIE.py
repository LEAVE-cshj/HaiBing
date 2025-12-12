import argparse
import calendar
import csv
import datetime
import os
import runpy

import requests
import numpy as np
import pandas as pd
from netCDF4 import Dataset
#import script_train
import script_test


#获取命令行的参数
def get_args():
    parser = argparse.ArgumentParser(description="Data download configs")
    parser.add_argument(
        "--expt_name",
        metavar="-e",
        type=str,
        nargs="?",
        default="icemonthly")
    parser.add_argument(
        "--output_folder",
        metavar="-f",
        type=str,
        nargs="?",
        default="expt_settings/outputs")
    parser.add_argument(
        "--use_gpu",
        metavar="-g",
        type=str,
        nargs="?",
        choices=["yes", "no"],
        default="yes")
    parser.add_argument(
        "--re_train",
        metavar="-re",
        type=str,
        nargs="?",
        choices=["yes", "no"],
        default="no")

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == "." else args.output_folder

    return args.expt_name, root_folder, args.use_gpu == "yes", args.re_train == "yes"

def create_ncdata(count,path,df,name,varname,col):
    df[col+'_avg']=0.0#avg
    now = datetime.datetime.now()
    year = now.year
    month = now.month - 1
    data_ori = Dataset(path)
    data_sst = data_ori.variables[name][:]
    l = (year- 1982) * 12 + month
    num = len(data_sst) - l
    data_sst = data_sst[num-1:-1]

    for i in range(0, l):
        df.loc[i, col + '_avg'] = data_sst[i].mean()

def create_csvdata(count,path,df,col,flag):
    # 读取CSV文件
    data = pd.read_csv(path)

    # 去除列名前后的空格
    data.columns = data.columns.str.strip()
    # 仅保留那些'Year'列是数字的行
    data = data[data['Year'].str.isdigit()]
    # 创建日期列（确保年月日格式匹配您的数据）
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']].astype(int).astype(str).agg('-'.join, axis=1))
    data['Extent'] = pd.to_numeric(data['Extent'], errors='coerce')

    # 计算每月的平均extent
    monthly_extent = data.groupby(data['Date'].dt.to_period('M'))['Extent'].mean()
    now = datetime.datetime.now()
    year = now.year
    month = now.month - 1
    l = (year - 1982) * 12 + month
    num = len(monthly_extent) - l
    data_sf = monthly_extent[num-1:-1]
    df[col+ '_avg'] = data_sf.values
# 获取新的数据
def get_new_data():

    filename1 = 'https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v3.0.csv'
    filelist1 = '/N_seaice_extent_daily_v3.0.csv'
    file_base = './newData/' + os.path.basename(filelist1)
    print('Downloading', file_base)
    response = requests.get(filename1, allow_redirects=True, stream=True)
    with open(file_base, 'wb') as outfile:
        outfile.write(response.content)
    #https: // downloads.psl.noaa.gov // Datasets / ncep.reanalysis / Monthlies / surface_gauss / air.2m.mon.mean.nc
    dspath = 'https://downloads.psl.noaa.gov'
    filelist = ['/Datasets/ncep.reanalysis/Monthlies/surface_gauss/air.2m.mon.mean.nc',
                '/Datasets/ncep.reanalysis/Monthlies/surface_gauss/shum.2m.mon.mean.nc',
                '/Datasets/ncep.reanalysis/Monthlies/surface_gauss/dswrf.sfc.mon.mean.nc',
                '/Datasets/ncep.reanalysis/Monthlies/surface_gauss/dlwrf.sfc.mon.mean.nc',
                '/Datasets/ncep.reanalysis/Monthlies/surface_gauss/csdsf.sfc.mon.mean.nc',
                '/Datasets/ncep.reanalysis/Monthlies/surface_gauss/csdlf.sfc.mon.mean.nc',
                '/Datasets/ncep.reanalysis/Monthlies/surface_gauss/uswrf.sfc.mon.mean.nc',
                '/Datasets/ncep.reanalysis/Monthlies/surface_gauss/prate.sfc.mon.mean.nc',
                '/Datasets/ncep.reanalysis/Monthlies/surface_gauss/runof.sfc.mon.mean.nc',
                '/Datasets/COBE2/sst.mon.mean.nc']
    for file in filelist:
        filename = dspath + file
        file_base = './newData/' + os.path.basename(file)
        print('Downloading', file_base)
        response = requests.get(filename)
        with open(file_base, 'wb') as outfile:
            outfile.write(response.content)

    data = pd.read_csv('./expt_settings/outputs/data/icemonly/jra553.csv')
    df = data[:]  # 1982-01
    df = df.reset_index(drop=True)
    count = 1
    count = create_ncdata(count, './newData/sst.mon.mean.nc', df, 'sst', 'sst','sst')  # COBE SST 1850.01-2024.02
    count = create_ncdata(count, './newData/air.2m.mon.mean.nc', df, 'air', 'air', 'air')
    count = create_ncdata(count, './newData/dlwrf.sfc.mon.mean.nc', df, 'dlwrf', 'dlwrf', 'dlwrf')
    count = create_ncdata(count, './newData/dswrf.sfc.mon.mean.nc', df, 'dswrf', 'dswrf', 'dswrf')
    count = create_ncdata(count, './newData/prate.sfc.mon.mean.nc', df, 'prate', 'prate', 'prate')
    count = create_ncdata(count, './newData/runof.sfc.mon.mean.nc', df, 'runof', 'runof', 'runof')
    count = create_ncdata(count, './newData/shum.2m.mon.mean.nc', df, 'shum', 'shum', 'shum')
    count = create_ncdata(count, './newData/csdlf.sfc.mon.mean.nc', df, 'csdlf', 'csdlf', 'csdlf')
    count = create_ncdata(count, './newData/csdsf.sfc.mon.mean.nc', df, 'csdsf', 'csdsf', 'csdsf')
    count = create_ncdata(count, './newData/uswrf.sfc.mon.mean.nc', df, 'uswrf', 'uswrf', 'uswrf')
    count = create_csvdata(count, "./newData/N_seaice_extent_daily_v3.0.csv", df, 'extent', False)
    df = df.loc[:, df.columns.str.contains('_avg')]
    df = df.rename(columns={'extent_avg': 'extend'})
    df['id'] = 1
    df['cat_id'] = 1

    # 创建起始日期
    start_date = pd.to_datetime('1982/01/01')
    now = datetime.datetime.now()
    year = now.year
    month = now.month - 1
    end_date = pd.to_datetime(str(year) + '/' + str(month) + '/' + '01')
    df['date'] = pd.date_range(start=start_date, end=end_date, freq='MS')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['days_in_month'] = df['date'].apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    df['days_from_start'] = 1096
    df.loc[1:, 'days_from_start'] = df['days_in_month'].cumsum().shift(1) + 1096
    df = df.drop(columns=['days_in_month'])

    df.to_csv("./expt_settings/outputs/data/icemonly/sienew.csv")
    print('Done')


if __name__ == '__main__':

    get_new_data()
    runpy.run_path('script_train.py')
    '''
    arges = get_args()
    if (arges[3]=='yes') :
        print("Re_Train")
        get_new_data()
        runpy.run_path('script_train.py')

    else:
        print("No Re_Train")
        runpy.run_path('script_test.py')
    '''
#runpy.run_path('plot_results.py')



