import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import plotly.express as px 

#print(sns.get_dataset_names())
ds = sns.load_dataset('mpg')
#healthexp 274 x 4
#taxis 6433 x 14
#planets 1035 x 6
#mpg(miles per gallon) 398 x 9

def check_for_null(ds):
    '''
    Checks if there are any null values in the dataset provided.
    '''
    
    #checks for missing values
    print("-Null values:\n",ds.isnull().sum())

    print(ds.info())
    return ds 


def generate_stats_breifly(ds):
    '''
    generate comprehensive description statistics
    '''

    print('\n\n-Quick overview of the dataset \n\n',ds)
    print('\n\n-Dimensions of the dataset\n\n', ds.shape)
    print('\n\n-More comprehensive overview\n\n',ds.describe(include='all'))
    print("\n\n- 2. Comprehensive Statistical Summary:")
    stats_summary = {
        'Mean(Average)': ds.mean(numeric_only=True),
        'Median(middle most value)': ds.median(numeric_only=True),
        'Mode(most frequently seen value)': ds.mode().iloc[0],
        'Standard Deviation(how far the values are from the mean)': ds.std(numeric_only=True),
        'Variance(how different the values are from each other)': ds.var(numeric_only=True),
        'Skewness(measures the asymmetry or lack of symmetry in a data distribution)': ds.skew(numeric_only=True),
    }
    
    for stat_name, stat_values in stats_summary.items():
        print(f"\n-{stat_name}:\n")
        print(stat_values)


def grouped_stats(ds):
    '''
    groups subsets which makes it easier for comparing data
    '''

    print(ds.groupby('cylinders')['displacement'].mean())
    print('\n\n',ds.groupby('mpg')['weight'].agg([ 'mean', 'max' , 'min']))
    print('\n\naverage mpg per model year\n',ds.groupby(['model_year','origin'], as_index=False)['mpg'].agg(['min','mean','max']))
    ds['horsepower_class'] = pd.cut(ds['horsepower'], bins=3, labels=('low_power','average_power','high_power'))
    print('classification of horsepower and its impact on mpg', ds.groupby('mpg')[['horsepower','horsepower_class']].min())

def create_visuals(ds):
    plt.subplot(2,1,1)
    plt.hist(ds['model_year'], bins = 10 , color= 'blue',)
    plt.ylabel('-Frequency (count) of cars-')
    plt.xlabel('-Differenet model year-')
    plt.title('-Different cars and their model_year-')

    plt.subplot(2,1,2)
    plt.bar(ds['origin'], ds['mpg'], color='blue')
    plt.xlabel('-origin-')
    plt.ylabel('-MPG (Miles Per Gallon)-')
    
    #plt.subplot(2,2,3)
    #plt.plot(ds[''], ds['model_year'],)
    # we will not use this here because the values cant be visualized in a line graph
    plt.tight_layout
    plt.show()


#check_for_null(ds)
#generate_stats_breifly(ds)
#grouped_stats(ds)
create_visuals(ds)


'''
       mpg  cylinders  displacement  horsepower  weight  acceleration  model_year  origin                       name
0    18.0          8         307.0       130.0    3504          12.0          70     usa  chevrolet chevelle malibu
1    15.0          8         350.0       165.0    3693          11.5          70     usa          buick skylark 320
2    18.0          8         318.0       150.0    3436          11.0          70     usa         plymouth satellite
3    16.0          8         304.0       150.0    3433          12.0          70     usa              amc rebel sst
4    17.0          8         302.0       140.0    3449          10.5          70     usa                ford torino
..    ...        ...           ...         ...     ...           ...         ...     ...                        ...
393  27.0          4         140.0        86.0    2790          15.6          82     usa            ford mustang gl
394  44.0          4          97.0        52.0    2130          24.6          82  europe                  vw pickup
395  32.0          4         135.0        84.0    2295          11.6          82     usa              dodge rampage
396  28.0          4         120.0        79.0    2625          18.6          82     usa                ford ranger
397  31.0          4         119.0        82.0    2720          19.4          82     usa                 chevy s-10
'''