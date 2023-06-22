import datetime
import pandas as pd
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
import numpy as np
from heapq import nlargest
import matplotlib.dates as mdates
import geopandas as gpd

def readDataAndfix(df):
    yearWeek = df.loc[:, 'year_week']

    def fixDate(row):
        date = datetime.datetime.strptime(row + '-1', '%Y-W%W-%w')
        return date

    for row in yearWeek:
        fixDate(row)

    df['formatted_date'] = yearWeek.apply(fixDate)
    df.to_csv('newData1.csv')
    df = pd.read_csv('newData1.csv')
    return df

#אחוז מקרי המוות מכלל מספר החולים לפי חודשים בשנת 2021 בהונגריה
def Q1(df):
    #filtering rows so that we will have only rows of hungary from 2021
    df = df[(df.countriesAndTerritories == "Hungary") & (df.year == 2021)]

    #preparing empty dictionaries-
    #cases dict- save all the cases in a *list* per month
    #deaths dict- save all the deaths in a *list* per month
    cases_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}
    deaths_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}

    #fill the dictionaries with what we wanted
    for index, row in df.iterrows():
        cases_dict[row['month']].append(row['cases'])
        deaths_dict[row['month']].append(row['deaths'])

    #loop over the two dict together and sum each list for every key
    for (cases_key, cases_value), (deaths_key, deaths_value) in zip(cases_dict.items(), deaths_dict.items()):
        cases_dict[cases_key] = sum(cases_value)
        deaths_dict[deaths_key] = sum(deaths_value)

    #make another dict that represent the percentage of every month
    percentage_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}
    for key in percentage_dict:
        percentage_dict[key] = (deaths_dict[key] / cases_dict[key]) * 100

    #generating plot
    names = list(percentage_dict.keys())
    month_names = [calendar.month_name[i] for i in names]  # Get month names using calendar.month_name
    values = list(percentage_dict.values())

    plt.figure(figsize=(10, 6))  # Adjust the figure size as per your needs
    bars = plt.bar(range(len(percentage_dict)), values, tick_label=month_names,color='#4CC9D8')  # Use month_names for tick labels

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(round(value, 2)) + '%', ha='center', va='bottom')

    #style
    plt.xlabel('Months')
    plt.ylabel('% Death Cases')
    plt.title('% Deaths in 2021 in the Country of Hungary')
    plt.xticks(rotation=90)  # Rotate the x-axis labels by 90 degrees
    plt.tight_layout()  # Automatically adjust the subplot parameters to fit the figure area
    plt.savefig("Q1.png")
    plt.show()

#מספר המאומתים החדשים בחודש מאי בספרד ומקדם ההדבקה נכון לתאריך 30/05/2022
def Q2(df):
   #filter the data for Spain and May 2022
   filtered_data = df[(df['country'] == 'Spain') & (df['formatted_date'].str.startswith('2022-05'))]

   #group the data by date and calculate the sum of new_cases
   daily_cases = filtered_data.groupby('formatted_date')['new_cases'].sum()


   #calculation of the corona infection coefficient
   new_cases = filtered_data['new_cases'].values.tolist()
   r_value = 0
   for i in range(1, len(new_cases)):
      if i == len(new_cases) - 1:
           r_value = new_cases[i] / new_cases[i - 1]

   # Create the bar plot
   plt.figure(figsize=(10, 6))
   daily_cases.plot(kind='bar',color='#F970CB')
   plt.xlabel('Date')
   plt.ylabel('New Cases')
   plt.title('New COVID-19 Cases in Spain - May 2022')
   plt.xticks(rotation=0)
   plt.gca().set_xticklabels(daily_cases.index)
   text_x = len(daily_cases) - 0.8
   text_y = daily_cases.max() * 0.95
   for i, value in enumerate(daily_cases):
       plt.text(i, value, str(int(value)), ha='center', va='bottom', fontsize=9)
   plt.text(text_x, text_y, f'R = {round(r_value, 5):,}', fontsize=12, ha='right')
   plt.savefig('Q2.png')
   plt.show()



#שיעור בדיקות שבוצעו על ציר הזמן בבולגריה
def Q3(df):
    #filter the data for the country Bulgaria and where positivity_rate is not NA
    bulgaria_data = df[(df['country'] == 'Bulgaria') & (df['positivity_rate'] != 'NA')]

    #extract the relevant columns (year_week and positivity_rate) from the filtered data
    dates = bulgaria_data['formatted_date']
    testing_rates = pd.to_numeric(bulgaria_data['testing_rate'], errors='coerce')

    #drop missing values (NaN) from both arrays simultaneously
    valid_data = pd.concat([dates, testing_rates], axis=1).dropna()

    #extract the filtered and aligned data
    dates = valid_data['formatted_date']
    testing_rates = valid_data['testing_rate']

    #plot the curve using Matplotlib
    plt.plot(dates, testing_rates)
    plt.xlabel('Date')
    plt.ylabel('testing_rate')
    plt.title('Testing Rate Curve - Bulgaria')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #set the X-axis locator to show major ticks at the start of each month
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.savefig("Q3.png")
    plt.show()

# מספר מקרי המוות בשנת 2021 בדנמרק לפי חודשים
def Q4(df):
    #filter the data for Denmark in 2021
    denmark_2021 = df[(df['countriesAndTerritories'] == 'Denmark') & (df['year'] == 2021)]

    #group the data by month and calculate the sum of deaths
    deaths_by_month = denmark_2021.groupby('month')['deaths'].sum()

    #create a bar graph
    plt.bar(deaths_by_month.index, deaths_by_month,color='#C70039')
    plt.xlabel('Month')
    plt.ylabel('Number of Deaths')
    plt.title('Deaths in Denmark (2021) by Month')
    #set the month names as x-axis labels
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    plt.xticks(range(len(deaths_by_month.index)), month_names, rotation=45)
    plt.tight_layout()
    plt.savefig("Q4.png")
    plt.show()

#5 המדינות בהן מספר המאושפזים הגבוה ביותר לפי אחוזים
def Q5(df):
    #group the data by country and find the highest value for each country
    highest_values = df.groupby('country')['value'].max()

    #sort the highest values in descending order
    sorted_values = highest_values.sort_values(ascending=False)

    #extract the top 5 values and their corresponding labels
    top_5 = sorted_values.head(5)
    labels = top_5.index.tolist()
    values = top_5.values.tolist()

    #create a pie chart using Matplotlib
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title('Top 5 Countries by Rate of Hospitalized')
    plt.savefig("Q5.png")
    plt.show()

#מספר המאושפזים בבתי החולים באוסטריה לאורך ציר הזמן על פי Daily hospital occupancy
def Q6(df):
    #filter the data for Austria and Daily hospital occupancy
    filtered_data = df[(df['country'] == 'Austria') & (df['indicator'] == 'Daily hospital occupancy')]

    #convert the date column to datetime format
    filtered_data.loc[:, 'date'] = pd.to_datetime(filtered_data['date'])

    #sort the data based on the date column
    filtered_data = filtered_data.sort_values('date')

    #plotting the curve
    plt.plot(filtered_data['date'], filtered_data['value'], color='purple')
    plt.title('Daily Hospital Occupancy in Austria')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Q6.png")
    plt.show()

#מספר מקרי המוות בשוודיה לפי רבעונים בשנת 2022
def Q7(df):
    #filter the data for Sweden in 2022
    Sweden_2022 = df[(df['countriesAndTerritories'] == 'Sweden') & (df['year'] == 2022)].copy()

    #convert the dateRep column to datetime with dayfirst=True
    Sweden_2022['dateRep'] = pd.to_datetime(Sweden_2022['dateRep'], dayfirst=True)

    #extract the quarters from the dateRep column
    Sweden_2022['quarter'] = Sweden_2022['dateRep'].dt.quarter

    #group the data by quarter and calculate the sum of deaths
    deaths_by_quarter = Sweden_2022.groupby('quarter')['deaths'].sum()

    #create a horizontal bar graph
    plt.barh(deaths_by_quarter.index, deaths_by_quarter, color='#F9F970')
    plt.xlabel('Number of Deaths')
    plt.ylabel('Quarter')
    plt.title('Deaths in Sweden (2022) by Quarter')

    #set the quarter labels as y-axis ticks
    quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    plt.yticks(deaths_by_quarter.index, quarter_labels)
    plt.savefig("Q7.png")
    plt.show()

#שיעור בדיקות חיוביות על ציר הזמן בקרואטיה
def Q8(df):
    #filter the data for the country Croatia and where positivity_rate is not NA
    croatia_data = df[(df['country'] == 'Croatia') & (df['positivity_rate'] != 'NA')]

    #extract the relevant columns (year_week and positivity_rate) from the filtered data
    dates = croatia_data['formatted_date']
    positivity_rates = pd.to_numeric(croatia_data['positivity_rate'], errors='coerce')

    #drop missing values (NaN) from both arrays simultaneously
    valid_data = pd.concat([dates, positivity_rates], axis=1).dropna()

    #extract the filtered and aligned data
    dates = valid_data['formatted_date']
    positivity_rates = valid_data['positivity_rate']

    #plot the curve using Matplotlib
    plt.plot(dates, positivity_rates, color='orange')
    plt.xlabel('Date')
    plt.ylabel('Positivity Rate')
    plt.title('Positivity Rate Curve - Croatia')
    plt.xticks(rotation=45)

    #set the X-axis locator to show major ticks at the start of each month
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout()
    plt.savefig("Q8.png")
    plt.show()


#כמות המאושפזים הרגילים ובטיפול נמרץ בתפוסה יומית באוסטריה לאורך ציר הזמן
def Q9(df):
    #filter the data for Austria and specific indicators
    filtered_data_hospital = df.loc[(df['country'] == 'Austria') & (df['indicator'] == 'Daily hospital occupancy')]
    filtered_data_icu = df.loc[(df['country'] == 'Austria') & (df['indicator'] == 'Daily ICU occupancy')]

    #convert the date column to datetime format
    filtered_data_hospital.loc[:, 'date'] = pd.to_datetime(filtered_data_hospital['date'])
    filtered_data_icu.loc[:, 'date'] = pd.to_datetime(filtered_data_icu['date'])

    #sort the data based on the date column
    filtered_data_hospital = filtered_data_hospital.sort_values('date')
    filtered_data_icu = filtered_data_icu.sort_values('date')

    #plotting the curves
    plt.plot(filtered_data_hospital['date'], filtered_data_hospital['value'], label='Daily Hospital Occupancy', color='green')
    plt.plot(filtered_data_icu['date'], filtered_data_icu['value'], label='Daily ICU Occupancy', color='red')
    plt.title('Hospitalization in Austria')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom spacing for x-axis labels
    plt.savefig("Q9.png")
    plt.show()

#מספר המאושפזים בשנת 2022 באוסטריה לפי חודשים
def Q10(df):
    #filter the data for Austria in 2022
    austria_2022 = df[(df['country'] == 'Austria') & (df['date'].str.startswith('2022'))]

    #group the data by month and calculate the sum of hospitalized patients
    hospitalized_by_month = austria_2022.groupby(austria_2022['date'].str[0:7])['value'].sum()

    #create a bar graph
    plt.bar(hospitalized_by_month.index, hospitalized_by_month, color='#B44CD8')
    plt.xlabel('Month')
    plt.ylabel('Number of Hospitalized Patients')
    plt.title('Hospitalized Patients in Austria (2022) by Month')

    #set the month names as x-axis labels
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    plt.xticks(range(len(hospitalized_by_month.index)), month_names, rotation=45)
    plt.tight_layout()
    plt.savefig("Q10.png")
    plt.show()

#מספר מקרי המוות בכל מדינה
def Q11(df):
    #handling NaN values
    df['deaths'] = df['deaths'].fillna(0)

    #handling infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['deaths'], inplace=True)

    unique_list = df['countriesAndTerritories'].unique()

    death_per_country_dict = {}
    for country in unique_list:
        death_per_country_dict[country] = df.loc[df['countriesAndTerritories'] == country, 'deaths'].sum()

    #generating plot
    names = list(death_per_country_dict.keys())
    values = list(death_per_country_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(death_per_country_dict)), values, tick_label=names, color='#F01850')

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(value)), ha='center', va='bottom', fontsize=7)

    #style
    plt.xlabel('Countries')
    plt.ylabel('Total Deaths')
    plt.title('Total Deaths per Country')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("Q11.png")
    plt.show()


#השוואה בין יחס המאושפזים בתפוסה יומית באירלנד לבין יחס המאושפזים בתפוסה יומית בהולנד בשנת 2021 לאחר נרמול גודל האוכלוסייה לאורך ציר הזמן
def Q12(df1, df3):
    # filter the data for Ireland and the Netherlands and the year 2021
    ireland_data = df3.loc[(df3['country'] == 'Ireland') & (df3['indicator'] == 'Daily hospital occupancy') & (
        df3['date'].str.startswith('2021'))]
    netherlands_data = df3.loc[(df3['country'] == 'Netherlands') & (df3['indicator'] == 'Daily hospital occupancy') & (
            df3['date'].str.startswith('2021'))]

    # convert the 'date' column to datetime format
    ireland_data.loc[:, 'date'] = pd.to_datetime(ireland_data['date'])
    netherlands_data.loc[:, 'date'] = pd.to_datetime(netherlands_data['date'])

    # extract the date and value columns
    ireland_dates = ireland_data['date']
    ireland_values = ireland_data['value']
    netherlands_dates = netherlands_data['date']
    netherlands_values = netherlands_data['value']

    # normalization
    ireland_population = df1.loc[df1['country'] == 'Ireland', 'population'].values[0]
    netherlands_population = df1.loc[df1['country'] == 'Netherlands', 'population'].values[0]
    normaliz_ireland = ireland_values / ireland_population
    normaliz_netherlands = netherlands_values / netherlands_population

    # plotting the curves
    plt.plot(ireland_dates, normaliz_ireland, label='Ireland', color='red')
    plt.plot(netherlands_dates, normaliz_netherlands, label='Netherlands')

    # add labels and title to the graph
    plt.xlabel('Date')
    plt.ylabel('% Daily Hospital Occupancy')
    plt.title('Comparison of Daily Hospital Occupancy In % - Ireland vs. Netherlands (2021)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Q12.png")
    plt.show()

#שיעור הבדיקות החיוביות לפי כמות האוכלוסייה בכל המדינות
def Q13(df):
    unique_list = df['country'].unique()

    #making empty dictionary of positivity rate by country
    avg_positivity_rate_dict = {}
    for country in unique_list:
        avg_positivity_rate_dict[country] = 0

    df = df[df['positivity_rate'].notna()]

    for index, row in df.iterrows():
        positivity_rate = (row['positivity_rate'])
        country = row['country']
        avg_positivity_rate_dict[country] = \
            (sum((avg_positivity_rate_dict[country], positivity_rate)) / 2)

    population_dict = {}
    for index, row in df.iterrows():
        if row['country'] not in population_dict:
            key = row['country']
            value = row['population']
            population_dict[key] = value / 1000

    y_values = list(avg_positivity_rate_dict.values())
    x_values = list(population_dict.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(x_values, y_values, color="#FF5733")
    plt.xlabel('population count')
    plt.ylabel('positivity rate')
    plt.title('countries positivity rate by population')

    for i, country in enumerate(avg_positivity_rate_dict.keys()):
        plt.annotate(country, (x_values[i], y_values[i]), fontsize=8)
    plt.savefig("Q13.png")
    plt.show()

#אחוז הנדבקים החדשים מתוך הבדיקות שהתבצעו בכל מדינה
def Q14(df):
    #drop rows with missing values in 'new_cases' or 'tests_done' columns
    df = df.dropna(subset=['new_cases', 'tests_done'])

    #group the data by country and calculate the sum of new_cases and tests_done for each country
    grouped_df = df.groupby('country').sum()

    #calculate the % of new cases out of tests done
    grouped_df['percentage'] = grouped_df['new_cases'] / grouped_df['tests_done'] * 100

    #sort the DataFrame by the calculated % in descending order
    grouped_df = grouped_df.sort_values('percentage', ascending=False)

    #create a bar plot using the sorted DataFrame
    plt.figure(figsize=(10, 6))
    ax = grouped_df['percentage'].plot(kind='bar')
    plt.xlabel('Country')
    plt.ylabel('%')
    plt.title('% of New Cases out of Tests Done by Country')
    plt.xticks(rotation=90)

    #add the values to the plot
    for i, v in enumerate(grouped_df['percentage']):
        ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=5)
    plt.tight_layout()
    plt.savefig("Q14.png")
    plt.show()

#אחוז הנדבקים החדשים בקורונה (מתוך כלל הנדבקים), בכל עונות השנה
def Q15(df):
    #extract the year and week number from the year_week column
    df[['year', 'week']] = df['year_week'].str.split('-', expand=True)

    #convert week column to integer
    df['week'] = df['week'].str.replace('W', '').astype(int)

    #determine the season for each week
    def get_season(week):
        if week <= 13:
            return "winter"
        elif week <= 26:
            return "spring"
        elif week <= 39:
            return "summer"
        else:
            return "fall"

    df['season'] = df['week'].apply(get_season)

    #calculate the total sum of new_cases
    total_cases = df['new_cases'].sum()

    #group the data by season and calculate the count of new_cases for each season
    grouped_df = df.groupby('season')['new_cases'].count()

    colors = ['#EDBB99', '#ABEBC6', '#F9E79F', '#AED6F1']

    #create a pie chart of count of new_cases for each season out of the total sum
    grouped_df.plot(kind='pie', autopct='%1.1f%%', colors=colors)
    plt.axis('equal')
    plt.title('Cases by Season')
    plt.xlabel(f"Total Cases: {total_cases}")
    plt.savefig("Q15.png")
    plt.show()

#חציון המתים עבור המדינות המובילות
def Q16(df):
    df = df[df.year == 2021]
    df = df[df['deaths'].notna()]

    deaths_dict = {}  # Empty dictionary

    for index, row in df.iterrows():
        key = row['countriesAndTerritories']
        val = row['deaths']
        if key not in deaths_dict:
            deaths_dict[key] = val
        elif isinstance(deaths_dict[key], list):
            deaths_dict[key].append(val)
        else:
            deaths_dict[key] = [deaths_dict[key], val]

    sum_deaths_dict = {}
    for key in deaths_dict:
        sum_deaths_dict[key] = sum(deaths_dict[key])

    #find 5 countries with the highes number of deaths
    top_five = nlargest(5, sum_deaths_dict, key=deaths_dict.get)

    top_five_dict = {}
    for key in deaths_dict:
        if key in top_five:
            top_five_dict[key] = deaths_dict[key]

    fig, ax = plt.subplots()
    ax.boxplot(top_five_dict.values())
    ax.set_xticklabels(top_five_dict.keys())
    plt.xlabel('The top 5 countries')
    plt.title('Median number of deaths for the top 5 countries')
    plt.savefig("Q16.png")
    plt.show()

#דירוג מקרי המוות לפי חודשים ב5 המדינות המובילות בשנת 2021
def Q17(df):
    #filter the data for deaths and year 2021
    deaths_data = df[(df['deaths'] > 0) & (df['year'] == 2021)]

    #group the data by month, year, and countriesAndTerritories
    grouped_data = deaths_data.groupby(['month', 'year', 'countriesAndTerritories'])['deaths'].sum().reset_index()

    #find the top 5 countries
    top_countries = grouped_data.groupby('countriesAndTerritories')['deaths'].sum().nlargest(5).index

    #filter the data for the top 5 countries and non-null deaths
    filtered_data = grouped_data[
        grouped_data['countriesAndTerritories'].isin(top_countries) & ~grouped_data['deaths'].isnull()]

    #prepare the data for plotting
    pivot_data = filtered_data.pivot_table(index='month', columns='countriesAndTerritories', values='deaths',aggfunc='sum')

    #create the heat map with a logarithmic scale
    plt.figure(figsize=(10, 6))
    sns.heatmap(np.log1p(pivot_data), annot=True, cmap='YlOrRd')
    plt.title('Deaths by Country and Month (2021)')
    plt.xlabel('Country')
    plt.ylabel('Month')
    plt.savefig("Q17.png")
    plt.show()

#סך מקרי המוות לפי שנים פר מדינה
def Q18(df):

    grouped = df.groupby(["countriesAndTerritories", "year"])['deaths'].sum().reset_index(name="sum")

    # get unique year
    years = grouped["year"].unique()

    # define a list of colors
    colors = ["#48FA56", "#33C13E", "#3B5D19", "k", "c", "y"]

    # create the plot
    plt.figure(figsize=(10, 6))
    for i, year in enumerate(years):
        filtered_df = grouped[grouped["year"] == year]
        plt.bar(filtered_df["countriesAndTerritories"], filtered_df["sum"], color=colors[i], label=year)

    plt.xlabel("Country")
    plt.ylabel("Number of Deaths")
    plt.title("Number of Deaths of Each Year for Each Country")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Q18.png")
    plt.show()

# ממוצע כמות הבדיקות שנעשו פר אזרח עבור כל מדינה
def Q19(df):
        # drop rows with missing values in 'new_cases' or 'tests_done' columns
        df = df.dropna(subset=['tests_done'])

        # group the data by country and calculate the sum of new_cases and tests_done for each country
        grouped_df = df.groupby('country').sum()

        # group the data by the country name and take the first pop value
        country_populations = df.groupby('country')['population'].first()

        # calculate avg tests done per country
        grouped_df['Avg'] = grouped_df['tests_done'] / country_populations

        # sort the DataFrame by the country name in descending order
        grouped_df = grouped_df.sort_values('country', ascending=True)

        # create a bar plot using the sorted DataFrame
        plt.figure(figsize=(10, 6))
        ax = grouped_df['Avg'].plot(kind='bar')
        plt.xlabel('Country')
        plt.ylabel('Avg')
        plt.title('Avg tests carried per week by country')
        plt.xticks(rotation=90)

        # add the values to the plot
        for i, v in enumerate(grouped_df['Avg']):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=5)
        plt.tight_layout()
        plt.savefig("Q19.png")
        plt.show()

#אחוז מקרי המוות מכלל המקרים בשנת 2021 עבור כל רבעון
def Q20(df):
    #convert 'dateRep' column to datetime
    df['dateRep'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')

    #filter data for the year 2021
    df = df[df['dateRep'].dt.year == 2021].copy()

    #calculate total deaths for each quarter
    df['quarter'] = pd.PeriodIndex(df['dateRep'], freq='Q')
    quarterly_deaths = df.groupby('quarter')['deaths'].sum()

    #create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(quarterly_deaths, labels=quarterly_deaths.index, autopct='%1.1f%%', radius=0.8)
    plt.title('Deaths from all death cases in 2021 (by quarter)')
    plt.savefig("Q20.png")
    plt.show()

#דשבורד – דירוג מקרי המוות בכל מדינה על פני מפה ונתונים סטטיסטים: סה"כ מקרי המוות וחמשת המדינות המובילות במספר מקרי המוות
def Q21(df):
    #calculate the sum of deaths by country
    sum_deaths = df.groupby('countriesAndTerritories')['deaths'].sum().reset_index()

    #load the geospatial data (shapefile) for countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    #merge the geospatial data with the sum of deaths
    merged = world.merge(sum_deaths, left_on='name', right_on='countriesAndTerritories')

    #plot the map
    fig, ax = plt.subplots(figsize=(15, 10))
    merged.plot(column='deaths', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    merged.apply(
        lambda x: ax.annotate(text=x['countriesAndTerritories'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

    ax.set_xlim(-30, 50)
    ax.set_ylim(25, 75)

    total_deaths = sum_deaths['deaths'].sum()
    ax.text(-29, 73, f'Total Deaths: {int(total_deaths):,}', fontsize=12, ha='left')

    top_countries = sum_deaths.nlargest(5, 'deaths')
    top_countries_text = '\n'.join([f'{country}: {int(deaths):,}' for country, deaths in zip(top_countries['countriesAndTerritories'], top_countries['deaths'])])
    ax.text(-29, 26, f'Top 5 Countries:\n{top_countries_text}', fontsize=12, ha='left')

    #add title and labels
    ax.set_title('Total Deaths by Country')
    plt.savefig("Q21.png")
    plt.show()


