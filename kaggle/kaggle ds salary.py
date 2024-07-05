import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import country_converter as cc
# Load dataset and convert it to pandas dataframe
file = pd.read_csv('ds_salaries.csv')
df = pd.DataFrame(file)
# the dataset features
# notice that All the columns are not null

df.info()
df.describe()
# chek if there are duplicates
# drop duplicates if exists

df.duplicated().sum()
df.drop_duplicates(keep = "first", inplace = True)
df1 = df.drop(['salary', 'salary_currency'], axis=1)
df1.head(1)
print('Experience level: ', df1['experience_level'].unique())
print('Employment type: ', df1['employment_type'].unique())
print('Employee residence: ', df1['employee_residence'].unique())
print('Company location: ', df1['company_location'].unique())
print('Company size: ', df1['company_size'].unique())

df_location = df1.groupby(by = "employee_residence")[["company_size"]].count().reset_index()
df_location = df_location.rename(columns ={"company_size":"number of workers"})
df_location['ISO3'] = cc.convert(names=df_location['employee_residence'], to='ISO3')


print(df_location)
fig = go.Figure(data=go.Choropleth(
    locations=df_location['ISO3'],
    z=df_location['number of workers'],
    text=df_location['ISO3'],
    autocolorscale=False,
    reversescale=False,
    colorbar_title='Numbers',
))

fig.update_layout(
    title_text='Employee residence location',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    width=1000,
    height=700
)

fig.show()

sns.histplot(data=df1, x="salary_in_usd", kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary in USD')
plt.ylabel('Count')
plt.tight_layout()
plt.gcf().set_size_inches(10, 7)
plt.show()

top_30_salary = df1[df1["experience_level"] == "EN"].groupby(by="job_title")[["salary_in_usd"]].mean().round().reset_index()
top_30_salary = top_30_salary.sort_values(by ="salary_in_usd", ascending = False ).head(30)
top_30_salary = top_30_salary.rename(columns = {"salary_in_usd": "Average salary in usd"})


fig = px.bar(top_30_salary,
             x="Average salary in usd",
             y='job_title',
             orientation='h',
             color='Average salary in usd',
             color_continuous_scale='Viridis',
             labels={'Average salary in usd': 'Average Salary in USD', 'job_title': 'Job Title'},
             title='Top 30 Job Titles by Average Salary')

fig.update_layout(yaxis={'categoryorder': 'total ascending'}, width=1000, height=700)
fig.show()

df2 = df1.groupby(by=["work_year", "experience_level"]).agg(average_salaries=("salary_in_usd", "mean")).round(2).reset_index()
df2_EN = df2[df2["experience_level"] == "EN"].sort_values(by="work_year")
df2_MI = df2[df2["experience_level"] == "MI"].sort_values(by="work_year")
df2_SE = df2[df2["experience_level"] == "SE"].sort_values(by="work_year")
df2_EX = df2[df2["experience_level"] == "EX"].sort_values(by="work_year")
colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

plt.figure(figsize=(10, 7))
plt.bar(range(0, 8, 2), df2_EN["average_salaries"], width=0.3, label='EN', color=colors[0])
plt.bar(np.arange(0.35, 8.35, 2), df2_MI["average_salaries"], width=0.3, label='MI', color=colors[1])
plt.bar(np.arange(0.7, 8.7, 2), df2_SE["average_salaries"], width=0.3, label='SE', color=colors[2])
plt.bar(np.arange(1.05, 9.05, 2), df2_EX["average_salaries"], width=0.3, label='EX', color=colors[3])

plt.xticks(np.arange(0.5, 8.5, 2), df2_EN["work_year"], rotation=30)
plt.xlabel("Years")
plt.ylabel("Average Salaries")
plt.title("Average Salaries of different experience levels in all countries from 2020 to 2023")
plt.legend()
plt.tight_layout()
plt.show()

df5 = df1.copy()
df5['experience_level'] = df5['experience_level'].replace({
    'EN': 'Entry-level/Junior',
    'MI': 'Mid-level/Intermediate',
    'SE': 'Senior-level/Expert',
    'EX': 'Executive-level/Director'
})
job_levels = df5['experience_level'].value_counts().reset_index()
job_levels.columns = ['experience_level', 'count']

fig = px.pie(job_levels, values='count', names='experience_level', color='experience_level',
             title='Experience Level Distribution',
             color_discrete_map={'Entry-level/Junior': 'lightcyan', 'Mid-level/Intermediate': 'royalblue', 'Senior-level/Expert': 'darkblue', 'Executive-level/Director': 'cyan'})

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(width=1000, height=700)
fig.show()

df5 = df1.copy()
df5['experience_level'] = df5['experience_level'].replace({
    'EN': 'Entry-level/Junior',
    'MI': 'Mid-level/Intermediate',
    'SE': 'Senior-level/Expert',
    'EX': 'Executive-level/Director'
})

job_levels = df5['experience_level'].value_counts().reset_index()
job_levels.columns = ['experience_level', 'count']
avg_salaries = df1.groupby('employment_type')['salary_in_usd'].mean().round(0).sort_values(ascending=False).reset_index()
colors = ["#5e81ac", "#a3be8c", "#d08770", "#bf616a"]

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(ax=ax, data=df1, x='employment_type', y='salary_in_usd', errorbar=None, hue='work_year', palette=colors)
ax.set(xlabel='', ylabel='Dollars', title='Average Salaries in Dollars Per Year')
ax.bar_label(ax.containers[3], padding=2)

plt.tight_layout()
plt.show()

df_size = df1['company_size'].value_counts().reset_index()
df_size.columns = ['company_size', 'count']
df_size['company_size'] = df_size['company_size'].replace({'M': 'Medium', 'S': 'Small', 'L': 'Large'})

fig = px.pie(df_size, values='count', names='company_size', color='company_size',
             title='Company Size Distribution',
             color_discrete_map={'Small': 'lightcyan', 'Medium': 'darkblue', 'Large': 'royalblue'})

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(width=1000, height=700)
fig.show()

df4 = df1.groupby(by=["work_year", "remote_ratio"]).agg(number=("job_title", "count")).reset_index()
df4_count = df4.groupby(by="work_year").agg(total_number=("number", "sum")).reset_index()
merged4 = df4.merge(df4_count, how="inner", on="work_year")
merged4["ratio"] = (merged4["number"] / merged4["total_number"] * 100).round(2)
merged4_0 = merged4[merged4["remote_ratio"] == 0]
merged4_50 = merged4[merged4["remote_ratio"] == 50]
merged4_100 = merged4[merged4["remote_ratio"] == 100]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=merged4_0["work_year"],
    y=merged4_0["ratio"],
    mode='lines+markers+text',
    name='Not Remote',
    text=merged4_0["ratio"].apply(lambda x: f"{x:.2f}%"),
    textposition='bottom center',
    line=dict(color='#443266')
))

fig.add_trace(go.Scatter(
    x=merged4_50["work_year"],
    y=merged4_50["ratio"],
    mode='lines+markers+text',
    name='Half Remote',
    text=merged4_50["ratio"].apply(lambda x: f"{x:.2f}%"),
    textposition='bottom center',
    line=dict(color='#5BB275')
))

fig.add_trace(go.Scatter(
    x=merged4_100["work_year"],
    y=merged4_100["ratio"],
    mode='lines+markers+text',
    name='Fully Remote',
    text=merged4_100["ratio"].apply(lambda x: f"{x:.2f}%"),
    textposition='bottom center',
    line=dict(color='#BDD964')
))

fig.update_layout(
    title='The Change of Remote Ratios in All Countries in Each Year',
    xaxis_title='Year',
    yaxis_title='Percentage %',
    legend_title='Remote Ratio',
    xaxis=dict(tickmode='linear', tick0=2020, dtick=1),
    width=1000,
    height=700
)
fig.show()

iso_country_codes = {
    'PL': 'POL', 'FR': 'FRA', 'IL': 'ISR', 'RU': 'RUS', 'CN': 'CHN', 'DK': 'DNK', 'CA': 'CAN', 'GB': 'GBR', 'US': 'USA',
    'AU': 'AUS', 'FI': 'FIN', 'EE': 'EST', 'AT': 'AUT', 'RO': 'ROU', 'MX': 'MEX', 'DE': 'DEU', 'ID': 'IDN', 'ES': 'ESP',
    'CR': 'CRI', 'BS': 'BHS', 'MY': 'MYS', 'PK': 'PAK', 'LU': 'LUX', 'AS': 'ASM', 'CZ': 'CZE', 'SG': 'SGP', 'IN': 'IND',
    'AL': 'ALB', 'CH': 'CHE'
}
foreign_employees = df1[df1['employee_residence'] != df1['company_location']]
foreign_employees = foreign_employees.groupby(by = "company_location")[["salary_in_usd"]].mean().round().reset_index()
foreign_employees = foreign_employees.rename(columns={'salary_in_usd': 'mean_salary_in_usd'})
foreign_employees = foreign_employees.sort_values(by = "mean_salary_in_usd",ascending=False)
print(foreign_employees)

local_employees = df1[df1['employee_residence'] == df1['company_location']]
local_employees = local_employees.groupby(by = "company_location")[["salary_in_usd"]].mean().round().reset_index()
local_employees = local_employees.rename(columns={'salary_in_usd': 'mean_salary_in_usd'})
local_employees = local_employees.sort_values(by = "mean_salary_in_usd",ascending=False)
print(local_employees)

local_employees['country_code'] = local_employees['company_location'].map(iso_country_codes)
foreign_employees['country_code'] = foreign_employees['company_location'].map(iso_country_codes)
min_salary = min(foreign_employees['mean_salary_in_usd'].min(), local_employees['mean_salary_in_usd'].min())
max_salary = max(foreign_employees['mean_salary_in_usd'].max(), local_employees['mean_salary_in_usd'].max())

fig = go.Figure(data=go.Choropleth(
    locations=foreign_employees['country_code'],
    z=foreign_employees['mean_salary_in_usd'],
    text=foreign_employees['company_location'],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix='$',
    colorbar_title='Mean Salary<br>USD',
    zmin=min_salary,
    zmax=max_salary
))

fig.update_layout(
    title_text='Average Salary of foreign workers by Company Location',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    width=1000,
    height=700
)
fig.show()

fig2 = go.Figure(data=go.Choropleth(
    locations=local_employees['country_code'],
    z=local_employees['mean_salary_in_usd'],
    text=local_employees['company_location'],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix='$',
    colorbar_title='Mean Salary<br>USD',
    zmin=min_salary,
    zmax=max_salary
))

fig2.update_layout(
    title_text='Average Salary of local workers by Company Location',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    width=1000,
    height=700
)
fig2.show()
