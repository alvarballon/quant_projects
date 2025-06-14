---
header-includes:
    - \usepackage[a4paper,
            bindingoffset=0.2in,
            left=1in,
            right=1in,
            top=1in,
            bottom=1in,
            footskip=.25in]{geometry}
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[R]{Alvaro Ballon Bordo}
    - \usepackage[most]{tcolorbox}
    - \usepackage{titlesec}
    - \usepackage{parskip}
    - \usepackage{empheq}
    - \usepackage[table, dvipsnames]{xcolor}
    - \definecolor{lightgreen}{HTML}{75E6DA}
    - \newcommand{\coloredeq}[1]{\begin{empheq}[box=\fcolorbox{black}{lightgreen!50!}]{align}#1\end{empheq}}
    - \definecolor{linequote}{RGB}{224,215,188}
    - \definecolor{backquote}{RGB}{249,245,233}
    - \definecolor{bordercolor}{RGB}{221,221,221}
---

### Backtesting of $99\%$/$10$-day Value at Risk

To determine the VaR breaches, we first need to calculate the following:

- the daily returns,
- the rolling 21-day standard deviations of the daily returns,
- the forward-realized 10-day returns,
- the 10-day-scaled standard deviations,
- the VaR.

Using `pandas`, let us import the `.csv` file and create `dataframe` with this information.

```python
df = pd.read_csv('data.csv')
df["returns"] = np.log(df["SP500"]/df["SP500"].shift(1))
df["fwd 10 day returns"] = np.log(df["SP500"].shift(-10)/df["SP500"])
df["deviations"] = df["returns"].rolling(21).std()
df["10 day deviation"] = np.sqrt(10 * df["deviations"]**2)
df["VaR"] = norm.ppf(0.01)*df["10 day deviation"]
```

Note that the VaR is taken to be negative, since we calulate returns, but we want to quantify loses. We show the output of `df.dropna().head()` below.

![](table1.png)

To quantify the VaR breaches, we check whether the forward realized 10-day returns are smaller than the VaR. We create a `"Breach"` column which contains a `1` if there is a breach and `0` otherwise.

```python
df["Breach"] = np.where( (df["fwd 10 day returns"] < 0)
& (df["fwd 10 day returns"] < df["VaR"]), 1, 0)
```

Below are some examples of breaches obtained by printing `df[df["Breach"]==1]`.

![](table2.png)

We can add up the `1`s to count the VaR breaches. Dividing by the length of the full dataframe yields the percentage of VaR breaches in the period of time under study.

```python
var_breach_count = np.sum(df["Breach"])
var_breach_pct = var_breach_count/len(df.dropna())
print(f"VaR breach counts: {var_breach_count} \n
VaR breach percentage: {var_breach_pct}")
```

```python
>> VaR breach counts: 25 
VaR breach percentage: 0.020508613617719443
```

We have found:

\coloredeq{\text{VaR breaches} = 25 \nonumber}
\coloredeq{\text{Breach percentage} = 2.05\% \nonumber}

We proceed to calculate consecutive VaR breaches, we leverage `"Breach"` column we created before to generate a `"Consecutive"` column. The element of this column are computed by multiplying the value of the `"Breach"` column at time $t$ by the value at time $t+1$. This means it will take the value $1$ if and only if there are consecutive breaches. 

```python
df["Consecutive"] = df["Breach"]*df["Breach"].shift(-1)
```

Indeed, if we print the table again, we observe that the behaviour is as expected.

![](table3.png)

We then carry out a similar calculation as the previous one to find the count and percentage.

```python
consecutive_count = np.sum(df["Consecutive"])
consecutive_pct = consecutive_count/len(df.dropna())
print(f"Consecutive breach counts: {consecutive_count}
\nVaR Consecutive breach percentage: {consecutive_pct}")
```

```python
>> Consecutive breach counts: 14.0 
VaR Consecutive breach percentage: 0.011484823625922888
```

We have found:

\coloredeq{\text{Consecutive breaches} = 14 \nonumber}
\coloredeq{\text{Consecutive breach percentage} = 1.15\% \nonumber}

The following `matplotlib.pyplot` script produces a plot that shows the percentages losses, value at risk, and breaches.

```python
x = np.arange(len(df))
y_1 = -1*df["VaR"]
y_2 = -1*df["fwd 10 day returns"]
x_breaches = df[df["Breach"] ==1].index
y_breaches = df[df["Breach"]==1]["fwd 10 day returns"]*(-1)
plt.scatter(x_breaches, y_breaches, marker = '*', 
zorder = 10, s=50, c = "#d8581c", label = "VaR breaches")
plt.plot(x, y_1, c = '#189ab4', label = "VaR")
plt.plot(x, y_2, c = '#05445e', label = "Negative returns")
plt.legend(loc='upper right')
plt.xlabel(r"$\textrm{Time (days)}$", fontsize = 12)
plt.ylabel(r"$\textrm{Percentage loss (/100\%)}$", fontsize = 12)
plt.title(r"$\textrm{S\&P 500 Value at Risk breaches}$", fontsize = 14)
```

![](plot2.png)

### Backtesting using EWMA

We create another dataframe `ewma_df` and define the returns and 10-day-forward returns in the same way as before.

```python
ewma_df = pd.read_csv('data.csv')
ewma_df["returns"] = np.log(ewma_df["SP500"]/ewma_df["SP500"].shift(1))
ewma_df["fwd 10 day returns"] = np.log(ewma_df["SP500"].shift(-10)/ewma_df["SP500"])
```

For the standard deviations, however, we use EWMA with $\lambda = 0.72.$ Since the first return on the table is `nan`, we fill that standard deviation with a zero and use the standard deviation `sigma` for the whole set for the *second* row in the table. The subsequent elements are filled using the EWMA formula recursively, as shown in the code snippet below.

```python
sigma = ewma_df["returns"].std()
lambda_ = 0.72 
deviations = np.zeros(len(ewma_df))
deviations[1] = sigma
for i in range(len(ewma_df)-2):

    deviations[i+2] = np.sqrt(lambda_*deviations[i+1]**2
    + (1-lambda_)*ewma_df["returns"][i+1]**2)

ewma_df["deviations"] = deviations
```

Having calculated the standard deviation, we can rescale them to calculate $\sigma_{10D}$ and $\text{VaR}_{10D}.$

```python
ewma_df["10 day deviation"] = np.sqrt(10 * ewma_df["deviations"]**2)
ewma_df["VaR"] = norm.ppf(0.01)*ewma_df["10 day deviation"]
```

We use the same procedure as before to find the amount of VaR breaches. 

```python
ewma_df["Breach"] = np.where( (ewma_df["fwd 10 day returns"] < 0) & 
(ewma_df["fwd 10 day returns"] < ewma_df["VaR"]), 1, 0)
ewma_breach_count = np.sum(ewma_df["Breach"])
ewma_breach_pct = var_breach_count/len(ewma_df.dropna())
print(f"VaR breach counts: {ewma_breach_count} \nVaR breach percentage: {ewma_breach_pct}")
```

```python
>> VaR breach counts: 32 
VaR breach percentage: 0.020177562550443905
```

We have found 

\coloredeq{\text{VaR breaches} = 32 \nonumber}
\coloredeq{\text{Breach percentage} = 2.01\% \nonumber}

We note that even though the number of breaches found with this method is higher, the percentage is smaller. The reason is that, since we did not calculate the standard deviation using rolling windows, fewer data points needed to be discarded. Indeed, we were now able to assign a standard deviation to the first data points. 

The code to compute this is almost identical as in Question 6.

```python
ewma_df["Consecutive"] = ewma_df["Breach"]*ewma_df["Breach"].shift(-1)
ewma_consecutive_count = np.sum(ewma_df["Consecutive"])
ewma_consecutive_pct = consecutive_count/len(ewma_df.dropna())
print(f"Consecutive breach counts: {ewma_consecutive_count}
 \nVaR Consecutive breach percentage: {ewma_consecutive_pct}")
```

```python
Consecutive breach counts: 17.0 
VaR Consecutive breach percentage: 0.011299435028248588
```

We have found

\coloredeq{\text{Consecutive VaR breaches} = 17 \nonumber}
\coloredeq{\text{Consecutive breach percentage} = 1.13\% \nonumber}

The same `pyplot` script as in Question 6 changing all instances of `df` with `ewma_df` produces the plot shown below.

![](plot3.png)