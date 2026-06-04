# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 16:24:10 2025

@author: advay
"""

import pandas as pd
import statsmodels.api as sm

# Load your conjoint data
file_path = r"C:\Users\advay\Downloads\final_conjoint_ready.csv"
df = pd.read_csv(file_path)
df.head()
'''
Scenario  Discount Type  Minimum Spend  Expiry  Redemption  Rating
0         4              4              6       4           5     4.6
1         5              5              5       5           5     5.0
2         5              2              2       1           5     3.0
3         3              2              2       2           3     2.4
4         5              5              7       7           5     5.8
'''
# Ensure proper datatypes
categorical_cols = ["Discount Type", "Minimum Spend", "Expiry", "Redemption"]
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")  # force numeric ratings

# Convert categorical to category type
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Dummy encode
X = pd.get_dummies(df[categorical_cols], drop_first=True)

# Add constant
X = sm.add_constant(X)

# Convert everything to float (important!)
X = X.astype(float)
y = df["Rating"].astype(float)

# Fit OLS model
model = sm.OLS(y, X).fit()

# Print summary
print(model.summary())
'''
 OLS Regression Results                            
==============================================================================
Dep. Variable:                 Rating   R-squared:                       0.943
Model:                            OLS   Adj. R-squared:                  0.937
Method:                 Least Squares   F-statistic:                     156.7
Date:                Sun, 09 Nov 2025   Prob (F-statistic):          1.93e-127
Time:                        08:24:53   Log-Likelihood:                -31.066
No. Observations:                 253   AIC:                             112.1
Df Residuals:                     228   BIC:                             200.5
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
const               0.9861      0.077     12.744      0.000       0.834       1.139
Discount Type_2     0.4854      0.089      5.462      0.000       0.310       0.660
Discount Type_3     0.5473      0.077      7.082      0.000       0.395       0.700
Discount Type_4     0.8586      0.076     11.299      0.000       0.709       1.008
Discount Type_5     1.1323      0.076     14.937      0.000       0.983       1.282
Discount Type_6     1.4339      0.082     17.419      0.000       1.272       1.596
Discount Type_7     1.7073      0.088     19.440      0.000       1.534       1.880
Minimum Spend_2     0.2559      0.095      2.690      0.008       0.068       0.443
Minimum Spend_3     0.5790      0.083      6.981      0.000       0.416       0.742
Minimum Spend_4     0.8177      0.078     10.462      0.000       0.664       0.972
Minimum Spend_5     1.0208      0.084     12.109      0.000       0.855       1.187
Minimum Spend_6     1.2258      0.079     15.573      0.000       1.071       1.381
Minimum Spend_7     1.5115      0.082     18.337      0.000       1.349       1.674
Expiry_2            0.1732      0.071      2.442      0.015       0.033       0.313
Expiry_3            0.4937      0.066      7.465      0.000       0.363       0.624
Expiry_4            0.6835      0.067     10.207      0.000       0.552       0.815
Expiry_5            0.9635      0.070     13.853      0.000       0.826       1.101
Expiry_6            1.0797      0.079     13.748      0.000       0.925       1.234
Expiry_7            1.3531      0.103     13.173      0.000       1.151       1.555
Redemption_2        0.2323      0.078      2.975      0.003       0.078       0.386
Redemption_3        0.4999      0.078      6.449      0.000       0.347       0.653
Redemption_4        0.5783      0.078      7.438      0.000       0.425       0.732
Redemption_5        0.7815      0.075     10.407      0.000       0.634       0.929
Redemption_6        1.0195      0.081     12.653      0.000       0.861       1.178
Redemption_7        1.1133      0.104     10.663      0.000       0.908       1.319
==============================================================================
Omnibus:                        0.727   Durbin-Watson:                   1.949
Prob(Omnibus):                  0.695   Jarque-Bera (JB):                0.542
Skew:                           0.106   Prob(JB):                        0.762
Kurtosis:                       3.081   Cond. No.                         12.4
==============================================================================
'''
# Extract part-worth utilities
part_worths = model.params.drop("const")
print("\nPart-Worth Utilities:\n", part_worths)
'''
Part-Worth Utilities:
 Discount Type_2    0.485353
Discount Type_3    0.547322
Discount Type_4    0.858559
Discount Type_5    1.132333
Discount Type_6    1.433856
Discount Type_7    1.707346
Minimum Spend_2    0.255929
Minimum Spend_3    0.578968
Minimum Spend_4    0.817722
Minimum Spend_5    1.020779
Minimum Spend_6    1.225802
Minimum Spend_7    1.511505
Expiry_2           0.173168
Expiry_3           0.493658
Expiry_4           0.683513
Expiry_5           0.963478
Expiry_6           1.079718
Expiry_7           1.353091
Redemption_2       0.232259
Redemption_3       0.499895
Redemption_4       0.578349
Redemption_5       0.781509
Redemption_6       1.019529
Redemption_7       1.113346
'''
importance = {}
for col in categorical_cols:
    # Find levels belonging to this attribute
    levels = [lvl for lvl in part_worths.index if lvl.startswith(col)]
    if levels:
        range_val = part_worths[levels].max() - part_worths[levels].min()
        importance[col] = range_val

# Convert to DataFrame
importance_df = pd.DataFrame.from_dict(importance, orient="index", columns=["Range"])
importance_df["Relative Importance (%)"] = (
    100 * importance_df["Range"] / importance_df["Range"].sum()
)

print("\nPart-worth utilities:")
print(part_worths)

print("\nAttribute importance:")
print(importance_df.round(2))
'''
Range  Relative Importance (%)
Discount Type   1.22                    26.92
Minimum Spend   1.26                    27.66
Expiry          1.18                    26.00
Redemption      0.88                    19.41
'''

# SEM updated #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import semopy
from semopy import Model
from semopy.inspector import inspect
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from semopy.inspector import inspect
from semopy import Model, calc_stats, semplot
import graphviz
from semopy import Model, semplot
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from pingouin import cronbach_alpha
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r"C:\Users\advay\Downloads\Tpb - Sheet1.csv")
df.head()
'''
a1  a2  a3  a4  a5  sn1  sn2  ...  pb4  bint1  bint2  bint3  beh1  beh2  beh3
0   5   6   5   5   6    5    5  ...    5      5      3      5     3     2     6
1   5   5   5   5   5    5    5  ...    5      5      5      5     2     1     5
2   7   7   7   7   6    5    5  ...    6      6      6      6     2     1     5
3   4   4   3   6   3    2    4  ...    4      4      3      3     2     2     2
4   7   7   7   7   7    7    7  ...    7      7      7      7     2     2     7
'''
data=pd.read_csv(r"C:\Users\advay\Downloads\dda.csv")
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum()) # no null values

# Data preprocessing
def preprocess_data(df):
    """Preprocess the data for factor analysis"""
    # Remove any completely empty rows/columns
    df_clean = df.dropna(how='all').dropna(axis=1, how='all')
   
    # Fill any remaining missing values with column mean
    df_filled = df_clean.fillna(df_clean.mean())
   
    return df_filled

df_clean = preprocess_data(df)
df_clean.head()
# STEP 1: EXPLORATORY FACTOR ANALYSIS (EFA)

def perform_efa(df, n_factors=None):
    """Perform Exploratory Factor Analysis"""
   
    # Test for factorability
    print("\n" + "="*50)
    print("EXPLORATORY FACTOR ANALYSIS (EFA)")
    print("="*50)
   
    # Bartlett's test of sphericity
    chi_square, p_value = calculate_bartlett_sphericity(df)
    print(f"Bartlett's Test: χ² = {chi_square:.3f}, p = {p_value:.3f}")
   
    if p_value < 0.05:
        print("✓ Data is suitable for factor analysis (Bartlett's test significant)")
    else:
        print("✗ Data may not be suitable for factor analysis")
   
    # KMO test
    kmo_all, kmo_model = calculate_kmo(df)
    print(f"KMO Measure: {kmo_model:.3f}")
    if kmo_model >= 0.8:
        print("✓ Marvelous (KMO > 0.8)")
    elif kmo_model >= 0.7:
        print("✓ Middling (KMO > 0.7)")
    elif kmo_model >= 0.6:
        print("✓ Mediocre (KMO > 0.6)")
    else:
        print("✗ Unacceptable (KMO < 0.6)")
   
    # Determine number of factors using parallel analysis and eigenvalues
    if n_factors is None:
        fa = FactorAnalyzer(rotation=None, method='ml')
        fa.fit(df)
        eigenvalues, _ = fa.get_eigenvalues()
       
        # Kaiser criterion (eigenvalues > 1)
        n_factors_kaiser = sum(eigenvalues > 1)
       
        # Scree plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
        plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue = 1')
        plt.title('Scree Plot')
        plt.xlabel('Factor Number')
        plt.ylabel('Eigenvalue')
        plt.legend()
        plt.grid(True)
        plt.show()
       
        print(f"Kaiser criterion suggests {n_factors_kaiser} factors")
        n_factors = n_factors_kaiser
   
    # Perform EFA
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='ml')
    fa.fit(df)
   
    # Get factor loadings
    loadings = fa.loadings_
   
    # Create factor loading matrix
    loadings_df = pd.DataFrame(
        loadings,
        index=df.columns,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
   
    print("\nFactor Loadings (Pattern Matrix):")
    print(loadings_df.round(3))
   
    # Get communalities
    communalities = fa.get_communalities()
    communalities_df = pd.DataFrame({
        'Variable': df.columns,
        'Communality': communalities
    })
    print("\nCommunalities:")
    print(communalities_df.round(3))
   
    return fa, loadings_df, n_factors

# Perform EFA
efa_model, loadings_df, n_factors = perform_efa(df_clean)
'''
Kaiser criterion suggests 4 factors

Factor Loadings (Pattern Matrix):
       Factor_1  Factor_2  Factor_3  Factor_4
a1        0.695     0.125     0.421     0.075
a2        0.653     0.209     0.263     0.025
a3        0.785     0.194     0.249    -0.059
a4        0.784     0.362     0.103     0.010
a5        0.477     0.469     0.375     0.068
sn1       0.330     0.662     0.151     0.040
sn2       0.390     0.740     0.087     0.098
sn3       0.287     0.674     0.305     0.065
sn4      -0.072     0.447     0.036     0.140
pb1       0.180     0.484     0.384     0.053
pb2       0.281     0.117     0.698     0.126
pb3       0.181     0.265     0.361     0.016
pb4       0.165     0.118     0.617     0.097
bint1     0.353     0.315     0.570     0.127
bint2     0.239     0.367     0.362     0.203
bint3     0.299     0.497     0.324     0.132
beh1      0.079     0.161     0.081     0.978
beh2      0.007     0.084     0.103     0.460
beh3      0.486     0.079     0.303     0.204

Communalities:
   Variable  Communality
0        a1        0.682
1        a2        0.540
2        a3        0.720
3        a4        0.756
4        a5        0.593
5       sn1        0.572
6       sn2        0.716
7       sn3        0.634
8       sn4        0.226
9       pb1        0.417
10      pb2        0.595
11      pb3        0.234
12      pb4        0.431
13    bint1        0.565
14    bint2        0.364
15    bint3        0.458
16     beh1        0.995
17     beh2        0.229
18     beh3        0.376
'''
# Identify which variables belong to which factors
def identify_factors(loadings_df, threshold=0.35):
    """Identify which variables load on which factors"""
    factor_structure = {}
   
    for factor in loadings_df.columns:
        high_loading_vars = loadings_df[loadings_df[factor].abs() > threshold].index.tolist()
        factor_structure[factor] = high_loading_vars
        print(f"\n{factor} (loadings > {threshold}):")
        for var in high_loading_vars:
            loading = loadings_df.loc[var, factor]
            print(f"  {var}: {loading:.3f}")
   
    return factor_structure

factor_structure = identify_factors(loadings_df)
'''
Factor_1 (loadings > 0.35):
  a1: 0.695
  a2: 0.653
  a3: 0.785
  a4: 0.784
  a5: 0.477
  sn2: 0.390
  bint1: 0.353
  beh3: 0.486

Factor_2 (loadings > 0.35):
  a4: 0.362
  a5: 0.469
  sn1: 0.662
  sn2: 0.740
  sn3: 0.674
  sn4: 0.447
  pb1: 0.484
  bint2: 0.367
  bint3: 0.497

Factor_3 (loadings > 0.35):
  a1: 0.421
  a5: 0.375
  pb1: 0.384
  pb2: 0.698
  pb3: 0.361
  pb4: 0.617
  bint1: 0.570
  bint2: 0.362

Factor_4 (loadings > 0.35):
  beh1: 0.978
  beh2: 0.460
  '''
# Interpretation:
# ✅ Alpha > 0.7 → Reliable
# ⚠️ Alpha < 0.6 → Needs review / item refinement

# =========================================
# STEP 5: CFA (CONFIRMATORY FACTOR ANALYSIS)
# =========================================
# Define model syntax from EFA-confirmed constructs
model_desc = """
    # Measurement model
    ATT =~ a1 + a2 + a3 + a4
    SN  =~ sn1 + sn2 + sn3
    PBC =~ pb1 + pb2 + pb3
    INT =~ bint1 + bint2 + bint3
    BEH =~ beh1 + beh2

      # Structural model
      INT ~ ATT + SN + PBC
      BEH ~ INT + PBC
"""

model = Model(model_desc)
model.fit(df_clean)

print("\nCFA Model Fit Indices:")
fit_stats = calc_stats(model)
fit_stats.columns
print(fit_stats[['DoF', 'chi2 p-value', 'CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']])
'''
DoF  chi2 p-value       CFI       TLI     RMSEA        AIC         BIC
Value   82  6.639134e-14  0.922024  0.900152  0.080303  74.298399  208.567199
'''
# Interpretation:
# ✅ Good Fit if:
#    - CFI > 0.90
#    - TLI > 0.90
#    - RMSEA < 0.08
#    - p-value > 0.05 (non-significant → good fit)
# ⚠️ Modify model if poor fit (remove weak items, add correlations)

# =========================================
# STEP 5: Reliability Check (Cronbach’s Alpha)
# =========================================
def alpha_for(items):
    df = df_clean[items]
    return cronbach_alpha(df)[0]

constructs = {
    'ATT': ['a1', 'a2', 'a3', 'a4', 'a5'],
    'SN': ['sn1', 'sn2', 'sn3', 'sn4'],
    'PBC': ['pb1', 'pb2', 'pb3', 'pb4'],
    'INT': ['bint1', 'bint2', 'bint3'],
    'BEH': ['beh1', 'beh2', 'beh3']
}

print("\nCronbach’s Alpha values:")
for c, items in constructs.items():
    print(f"{c}: {alpha_for(items):.3f}")

'''
Cronbach’s Alpha values:
ATT: 0.884
SN: 0.770
PBC: 0.711
INT: 0.765
BEH: 0.427
'''

sem_scores = model.predict_factors(df_clean)

# Display first few rows
print("Latent variable scores (from SEM):")
print(sem_scores.head())
'''
ATT       BEH       INT       PBC        SN
0  0.026253  0.837945  0.317334  0.227933  0.361007
1 -0.076310 -0.162055  0.364420  0.265563  0.363737
2  1.379151 -0.162055  0.951784  0.799076  0.462182
3 -0.874181 -0.162055 -0.980664 -0.804735 -1.124666
4  1.676108 -0.162055  2.139923  1.861451  2.138758
'''

####################################### clustering #######################################33
import os
os.chdir(r"C:\Users\advay\Downloads")
#sem_scores.to_csv('semfinalscore.csv',index=False)
# Assuming you already ran the conjoint model
part_worths_df = part_worths.reset_index()
part_worths_df.columns = ['Attribute_Level', 'Part_Worth']

# Attribute Importance
importance_df = importance_df.reset_index().rename(columns={'index':'Attribute', 'Relative Importance (%)':'Importance'})

print("\nPart-worth utilities:\n", part_worths_df)
'''
Part-worth utilities:
     Attribute_Level  Part_Worth
0   Discount Type_2    0.485353
1   Discount Type_3    0.547322
2   Discount Type_4    0.858559
3   Discount Type_5    1.132333
4   Discount Type_6    1.433856
5   Discount Type_7    1.707346
6   Minimum Spend_2    0.255929
7   Minimum Spend_3    0.578968
8   Minimum Spend_4    0.817722
9   Minimum Spend_5    1.020779
10  Minimum Spend_6    1.225802
11  Minimum Spend_7    1.511505
12         Expiry_2    0.173168
13         Expiry_3    0.493658
14         Expiry_4    0.683513
15         Expiry_5    0.963478
16         Expiry_6    1.079718
17         Expiry_7    1.353091
18     Redemption_2    0.232259
19     Redemption_3    0.499895
20     Redemption_4    0.578349
21     Redemption_5    0.781509
22     Redemption_6    1.019529
23     Redemption_7    1.113346
'''
print("\nAttribute importances:\n", importance_df)
'''
Attribute importances:
        Attribute     Range  Importance
0  Discount Type  1.221994   26.924591
1  Minimum Spend  1.255575   27.664497
2         Expiry  1.179923   25.997623
3     Redemption  0.881088   19.413289
'''
# Save for merging later
#part_worths_df.to_csv("conjoint_partworths.csv", index=False)
#importance_df.to_csv("conjoint_importance.csv", index=False)
#all the files

# Load dda
dda = pd.read_csv(r"C:\Users\advay\Downloads\dda.csv")

# Keep only the psychological profiling columns (clt1–clt7)
psych_cols = ['clt1', 'clt2', 'clt3', 'clt4', 'clt5', 'clt6', 'clt7']
dda = dda[psych_cols].copy()

# Drop missing rows
dda = dda.dropna()

print("Cleaned DDA data shape:", dda.shape)
print(dda.head())
'''
clt1  clt2  clt3  clt4  clt5  clt6  clt7
0     4     3     3     5     6     5     5
1     5     5     5     5     1     5     5
2     5     5     5     2     6     5     6
3     3     3     2     2     5     4     3
4     4     7     7     7     1     6     6
'''

# Select the main constructs
sem_constructs = ['ATT', 'SN', 'PBC', 'INT', 'BEH']
sem_scores = sem_scores[sem_constructs].copy()
print("SEM scores shape:", sem_scores.shape)
# Suppose 'importance_df' is your attribute-level importance DataFrame
importance_df = importance_df.reset_index().rename(columns={'index':'Attribute'})
importance_df.columns
importance_dict = dict(zip(importance_df['Attribute'], importance_df['Importance']))

# Convert to DataFrame (same value repeated for all respondents)
conjoint_df = pd.DataFrame([importance_dict] * len(sem_scores))

print("Conjoint data shape:", conjoint_df.shape)

# Align and merge
merged_data = pd.concat([sem_scores.reset_index(drop=True),
                         dda.reset_index(drop=True),
                         conjoint_df.reset_index(drop=True)], axis=1)

# Drop missing values if any
merged_data = merged_data.dropna()
merged_data.to_csv('forcluster.csv',index=False)
print("Final merged data shape:", merged_data.shape)
print(merged_data.head())
'''
ATT        SN       PBC       INT  ...  clt5  clt6  clt7  Attribute
0  0.026253  0.361007  0.227933  0.317334  ...     6     5     5  27.664497
1 -0.076310  0.363737  0.265563  0.364420  ...     1     5     5  27.664497
2  1.379151  0.462182  0.799076  0.951784  ...     6     5     6  27.664497
3 -0.874181 -1.124666 -0.804735 -0.980664  ...     5     4     3  27.664497
4  1.676108  2.138758  1.861451  2.139923  ...     1     6     6  27.664497
'''
#sem_scores.to_csv("sem_scores.csv", index=False)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
clust=pd.read_csv(r"C:\Users\advay\Downloads\forcluster.csv")
clust.columns
'''
Index(['ATT', 'SN', 'PBC', 'INT', 'BEH', 'clt1', 'clt2', 'clt3', 'clt4',
       'clt5', 'clt6', 'clt7', 'Attribute'],
      dtype='object')
'''
scaled_data=clust[['ATT', 'SN', 'PBC', 'INT', 'BEH', 'clt1', 'clt2', 'clt3', 'clt4',
       'clt5', 'clt6', 'clt7', 'Attribute']]
# Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(scaled_data[['clt1', 'clt2', 'clt3', 'clt4',
       'clt5', 'clt6', 'clt7']])
scaled_data=pd.DataFrame(scaled_data,columns=['clt1', 'clt2', 'clt3', 'clt4',
       'clt5', 'clt6', 'clt7'])
clust1=clust[['ATT', 'SN', 'PBC', 'INT', 'BEH','Attribute']]
scaled_data=pd.concat([scaled_data,clust1],axis=1)
# Compute correlation matrix for psychological variables
corr = scaled_data[['ATT','SN','PBC','INT','BEH']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Psychological Variables')
plt.show()
# Step 2: Apply PCA (keep enough components to explain 90% variance)
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(scaled_data)
print(f"PCA retained {X_pca.shape[1]} uncorrelated components")
inertia, silhouette = [], []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(scaled_data, kmeans.labels_))
# Elbow plot
plt.plot(K, inertia, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
# Silhouette plot
plt.plot(K, silhouette, 'ro-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()
# Step 3: Cluster on PCA-transformed data
kmeans = KMeans(n_clusters=3, random_state=42)
clust['Cluster_PCA'] = kmeans.fit_predict(X_pca)
y_pred=kmeans.fit_predict(X_pca)
# Step 4: Check explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_.sum()) #0.9034862923504161
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_pred, palette='deep', s=70)
plt.title(f"K-Means Clustering (k=3)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()
kmeans.cluster_centers_



#anova
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Variables to test
variables = ['ATT', 'SN', 'PBC', 'INT', 'BEH',
             'clt1', 'clt2', 'clt3', 'clt4', 'clt5', 'clt6', 'clt7']

print("=== One-Way ANOVA Results and Interpretation ===\n")

anova_summary = []

for var in variables:
    # Perform ANOVA
    groups = [clust[clust['Cluster_PCA'] == c][var] for c in sorted(clust['Cluster_PCA'].unique())]
    f_stat, p_val = stats.f_oneway(*groups)
    
    # Get full ANOVA table (like R's aov)
    model = ols(f'{var} ~ C(Cluster_PCA)', data=clust).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    
    # Interpret results
    if p_val < 0.001:
        interpretation = "Highly significant — very strong cluster differentiation"
    elif p_val < 0.05:
        interpretation = "Significant — clusters differ on this variable"
    else:
        interpretation = "Not significant — no meaningful cluster difference"
    
    # Print summary for each variable
    print(f"{var}:")
    print(aov_table)
    print(f"→ F = {f_stat:.3f}, p = {p_val:.5f} → {interpretation}\n{'-'*80}")
    
    anova_summary.append({
        'Variable': var,
        'F-statistic': f_stat,
        'p-value': p_val,
        'Interpretation': interpretation
    })

'''
ATT:
                    sum_sq     df           F        PR(>F)
C(Cluster_PCA)  166.199155    2.0  137.318152  5.757164e-41
Residual        151.290227  250.0         NaN           NaN
→ F = 137.318, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
SN:
                    sum_sq     df           F        PR(>F)
C(Cluster_PCA)  233.597919    2.0  255.144277  4.170562e-61
Residual        114.444033  250.0         NaN           NaN
→ F = 255.144, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
PBC:
                    sum_sq     df           F        PR(>F)
C(Cluster_PCA)  144.277284    2.0  295.728935  1.298536e-66
Residual         60.983754  250.0         NaN           NaN
→ F = 295.729, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
INT:
                    sum_sq     df           F        PR(>F)
C(Cluster_PCA)  219.593272    2.0  302.832608  1.601461e-67
Residual         90.641359  250.0         NaN           NaN
→ F = 302.833, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
BEH:
                    sum_sq     df          F        PR(>F)
C(Cluster_PCA)   17.071068    2.0  16.253867  2.310513e-07
Residual        131.284664  250.0        NaN           NaN
→ F = 16.254, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
clt1:
                    sum_sq     df          F        PR(>F)
C(Cluster_PCA)  269.070079    2.0  54.376516  2.472029e-20
Residual        618.534664  250.0        NaN           NaN
→ F = 54.377, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
clt2:
                    sum_sq     df          F        PR(>F)
C(Cluster_PCA)  204.331611    2.0  38.533454  2.587768e-15
Residual        662.838350  250.0        NaN           NaN
→ F = 38.533, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
clt3:
                    sum_sq     df          F        PR(>F)
C(Cluster_PCA)  199.505734    2.0  41.885688  2.048165e-16
Residual        595.387546  250.0        NaN           NaN
→ F = 41.886, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
clt4:
                    sum_sq     df          F        PR(>F)
C(Cluster_PCA)  335.538435    2.0  73.162524  9.684135e-26
Residual        573.275795  250.0        NaN           NaN
→ F = 73.163, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
clt5:
                    sum_sq     df         F        PR(>F)
C(Cluster_PCA)  130.703549    2.0  23.49407  4.467996e-10
Residual        695.407123  250.0       NaN           NaN
→ F = 23.494, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
clt6:
                    sum_sq     df         F    PR(>F)
C(Cluster_PCA)   27.354887    2.0  5.091712  0.006801
Residual        671.554204  250.0       NaN       NaN
→ F = 5.092, p = 0.00680 → Significant — clusters differ on this variable
--------------------------------------------------------------------------------
clt7:
                    sum_sq     df          F        PR(>F)
C(Cluster_PCA)   91.387487    2.0  20.276065  6.910243e-09
Residual        563.395122  250.0        NaN           NaN
→ F = 20.276, p = 0.00000 → Highly significant — very strong cluster differentiation
--------------------------------------------------------------------------------
'''

# Summary DataFrame
anova_df = pd.DataFrame(anova_summary).sort_values('p-value')
print("\n=== Summary of ANOVA Across All Variables ===")
print(anova_df.to_string(index=False))
'''
Variable  F-statistic      p-value                                           Interpretation
     INT   302.832608 1.601461e-67 Highly significant — very strong cluster differentiation
     PBC   295.728935 1.298536e-66 Highly significant — very strong cluster differentiation
      SN   255.144277 4.170562e-61 Highly significant — very strong cluster differentiation
     ATT   137.318152 5.757164e-41 Highly significant — very strong cluster differentiation
    clt4    73.162524 9.684135e-26 Highly significant — very strong cluster differentiation
    clt1    54.376516 2.472029e-20 Highly significant — very strong cluster differentiation
    clt3    41.885688 2.048165e-16 Highly significant — very strong cluster differentiation
    clt2    38.533454 2.587768e-15 Highly significant — very strong cluster differentiation
    clt5    23.494070 4.467996e-10 Highly significant — very strong cluster differentiation
    clt7    20.276065 6.910243e-09 Highly significant — very strong cluster differentiation
     BEH    16.253867 2.310513e-07 Highly significant — very strong cluster differentiation
    clt6     5.091712 6.800609e-03           Significant — clusters differ on this variable
'''

## logistic regression ##
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,f1_score

# Load data
logistic_df = pd.read_excel(r"C:\Users\advay\Downloads\logistic.xlsx")
sem_scores = pd.read_csv(r"C:\Users\advay\Downloads\semfinalscore.csv")
conjoint_importance = pd.read_csv(r"C:\Users\advay\Downloads\conjoint_importance.csv")

# --- Step 1: Clean logistic data ---
# Convert outcome variable (lr1) to binary: 1 = buy, 0 = not buy
logistic_df["lr1"] = logistic_df["lr1"].apply(lambda x: 1 if x == 1 else 0)

# =============================================================================
# # Encode categorical variable (lr6)
# logistic_df = pd.get_dummies(logistic_df, columns=["lr6"], drop_first=True)
# =============================================================================

# --- Step 2: Merge SEM & Conjoint results ---
# Merge by index (assuming same respondents order)
merged_df = pd.concat([logistic_df, sem_scores], axis=1)
merged_df.head()
'''
clt1  clt2  clt3  clt4  ...       BEH       INT       PBC        SN
0     4     3     3     5  ...  0.837945  0.317334  0.227933  0.361007
1     5     5     5     5  ... -0.162055  0.364420  0.265563  0.363737
2     5     5     5     2  ... -0.162055  0.951784  0.799076  0.462182
3     3     3     2     2  ... -0.162055 -0.980664 -0.804735 -1.124666
4     4     7     7     7  ... -0.162055  2.139923  1.861451  2.138758
'''
# Add conjoint importance as global weights (scaled)
for i, row in conjoint_importance.iterrows():
    merged_df[f"importance_{row['Attribute.3']}"] = row["Importance"]

# --- Step 3: Define predictors and target ---
X = merged_df.drop(columns=["lr1"])
y = merged_df["lr1"]

y.value_counts()
'''
lr1
1    205
0     48
Name: count, dtype: int64
'''
# Optional: scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# --- Step 4: Split & fit logistic regression ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

log_model = LogisticRegression(max_iter=1000)
result=log_model.fit(X_train, y_train)

# --- Step 5: Evaluate ---
y_pred = log_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
'''
Confusion Matrix:
 [[ 7  6]
 [ 2 61]]
 '''
print("\nClassification Report:\n", classification_report(y_test, y_pred))
'''
Classification Report:
               precision    recall  f1-score   support

           0       0.78      0.54      0.64        13
           1       0.91      0.97      0.94        63

    accuracy                           0.89        76
   macro avg       0.84      0.75      0.79        76
weighted avg       0.89      0.89      0.89        76
'''
print("\nAccuracy Score: \n",accuracy_score(y_test,y_pred)) #0.8947368421052632
print("\nf1 Score: \n",f1_score(y_test,y_pred)) #0.9384615384615385
# --- Step 6: Check most influential predictors ---
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nTop Influential Predictors:\n", coeff_df.head(10))
'''
Top Influential Predictors:
                      Feature   Coefficient
7                        lr2  1.237610e+00
15                        SN  3.969628e-01
2                       clt3  3.495963e-01
11                       ATT  3.160868e-01
0                       clt1  3.038876e-01
9                        lr5  2.862235e-01
8                        lr4  2.059453e-01
13                       INT  1.541481e-01
17  importance_Minimum Spend  4.076513e-16
18         importance_Expiry  4.076513e-16
'''

# Create hypothetical offers
offers = pd.DataFrame({
    'Platform': ['Myntra', 'Flipkart'],
    'Discount_Type': ['Percent', 'Flat'],
    'Discount_Value': [20, 300],
    'Min_Spend': [1000, 1200],
    'Expiry': [3, 1],  # in days
    'Timer_Shown': [1, 0],  # 1=Yes, 0=No
    # Now include SEM-related latent variables (approx mean or cluster-based)
    'Attitude': [4.1, 4.3],
    'SN': [3.8, 4.0],
    'PBC': [4.0, 3.9],
    # Conjoint-based overall importance (you can pick one representative value)
    'Importance_Price': [0.75, 0.78],
    'Importance_Discount': [0.80, 0.82],
    'Importance_Expiry': [0.70, 0.68]
})

# Get model feature names from training data
model_features = list(X.columns)

# Align new offers with model features — fill missing with 0 if not present
for col in model_features:
    if col not in offers.columns:
        offers[col] = 0

# Ensure same column order
offers = offers[model_features]
# Apply the same scaling
offers_scaled = scaler.transform(offers)
# Predict purchase probabilities
offers['Purchase_Prob'] = log_model.predict_proba(offers_scaled)[:, 1]

print(offers[['Purchase_Prob']])
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,5))
sns.barplot(data=offers, x=['Myntra', 'Flipkart'], y=offers['Purchase_Prob'], palette='mako')
plt.title("Predicted Purchase Probability by Platform", fontsize=14, fontweight='bold')
plt.ylabel("Purchase Probability")
plt.xlabel("")
for i, val in enumerate(offers['Purchase_Prob']):
    plt.text(i, val + 0.02, f"{val*100:.1f}%", ha='center', fontweight='bold')
plt.ylim(0, 1)
plt.show()


















