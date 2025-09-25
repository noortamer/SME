import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import arabic_reshaper
from bidi.algorithm import get_display
import warnings
warnings.filterwarnings('ignore')

# Global settings
plt.rcParams['font.family'] = 'Amiri'
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

DATA_PATH = 'data/finalscores.csv'
df_global = None

def load_data():
    global df_global
    print("Loading dataset...")
    df_global = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df_global.shape[0]:,} rows × {df_global.shape[1]} columns")
    return df_global

def data_overview_and_quality():
    global df_global
    df = df_global.copy()
    
    print("="*80)
    print("1. DATA OVERVIEW AND QUALITY CHECKS")
    print("="*80)
    
    # Basic info
    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print()
    
    # Data types analysis
    print("DATA TYPES ANALYSIS:")
    print("-" * 40)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    print()
    
    # Detailed column info
    print("COLUMN INFORMATION:")
    print("-" * 60)
    for i, (col, dtype) in enumerate(df.dtypes.items()):
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        unique_vals = df[col].nunique()
        print(f"{i+1:2d}. {col:<35} | {str(dtype):<12} | Non-null: {non_null:>6,} | Null: {null_count:>6,} | Unique: {unique_vals:>6,}")
    
    # Missing values analysis
    print("\nMISSING VALUES ANALYSIS:")
    print("-" * 40)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percent.values
    })
    missing_df = missing_df[missing_df.Missing_Count > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
        
        # Create missing values heatmap
        plt.figure(figsize=(12, 8))
        missing_matrix = df.isnull()
        ax = sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.xticks(rotation=45, ha='right')
        labels = df.columns
        reshaped_labels = [get_display(arabic_reshaper.reshape(label)) for label in labels]
        ax.set_xticks(ax.get_xticks()) 
        ax.set_xticklabels(reshaped_labels, rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        plt.savefig('analysis/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No missing values found!")
    
    # Duplicate analysis
    print("\nDUPLICATE ANALYSIS:")
    print("-" * 40)
    total_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates:,}")
    
    # Check duplicates by key columns
    if 'الرقم_الضريبي' in df.columns and 'السنة' in df.columns:
        key_duplicates = df.duplicated(subset=['الرقم_الضريبي', 'السنة']).sum()
        print(f"Duplicate company-year combinations: {key_duplicates:,}")
    
    # Cardinality analysis
    print("\nCARDINALITY ANALYSIS (Top categorical columns):")
    print("-" * 40)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:5]:  # Top 5 categorical columns
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count:,} unique values")
        if unique_count <= 20:  # Show top values for manageable categories
            top_values = df[col].value_counts().head(10)
            print("  Top values:")
            for val, count in top_values.items():
                print(f"    {val}: {count:,} ({count/len(df)*100:.1f}%)")
        print()
    
    # Data range validation
    print("DATA RANGE VALIDATION:")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    range_checks = {
        'السنة': (2020, 2025),
        'الموظفون': (0, 1000),
        'نمو_المبيعات': (-1, 50),
        'نمو_الموظفين': (-1, 20),
        'score': (0, 10)
    }
    
    for col, (min_val, max_val) in range_checks.items():
        if col in numeric_cols:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)][col].count()
            print(f"{col}: {out_of_range:,} values outside range [{min_val}, {max_val}]")
    
    # Zero/negative value checks
    print("\nZERO/NEGATIVE VALUE ANALYSIS:")
    print("-" * 40)
    for col in numeric_cols:
        zeros = (df[col] == 0).sum()
        negatives = (df[col] < 0).sum()
        if zeros > 0 or negatives > 0:
            print(f"{col}: {zeros:,} zeros, {negatives:,} negatives")
    
    return df

def univariate_analysis():
    """2. Univariate Analysis (Per-Column Distributions and Statistics)"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("2. UNIVARIATE ANALYSIS")
    print("="*80)
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print("NUMERIC COLUMNS ANALYSIS:")
    print("-" * 60)
    
    # Create comprehensive statistics
    stats_data = []
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            stats_row = {
                'Column': col,
                'Count': len(series),
                'Mean': series.mean(),
                'Median': series.median(),
                'Std': series.std(),
                'Min': series.min(),
                'Max': series.max(),
                'Q25': series.quantile(0.25),
                'Q75': series.quantile(0.75),
                'Skewness': series.skew(),
                'Kurtosis': series.kurtosis(),
                'P95': series.quantile(0.95),
                'P99': series.quantile(0.99)
            }
            stats_data.append(stats_row)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Display key statistics for major columns
    key_financial_cols = ['المبيعات_جنيه', 'الإيرادات_جنيه', 'الموظفون', 'نمو_المبيعات', 
                          'نمو_الموظفين', 'العائد_على_رأس_المال', 'score']
    
    for col in key_financial_cols:
        if col in stats_df['Column'].values:
            row = stats_df[stats_df['Column'] == col].iloc[0]
            print(f"\n{col}:")
            print(f"  Mean: {row['Mean']:,.2f} | Median: {row['Median']:,.2f} | Std: {row['Std']:,.2f}")
            print(f"  Range: [{row['Min']:,.2f}, {row['Max']:,.2f}]")
            print(f"  Q25-Q75: [{row['Q25']:,.2f}, {row['Q75']:,.2f}]")
            print(f"  P95: {row['P95']:,.2f} | P99: {row['P99']:,.2f}")
            print(f"  Skewness: {row['Skewness']:.2f} | Kurtosis: {row['Kurtosis']:.2f}")
    
    # Create distribution plots for key variables
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    plot_cols = ['المبيعات_جنيه', 'الموظفون', 'نمو_المبيعات', 'العائد_على_رأس_المال', 'score', 'عمر_المنشأة']
    
    for i, col in enumerate(plot_cols):
        if col in df.columns and i < len(axes):
            # Handle potential outliers by using log scale for highly skewed data
            data = df[col].dropna()
            reshaped_col = get_display(arabic_reshaper.reshape(col))
            
            if col in ['المبيعات_جنيه', 'الإيرادات_جنيه']:
                # Use log scale for financial data
                log_data = np.log1p(data[data > 0])
                axes[i].hist(log_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(get_display(arabic_reshaper.reshape(f'Log Distribution: {col}')))
                axes[i].set_xlabel('Log Scale')
            else:
                axes[i].hist(data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[i].set_title(get_display(arabic_reshaper.reshape(f'Distribution: {col}')))
            
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/univariate_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Categorical analysis
    print("\nCATEGORICAL COLUMNS ANALYSIS:")
    print("-" * 60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"\n{col} (Unique values: {unique_count:,}):")
        
        if unique_count <= 50:  # Show frequency for manageable categories
            freq = df[col].value_counts().head(10)
            total = len(df[col].dropna())
            
            for value, count in freq.items():
                percentage = (count / total) * 100
                print(f"  {value}: {count:,} ({percentage:.1f}%)")
        
        # Create bar plot for top categories
        if unique_count <= 20 and unique_count > 1:
            plt.figure(figsize=(12, 6))
            freq_data = df[col].value_counts().head(15)
            ax = freq_data.plot(kind='bar', color='coral')
            reshaped_col = get_display(arabic_reshaper.reshape(col))
            plt.title(f'Frequency Distribution: {reshaped_col}')
            plt.xlabel(reshaped_col)
            plt.ylabel('Count')
            
            reshaped_labels = [get_display(arabic_reshaper.reshape(str(label.get_text()))) for label in ax.get_xticklabels()]
            ax.set_xticklabels(reshaped_labels, rotation=45, ha='right')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'analysis/categorical_{col.replace("/", "_")}_freq.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    return stats_df

def bivariate_analysis():
    """3. Bivariate Analysis (Pairwise Relationships)"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("3. BIVARIATE ANALYSIS")
    print("="*80)
    
    # Numeric-numeric correlations
    print("NUMERIC-NUMERIC CORRELATIONS:")
    print("-" * 40)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    financial_cols = ['المبيعات_جنيه', 'الإيرادات_جنيه', 'الموظفون', 'رأس_المال_المدفوع_جنيه',
                      'نمو_المبيعات', 'نمو_الموظفين', 'العائد_على_رأس_المال', 'score', 'عمر_المنشأة']
    
    # Filter to existing columns
    available_financial = [col for col in financial_cols if col in numeric_cols]
    
    # Calculate correlation matrix
    corr_matrix = df[available_financial].corr()
    
    # Display top correlations
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_values = corr_matrix.mask(mask).stack().sort_values(key=abs, ascending=False)
    
    print("Top 10 correlations:")
    for (var1, var2), corr_val in corr_values.head(10).items():
        print(f"  {var1} ↔ {var2}: {corr_val:.3f}")
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    reshaped_labels = [get_display(arabic_reshaper.reshape(col)) for col in corr_matrix.columns]
    ax = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                     square=True, fmt='.2f', cbar_kws={"shrink": .8}, 
                     xticklabels=reshaped_labels, yticklabels=reshaped_labels)
    plt.title('Correlation Matrix: Financial Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Key scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sales vs Employees
    if 'المبيعات_جنيه' in df.columns and 'الموظفون' in df.columns:
        axes[0,0].scatter(df['الموظفون'], df['المبيعات_جنيه'], alpha=0.5, s=20)
        axes[0,0].set_xlabel(get_display(arabic_reshaper.reshape('Employees (الموظفون)')))
        axes[0,0].set_ylabel(get_display(arabic_reshaper.reshape('Sales (المبيعات_جنيه)')))
        axes[0,0].set_title('Sales vs Employees')
        axes[0,0].set_yscale('log')
        axes[0,0].grid(True, alpha=0.3)
    
    # Growth correlations
    if 'نمو_المبيعات' in df.columns and 'نمو_الموظفين' in df.columns:
        growth_data = df[['نمو_المبيعات', 'نمو_الموظفين']].dropna()
        # Filter extreme outliers for better visualization
        growth_filtered = growth_data[(growth_data['نمو_المبيعات'].between(-1, 5)) & 
                                      (growth_data['نمو_الموظفين'].between(-1, 3))]
        
        axes[0,1].scatter(growth_filtered['نمو_المبيعات'], growth_filtered['نمو_الموظفين'], 
                          alpha=0.5, s=20, color='green')
        axes[0,1].set_xlabel(get_display(arabic_reshaper.reshape('Sales Growth (نمو_المبيعات)')))
        axes[0,1].set_ylabel(get_display(arabic_reshaper.reshape('Employee Growth (نمو_الموظفين)')))
        axes[0,1].set_title('Sales Growth vs Employee Growth')
        axes[0,1].grid(True, alpha=0.3)
    
    # Score vs ROI
    if 'score' in df.columns and 'العائد_على_رأس_المال' in df.columns:
        roi_filtered = df[df['العائد_على_رأس_المال'] < 200]  # Filter extreme outliers
        axes[1,0].scatter(roi_filtered['score'], roi_filtered['العائد_على_رأس_المال'], 
                          alpha=0.5, s=20, color='purple')
        axes[1,0].set_xlabel('Score')
        axes[1,0].set_ylabel(get_display(arabic_reshaper.reshape('Return on Capital (العائد_على_رأس_المال)')))
        axes[1,0].set_title('Score vs Return on Capital')
        axes[1,0].grid(True, alpha=0.3)
    
    # Age vs Sales
    if 'عمر_المنشأة' in df.columns and 'المبيعات_جنيه' in df.columns:
        axes[1,1].scatter(df['عمر_المنشأة'], df['المبيعات_جنيه'], alpha=0.5, s=20, color='orange')
        axes[1,1].set_xlabel(get_display(arabic_reshaper.reshape('Company Age (عمر_المنشأة)')))
        axes[1,1].set_ylabel(get_display(arabic_reshaper.reshape('Sales (المبيعات_جنيه)')))
        axes[1,1].set_title('Company Age vs Sales')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/bivariate_scatterplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Numeric-categorical analysis
    print("\nNUMERIC-CATEGORICAL ANALYSIS:")
    print("-" * 40)
    
    # Score by Sector analysis
    if 'القطاع' in df.columns and 'score' in df.columns:
        sector_scores = df.groupby('القطاع')['score'].agg(['mean', 'median', 'std', 'count']).round(2)
        sector_scores = sector_scores.sort_values('mean', ascending=False)
        
        print("Score statistics by Sector (Top 10):")
        print(sector_scores.head(10).to_string())
        
        # Boxplot of scores by sector (top sectors only)
        top_sectors = sector_scores.head(8).index
        sector_data = df[df['القطاع'].isin(top_sectors)]
        
        plt.figure(figsize=(15, 8))
        ax = sns.boxplot(data=sector_data, x='القطاع', y='score')
        reshaped_labels = [get_display(arabic_reshaper.reshape(label.get_text())) for label in ax.get_xticklabels()]
        ax.set_xticklabels(reshaped_labels, rotation=45, ha='right')
        plt.title('Score Distribution by Sector (Top 8 Sectors)')
        plt.xlabel(get_display(arabic_reshaper.reshape('القطاع')))
        plt.tight_layout()
        plt.savefig('analysis/score_by_sector_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Sales by SME category
    if 'فئة_SME' in df.columns and 'المبيعات_جنيه' in df.columns:
        sme_sales = df.groupby('فئة_SME')['المبيعات_جنيه'].agg(['mean', 'median', 'count'])
        print(f"\nSales by SME Category:")
        print(sme_sales.to_string())
        
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=df, x='فئة_SME', y='المبيعات_جنيه')
        reshaped_labels = [get_display(arabic_reshaper.reshape(label.get_text())) for label in ax.get_xticklabels()]
        ax.set_xticklabels(reshaped_labels)
        plt.xlabel(get_display(arabic_reshaper.reshape('فئة_SME')))
        plt.ylabel(get_display(arabic_reshaper.reshape('المبيعات_جنيه')))
        plt.yscale('log')
        plt.title('Sales Distribution by SME Category')
        plt.savefig('analysis/sales_by_sme_category.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return corr_matrix

def multivariate_analysis():
    """4. Multivariate Analysis (3+ Variables)"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("4. MULTIVARIATE ANALYSIS")
    print("="*80)
    
    # Select numeric columns for multivariate analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    key_vars = ['المبيعات_جنيه', 'الموظفون', 'العائد_على_rأس_المال', 'score', 'عمر_المنشأة']
    available_vars = [col for col in key_vars if col in numeric_cols]
    
    if len(available_vars) >= 3:
        # Prepare data for dimensionality reduction
        analysis_data = df[available_vars].dropna()
        
        # Remove extreme outliers using IQR method
        Q1 = analysis_data.quantile(0.25)
        Q3 = analysis_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter outliers
        mask = ~((analysis_data < lower_bound) | (analysis_data > upper_bound)).any(axis=1)
        clean_data = analysis_data[mask]
        
        print(f"Data prepared for analysis: {len(clean_data):,} observations")
        print(f"Variables included: {available_vars}")
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)
        
        # PCA Analysis
        print("\nPCA ANALYSIS:")
        print("-" * 40)
        
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print("Principal Components Explained Variance:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            print(f"  PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)")
        
        # Plot PCA results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance plot
        ax1.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, color='steelblue')
        ax1.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'ro-', color='red')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance')
        ax1.grid(True, alpha=0.3)
        
        # 2D PCA scatter plot
        ax2.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=20)
        ax2.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        ax2.set_title('2D PCA Projection')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis/pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance in PCA
        print("\nPCA Component Loadings (Top 2 Components):")
        feature_names = available_vars
        for i in range(min(2, len(pca.components_))):
            print(f"\nPC{i+1} loadings:")
            loadings = pca.components_[i]
            for feature, loading in zip(feature_names, loadings):
                print(f"  {feature}: {loading:.3f}")
        
        # Grouped multivariate analysis
        print("\nGROUPED MULTIVARIATE ANALYSIS:")
        print("-" * 40)
        
        if 'القطاع' in df.columns:
            # Get top sectors by frequency
            top_sectors = df['القطاع'].value_counts().head(5).index
            
            # Pivot table: Mean values by sector and year
            if 'السنة' in df.columns:
                pivot_data = df[df['القطاع'].isin(top_sectors)]
                pivot_table = pivot_data.groupby(['القطاع', 'السنة'])['score'].mean().unstack(fill_value=0)
                
                print("Average Score by Sector and Year (Top 5 Sectors):")
                print(pivot_table.round(2).to_string())
                
                # Heatmap of sector performance over years
                plt.figure(figsize=(12, 8))
                reshaped_yticklabels = [get_display(arabic_reshaper.reshape(label)) for label in pivot_table.index]
                ax = sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlBu_r', yticklabels=reshaped_yticklabels)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                plt.title('Average Score by Sector and Year')
                plt.ylabel(get_display(arabic_reshaper.reshape('Sector (القطاع)')))
                plt.xlabel(get_display(arabic_reshaper.reshape('Year (السنة)')))
                plt.tight_layout()
                plt.savefig('analysis/sector_year_heatmap.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Three-way interaction analysis
        if 'فئة_SME' in df.columns and 'القطاع' in df.columns:
            interaction_data = df.groupby(['فئة_SME', 'القطاع', 'السنة']).agg({
                'المبيعات_جنيه': 'mean',
                'score': 'mean',
                'الموظفون': 'mean'
            }).round(2)
            
            print("\nThree-way Analysis Sample (SME × Sector × Year):")
            print(interaction_data.head(10).to_string())
    
    return pca_result if 'pca_result' in locals() else None

def timeseries_trend_analysis():
    """5. Time-Series and Trend Analysis"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("5. TIME-SERIES AND TREND ANALYSIS")
    print("="*80)
    
    if 'السنة' not in df.columns:
        print("No year column found for time series analysis")
        return
    
    # Yearly aggregation trends
    print("YEARLY TRENDS:")
    print("-" * 40)
    
    yearly_stats = df.groupby('السنة').agg({
        'المبيعات_جنيه': ['sum', 'mean', 'median', 'count'],
        'الموظفون': ['sum', 'mean'],
        'score': ['mean', 'median'],
        'نمو_المبيعات': ['mean', 'median'],
        'نمو_الموظفين': ['mean', 'median']
    }).round(2)
    
    yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns]
    print("Yearly Statistics Summary:")
    print(yearly_stats.to_string())
    
    # Create comprehensive time series plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total sales trend
    yearly_sales = df.groupby('السنة')['المبيعات_جنيه'].sum() / 1e9  # Convert to billions
    axes[0,0].plot(yearly_sales.index, yearly_sales.values, marker='o', linewidth=2, markersize=8)
    axes[0,0].set_title('Total Sales Trend (Billions EGP)')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Sales (Billions EGP)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Average score trend
    yearly_score = df.groupby('السنة')['score'].mean()
    axes[0,1].plot(yearly_score.index, yearly_score.values, marker='s', linewidth=2, 
                   markersize=8, color='green')
    axes[0,1].set_title('Average Score Trend')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Average Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # Company count trend
    yearly_companies = df.groupby('السنة').size()
    axes[1,0].bar(yearly_companies.index, yearly_companies.values, alpha=0.7, color='coral')
    axes[1,0].set_title('Number of Companies by Year')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Company Count')
    axes[1,0].grid(True, alpha=0.3)
    
    # Growth trends
    if 'نمو_المبيعات' in df.columns:
        growth_trend = df.groupby('السنة')['نمو_المبيعات'].median()
        axes[1,1].plot(growth_trend.index, growth_trend.values, marker='^', linewidth=2,
                       markersize=8, color='purple')
        axes[1,1].set_title('Median Sales Growth Trend')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Median Sales Growth')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('analysis/timeseries_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Cohort analysis by start year
    if 'start_year' in df.columns:
        print("\nCOHORT ANALYSIS:")
        print("-" * 40)
        
        # Analyze performance by company start year
        cohort_analysis = df.groupby(['start_year', 'السنة']).agg({
            'score': 'mean',
            'المبيعات_جنيه': 'mean',
            'عمر_المنشأة': 'mean'
        }).round(2)
        
        # Create cohort heatmap for scores
        cohort_pivot = df.groupby(['start_year', 'السنة'])['score'].mean().unstack(fill_value=0)
        
        if len(cohort_pivot) > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(cohort_pivot, annot=True, fmt='.1f', cmap='viridis')
            plt.title('Cohort Analysis: Average Score by Start Year and Current Year')
            plt.xlabel(get_display(arabic_reshaper.reshape('Current Year (السنة)')))
            plt.ylabel('Start Year')
            plt.tight_layout()
            plt.savefig('analysis/cohort_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("Sample Cohort Data (Start Year × Current Year):")
        print(cohort_analysis.head(15).to_string())
    
    # Calculate growth trajectories for individual companies
    if 'الرقم_الضريبي' in df.columns:
        print("\nCOMPANY GROWTH TRAJECTORIES:")
        print("-" * 40)
        
        # Sample companies with multiple years of data
        company_years = df.groupby('الرقم_الضريبي').size()
        multi_year_companies = company_years[company_years >= 3].index[:10]  # Top 10 companies with 3+ years
        
        if len(multi_year_companies) > 0:
            plt.figure(figsize=(12, 8))
            for i, company_id in enumerate(multi_year_companies):
                company_data = df[df['الرقم_الضريبي'] == company_id].sort_values('السنة')
                plt.plot(company_data['السنة'], company_data['score'], marker='o', 
                         alpha=0.7, linewidth=2, label=f'Company {i+1}')
            
            plt.title('Score Trajectories for Sample Companies (3+ years of data)')
            plt.xlabel('Year')
            plt.ylabel('Score')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('analysis/company_trajectories.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    return yearly_stats

def outlier_anomaly_detection():
    """6. Outlier and Anomaly Detection"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("6. OUTLIER AND ANOMALY DETECTION")
    print("="*80)
    
    # Define key numeric columns for outlier analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    key_vars = ['المبيعات_جنيه', 'الموظفون', 'نمو_المبيعات', 'نمو_الموظفين', 
                'العائد_على_رأس_المال', 'score']
    available_vars = [col for col in key_vars if col in numeric_cols]
    
    outlier_summary = {}
    
    print("UNIVARIATE OUTLIER DETECTION:")
    print("-" * 40)
    
    for col in available_vars:
        series = df[col].dropna()
        
        if len(series) > 0:
            # Z-score method (>3 or <-3)
            z_scores = np.abs(stats.zscore(series))
            z_outliers = np.sum(z_scores > 3)
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = np.sum((series < lower_bound) | (series > upper_bound))
            
            outlier_summary[col] = {
                'z_score_outliers': z_outliers,
                'iqr_outliers': iqr_outliers,
                'total_values': len(series)
            }
            
            print(f"{col}:")
            print(f"  Z-score outliers (>3σ): {z_outliers:,} ({z_outliers/len(series)*100:.1f}%)")
            print(f"  IQR outliers: {iqr_outliers:,} ({iqr_outliers/len(series)*100:.1f}%)")
            print(f"  Range: [{series.min():,.2f}, {series.max():,.2f}]")
            print()
    
    # Create outlier visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(available_vars[:6]):
        if col in df.columns:
            # Box plot to show outliers
            data = df[col].dropna()
            reshaped_col = get_display(arabic_reshaper.reshape(col))
            
            # Handle extreme values for better visualization
            if col in ['المبيعات_جنيه', 'الإيرادات_جنيه']:
                # Use log scale for highly skewed financial data
                log_data = np.log1p(data[data > 0])
                axes[i].boxplot(log_data, vert=True)
                axes[i].set_title(get_display(arabic_reshaper.reshape(f'Outliers (Log Scale): {col}')))
                axes[i].set_ylabel('Log Scale')
            else:
                # Filter extreme outliers for better visualization (keep 99.5%)
                lower_percentile = data.quantile(0.002)
                upper_percentile = data.quantile(0.998)
                filtered_data = data[(data >= lower_percentile) & (data <= upper_percentile)]
                
                axes[i].boxplot(filtered_data, vert=True)
                axes[i].set_title(get_display(arabic_reshaper.reshape(f'Outliers: {col}')))
                axes[i].set_ylabel(reshaped_col)
            
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Multivariate outlier detection
    print("MULTIVARIATE OUTLIER DETECTION:")
    print("-" * 40)
    
    # Select variables for multivariate analysis
    mv_vars = ['المبيعات_جنيه', 'الموظفون', 'score']
    mv_available = [col for col in mv_vars if col in df.columns]
    
    if len(mv_available) >= 2:
        mv_data = df[mv_available].dropna()
        
        # Remove extreme univariate outliers first
        for col in mv_available:
            Q1 = mv_data[col].quantile(0.01)
            Q3 = mv_data[col].quantile(0.99)
            mv_data = mv_data[(mv_data[col] >= Q1) & (mv_data[col] <= Q3)]
        
        print(f"Data for multivariate analysis: {len(mv_data):,} observations")
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(mv_data)
        
        # Calculate Mahalanobis distance
        cov_matrix = np.cov(scaled_data.T)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        
        mean_vector = np.mean(scaled_data, axis=0)
        mahal_distances = []
        
        for i in range(len(scaled_data)):
            diff = scaled_data[i] - mean_vector
            mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
            mahal_distances.append(mahal_dist)
        
        mahal_distances = np.array(mahal_distances)
        
        # Identify multivariate outliers (top 5%)
        mahal_threshold = np.percentile(mahal_distances, 95)
        mv_outliers = np.sum(mahal_distances > mahal_threshold)
        
        print(f"Multivariate outliers (top 5%): {mv_outliers:,}")
        print(f"Mahalanobis distance threshold: {mahal_threshold:.2f}")
        
        # Scatter plot with outliers highlighted
        if len(mv_available) >= 2:
            plt.figure(figsize=(10, 8))
            
            # Normal points
            normal_mask = mahal_distances <= mahal_threshold
            plt.scatter(mv_data.iloc[:, 0][normal_mask], mv_data.iloc[:, 1][normal_mask], 
                        alpha=0.6, s=20, label='Normal', color='blue')
            
            # Outlier points
            outlier_mask = mahal_distances > mahal_threshold
            plt.scatter(mv_data.iloc[:, 0][outlier_mask], mv_data.iloc[:, 1][outlier_mask], 
                        alpha=0.8, s=50, label='Outliers', color='red', marker='x')
            
            plt.xlabel(get_display(arabic_reshaper.reshape(mv_available[0])))
            plt.ylabel(get_display(arabic_reshaper.reshape(mv_available[1])))
            plt.title('Multivariate Outliers (Mahalanobis Distance)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Use log scale if needed
            if mv_available[0] in ['المبيعات_جنيه', 'الإيرادات_جنيه']:
                plt.xscale('log')
            if len(mv_available) > 1 and mv_available[1] in ['المبيعات_جنيه', 'الإيرادات_جنيه']:
                plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('analysis/multivariate_outliers.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Contextual outliers (within groups)
    print("CONTEXTUAL OUTLIER DETECTION:")
    print("-" * 40)
    
    if 'القطاع' in df.columns and 'score' in df.columns:
        # Find outliers within each sector
        sector_outliers = {}
        
        for sector in df['القطاع'].value_counts().head(10).index:
            sector_data = df[df['القطاع'] == sector]['score']
            
            if len(sector_data) > 10:  # Minimum sample size
                z_scores = np.abs(stats.zscore(sector_data))
                sector_outliers_count = np.sum(z_scores > 2.5)
                sector_outliers[sector] = {
                    'outliers': sector_outliers_count,
                    'total': len(sector_data),
                    'percentage': (sector_outliers_count / len(sector_data)) * 100
                }
        
        print("Score outliers by sector (top 10 sectors):")
        for sector, data in sector_outliers.items():
            print(f"  {sector[:50]}: {data['outliers']} outliers ({data['percentage']:.1f}%) out of {data['total']}")
    
    return outlier_summary

def grouping_aggregation_analysis():
    """7. Grouping and Aggregation Analysis"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("7. GROUPING AND AGGREGATION ANALYSIS")
    print("="*80)
    
    # Analysis by sector
    print("ANALYSIS BY SECTOR:")
    print("-" * 40)
    
    if 'القطاع' in df.columns:
        sector_analysis = df.groupby('القطاع').agg({
            'المبيعات_جنيه': ['sum', 'mean', 'median', 'count'],
            'الموظفون': ['sum', 'mean', 'median'],
            'score': ['mean', 'median', 'std'],
            'نمو_المبيعات': ['mean', 'median'],
            'العائد_على_رأس_المال': ['mean', 'median']
        }).round(2)
        
        sector_analysis.columns = ['_'.join(col).strip() for col in sector_analysis.columns]
        sector_sorted = sector_analysis.sort_values('المبيعات_جنيه_sum', ascending=False)
        
        print("Top 10 Sectors by Total Sales:")
        print(sector_sorted[['المبيعات_جنيه_sum', 'المبيعات_جنيه_mean', 
                               'score_mean', 'الموظفون_sum']].head(10).to_string())
        
        # Create sector performance visualization
        top_10_sectors = sector_sorted.head(10)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        reshaped_labels = [get_display(arabic_reshaper.reshape(s[:20] + '...' if len(s) > 20 else s)) for s in top_10_sectors.index]

        # Total sales by sector
        ax1.bar(range(len(top_10_sectors)), top_10_sectors['المبيعات_جنيه_sum']/1e9, 
                color='skyblue', alpha=0.7)
        ax1.set_title('Total Sales by Sector (Top 10, Billions EGP)')
        ax1.set_ylabel('Sales (Billions EGP)')
        ax1.set_xticks(range(len(top_10_sectors)))
        ax1.set_xticklabels(reshaped_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Average score by sector
        ax2.bar(range(len(top_10_sectors)), top_10_sectors['score_mean'], 
                color='lightgreen', alpha=0.7)
        ax2.set_title('Average Score by Sector (Top 10)')
        ax2.set_ylabel('Average Score')
        ax2.set_xticks(range(len(top_10_sectors)))
        ax2.set_xticklabels(reshaped_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Company count by sector
        ax3.bar(range(len(top_10_sectors)), top_10_sectors['المبيعات_جنيه_count'], 
                color='coral', alpha=0.7)
        ax3.set_title('Company Count by Sector (Top 10)')
        ax3.set_ylabel('Number of Companies')
        ax3.set_xticks(range(len(top_10_sectors)))
        ax3.set_xticklabels(reshaped_labels, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Average employees by sector
        ax4.bar(range(len(top_10_sectors)), top_10_sectors['الموظفون_mean'], 
                color='gold', alpha=0.7)
        ax4.set_title('Average Employees by Sector (Top 10)')
        ax4.set_ylabel('Average Employees')
        ax4.set_xticks(range(len(top_10_sectors)))
        ax4.set_xticklabels(reshaped_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis/sector_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Analysis by SME category
    print("\nANALYSIS BY SME CATEGORY:")
    print("-" * 40)
    
    if 'فئة_SME' in df.columns:
        sme_analysis = df.groupby('فئة_SME').agg({
            'المبيعات_جنيه': ['mean', 'median', 'std', 'count'],
            'الموظفون': ['mean', 'median'],
            'score': ['mean', 'median', 'std'],
            'نمو_المبيعات': ['mean', 'median'],
            'العائد_على_رأس_المال': ['mean', 'median']
        }).round(2)
        
        sme_analysis.columns = ['_'.join(col).strip() for col in sme_analysis.columns]
        
        print("SME Category Analysis:")
        print(sme_analysis.to_string())
        
        # SME category comparison plot
        plt.figure(figsize=(14, 8))
        
        categories = sme_analysis.index
        reshaped_categories = [get_display(arabic_reshaper.reshape(cat)) for cat in categories]
        x = np.arange(len(categories))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sales comparison
        ax1.bar(x - width/2, sme_analysis['المبيعات_جنيه_mean']/1e6, width, 
                label='Mean Sales', alpha=0.7, color='steelblue')
        ax1.bar(x + width/2, sme_analysis['المبيعات_جنيه_median']/1e6, width, 
                label='Median Sales', alpha=0.7, color='lightcoral')
        ax1.set_title('Sales Comparison by SME Category (Millions EGP)')
        ax1.set_xlabel('SME Category')
        ax1.set_ylabel('Sales (Millions EGP)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(reshaped_categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Score comparison
        ax2.bar(x - width/2, sme_analysis['score_mean'], width, 
                label='Mean Score', alpha=0.7, color='green')
        ax2.bar(x + width/2, sme_analysis['score_median'], width, 
                label='Median Score', alpha=0.7, color='orange')
        ax2.set_title('Score Comparison by SME Category')
        ax2.set_xlabel('SME Category')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(reshaped_categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis/sme_category_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Top and bottom performers
    print("\nTOP AND BOTTOM PERFORMERS:")
    print("-" * 40)
    
    if 'score' in df.columns and 'اسم_الشركة' in df.columns:
        # Top performers by score
        top_companies = df.nlargest(10, 'score')[['اسم_الشركة', 'القطاع', 'فئة_SME', 'score', 'المبيعات_جنيه']]
        print("Top 10 Companies by Score:")
        print(top_companies.to_string(index=False))
        
        # Bottom performers by score
        bottom_companies = df.nsmallest(10, 'score')[['اسم_الشركة', 'القطاع', 'فئة_SME', 'score', 'المبيعات_جنيه']]
        print("\nBottom 10 Companies by Score:")
        print(bottom_companies.to_string(index=False))
    
    # Company lifetime analysis
    if 'الرقم_الضريبي' in df.columns:
        print("\nCOMPANY LIFETIME ANALYSIS:")
        print("-" * 40)
        
        company_lifetime = df.groupby('الرقم_الضريبي').agg({
            'score': ['mean', 'std', 'count'],
            'المبيعات_جنيه': ['mean', 'sum'],
            'السنة': ['min', 'max'],
            'اسم_الشركة': 'first',
            'القطاع': 'first'
        }).round(2)
        
        company_lifetime.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                    for col in company_lifetime.columns]
        
        # Companies with most years of data
        most_years = company_lifetime.sort_values('score_count', ascending=False).head(10)
        print("Companies with Most Years of Data:")
        # FIX: Use the new column names 'اسم_الشركة_first' and 'القطاع_first'
        print(most_years[['اسم_الشركة_first', 'القطاع_first', 'score_count', 'score_mean']].to_string())
        
        # Best average performers (companies with 3+ years)
        consistent_performers = company_lifetime[company_lifetime['score_count'] >= 3]
        top_consistent = consistent_performers.sort_values('score_mean', ascending=False).head(10)
        print("\nTop Consistent Performers (3+ years):")
        print(top_consistent[['اسم_الشركة_first', 'القطاع_first', 'score_mean', 'score_count']].to_string())
    
    return sector_analysis if 'sector_analysis' in locals() else None

def derived_metrics_analysis():
    """8. Derived Metrics and Feature Engineering Insights"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("8. DERIVED METRICS AND FEATURE ENGINEERING")
    print("="*80)
    
    # Create derived metrics
    print("CREATING DERIVED METRICS:")
    print("-" * 40)
    
    # Sales per employee ratio
    if 'المبيعات_جنيه' in df.columns and 'الموظفون' in df.columns:
        df['sales_per_employee'] = df['المبيعات_جنيه'] / df['الموظفون']
        print("✓ Created: Sales per Employee (sales_per_employee)")
    
    # Capital efficiency ratio
    if 'المبيعات_جنيه' in df.columns and 'رأس_المال_المدفوع_جنيه' in df.columns:
        df['capital_turnover'] = df['المبيعات_جنيه'] / df['رأس_المال_المدفوع_جنيه']
        print("✓ Created: Capital Turnover (capital_turnover)")
    
    # VAT compliance ratio
    if 'ضريبة_القيمة_المضافة_المصرح_بها' in df.columns and 'ضريبة_القيمة_المضافة_المتوقعة' in df.columns:
        df['vat_compliance_ratio'] = df['ضريبة_القيمة_المضافة_المصرح_بها'] / df['ضريبة_القيمة_المضافة_المتوقعة']
        print("✓ Created: VAT Compliance Ratio (vat_compliance_ratio)")
    
    # VAT discrepancy
    if 'ضريبة_القيمة_المضافة_المصرح_بها' in df.columns and 'ضريبة_القيمة_المضافة_المتوقعة' in df.columns:
        df['vat_discrepancy'] = (df['ضريبة_القيمة_المضافة_المتوقعة'] - df['ضريبة_القيمة_المضافة_المصرح_بها']) / df['ضريبة_القيمة_المضافة_المتوقعة']
        print("✓ Created: VAT Discrepancy (vat_discrepancy)")
    
    # Company age categories
    if 'عمر_المنشأة' in df.columns:
        df['age_category'] = pd.cut(df['عمر_المنشأة'], 
                                    bins=[0, 5, 10, 15, 100], 
                                    labels=['Young (0-5)', 'Growing (6-10)', 'Mature (11-15)', 'Established (15+)'])
        print("✓ Created: Age Categories (age_category)")
    
    # Size by employees
    if 'الموظفون' in df.columns:
        df['employee_size_category'] = pd.cut(df['الموظفون'], 
                                              bins=[0, 10, 50, 100, 1000], 
                                              labels=['Micro (1-10)', 'Small (11-50)', 'Medium (51-100)', 'Large (100+)'])
        print("✓ Created: Employee Size Categories (employee_size_category)")
    
    # Growth categories
    if 'نمو_المبيعات' in df.columns:
        df['growth_category'] = pd.cut(df['نمو_المبيعات'], 
                                       bins=[-2, -0.1, 0.1, 0.5, 100], 
                                       labels=['Declining', 'Stable', 'Growing', 'High Growth'])
        print("✓ Created: Growth Categories (growth_category)")
    
    print()
    
    # Analyze derived metrics
    derived_cols = ['sales_per_employee', 'capital_turnover', 'vat_compliance_ratio', 'vat_discrepancy']
    available_derived = [col for col in derived_cols if col in df.columns]
    
    print("DERIVED METRICS STATISTICS:")
    print("-" * 40)
    
    for col in available_derived:
        series = df[col].dropna()
        
        if len(series) > 0:
            # Remove extreme outliers for statistics
            Q1 = series.quantile(0.01)
            Q3 = series.quantile(0.99)
            clean_series = series[(series >= Q1) & (series <= Q3)]
            
            print(f"{col}:")
            print(f"  Count: {len(series):,}")
            print(f"  Mean: {clean_series.mean():,.2f}")
            print(f"  Median: {clean_series.median():,.2f}")
            print(f"  Std: {clean_series.std():,.2f}")
            print(f"  Range: [{series.min():,.2f}, {series.max():,.2f}]")
            print(f"  25th-75th: [{series.quantile(0.25):,.2f}, {series.quantile(0.75):,.2f}]")
            print()
    
    # Visualize derived metrics
    if len(available_derived) > 0:
        n_cols = min(len(available_derived), 4)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(available_derived[:4]):
            series = df[col].dropna()
            
            # Remove extreme outliers for visualization
            Q1 = series.quantile(0.005)
            Q3 = series.quantile(0.995)
            clean_series = series[(series >= Q1) & (series <= Q3)]
            
            if col == 'sales_per_employee':
                # Use log scale for sales per employee
                log_series = np.log1p(clean_series[clean_series > 0])
                axes[i].hist(log_series, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution: {col} (Log Scale)')
                axes[i].set_xlabel('Log Scale')
            else:
                axes[i].hist(clean_series, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[i].set_title(f'Distribution: {col}')
                axes[i].set_xlabel(col)
            
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis/derived_metrics_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Correlation analysis with derived metrics
    print("CORRELATION WITH DERIVED METRICS:")
    print("-" * 40)
    
    if 'sales_per_employee' in df.columns and 'score' in df.columns:
        # Remove outliers for correlation
        correlation_data = df[['sales_per_employee', 'score', 'capital_turnover']].dropna()
        
        # Filter extreme outliers
        for col in correlation_data.columns:
            Q1 = correlation_data[col].quantile(0.01)
            Q3 = correlation_data[col].quantile(0.99)
            correlation_data = correlation_data[(correlation_data[col] >= Q1) & (correlation_data[col] <= Q3)]
        
        if len(correlation_data) > 100:
            corr_with_derived = correlation_data.corr()
            
            print("Correlations involving derived metrics:")
            for i, col1 in enumerate(corr_with_derived.columns):
                for j, col2 in enumerate(corr_with_derived.columns):
                    if i < j:
                        corr_val = corr_with_derived.loc[col1, col2]
                        if abs(corr_val) > 0.1:  # Show meaningful correlations
                            print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
    
    # Category analysis
    print("\nCATEGORY ANALYSIS:")
    print("-" * 40)
    
    categorical_derived = ['age_category', 'employee_size_category', 'growth_category']
    available_categories = [col for col in categorical_derived if col in df.columns]
    
    for col in available_categories:
        print(f"\n{col} distribution:")
        freq = df[col].value_counts()
        total = len(df[col].dropna())
        
        for category, count in freq.items():
            percentage = (count / total) * 100
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        # Mean score by category
        if 'score' in df.columns:
            category_scores = df.groupby(col)['score'].agg(['mean', 'count']).round(2)
            print(f"  Average score by {col}:")
            for category in category_scores.index:
                mean_score = category_scores.loc[category, 'mean']
                count = category_scores.loc[category, 'count']
                print(f"    {category}: {mean_score:.2f} (n={count:,})")
    
    return df[available_derived + available_categories] if len(available_derived + available_categories) > 0 else None

def data_quality_bias_checks():
    """9. Data Quality and Bias Checks"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("9. DATA QUALITY AND BIAS CHECKS")
    print("="*80)
    
    # Imbalance detection
    print("IMBALANCE DETECTION:")
    print("-" * 40)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols[:5]:  # Check top 5 categorical columns
        freq = df[col].value_counts()
        total = len(df[col].dropna())
        
        # Calculate concentration (top value percentage)
        top_percentage = (freq.iloc[0] / total) * 100 if len(freq) > 0 else 0
        
        print(f"{col}:")
        print(f"  Unique values: {len(freq):,}")
        print(f"  Top value concentration: {top_percentage:.1f}%")
        
        if top_percentage > 50:
            print(f"  ⚠️  HIGH IMBALANCE: Top category represents {top_percentage:.1f}% of data")
        
        # Show top categories
        print(f"  Top 5 categories:")
        for i, (value, count) in enumerate(freq.head(5).items()):
            percentage = (count / total) * 100
            print(f"    {i+1}. {str(value)[:30]}: {count:,} ({percentage:.1f}%)")
        print()
    
    # Temporal bias check
    print("TEMPORAL BIAS ANALYSIS:")
    print("-" * 40)
    
    if 'السنة' in df.columns:
        yearly_counts = df['السنة'].value_counts().sort_index()
        print("Data distribution by year:")
        
        total_records = len(df)
        for year, count in yearly_counts.items():
            percentage = (count / total_records) * 100
            print(f"  {year}: {count:,} records ({percentage:.1f}%)")
        
        # Check if there's significant imbalance
        min_year_count = yearly_counts.min()
        max_year_count = yearly_counts.max()
        imbalance_ratio = max_year_count / min_year_count if min_year_count > 0 else float('inf')
        
        if imbalance_ratio > 2:
            print(f"  ⚠️  TEMPORAL IMBALANCE: Ratio of max to min year: {imbalance_ratio:.1f}")
        else:
            print(f"  ✓ Temporal distribution appears balanced (ratio: {imbalance_ratio:.1f})")
        
        # Visualize yearly distribution
        plt.figure(figsize=(10, 6))
        yearly_counts.plot(kind='bar', color='steelblue', alpha=0.7)
        plt.title('Data Distribution by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analysis/temporal_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Missing data correlation analysis
    print("\nMISSING DATA CORRELATION:")
    print("-" * 40)
    
    missing_data = df.isnull()
    
    if missing_data.sum().sum() > 0:
        # Create missing data correlation matrix
        missing_corr = missing_data.corr()
        
        # Find columns with missing data
        cols_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]
        
        print(f"Columns with missing data: {len(cols_with_missing)}")
        for col in cols_with_missing:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count:,} ({missing_percent:.1f}%)")
        
        # Check if missingness correlates with other variables
        if 'score' in df.columns and len(cols_with_missing) > 0:
            print("\nCorrelation between missingness and score:")
            for col in cols_with_missing:
                missing_mask = df[col].isnull()
                if missing_mask.sum() > 0:
                    score_with_missing = df[missing_mask]['score'].mean()
                    score_without_missing = df[~missing_mask]['score'].mean()
                    difference = score_with_missing - score_without_missing
                    
                    print(f"  {col}:")
                    print(f"    Score when missing: {score_with_missing:.2f}")
                    print(f"    Score when present: {score_without_missing:.2f}")
                    print(f"    Difference: {difference:.2f}")
                    
                    if abs(difference) > 0.5:
                        print(f"    ⚠️  SIGNIFICANT BIAS: Missing data correlates with score")
    else:
        print("No missing data found for correlation analysis")
    
    # Consistency checks
    print("\nCONSISTENCY CHECKS:")
    print("-" * 40)
    
    consistency_issues = 0
    
    # Check if growth calculations are consistent
    if all(col in df.columns for col in ['نمو_المبيعات', 'المبيعات_جنيه', 'السنة', 'الرقم_الضريبي']):
        print("Checking sales growth consistency...")
        
        # Sample check for consistency
        sample_companies = df['الرقم_الضريبي'].value_counts().head(100).index
        
        inconsistent_count = 0
        for company_id in sample_companies:
            company_data = df[df['الرقم_الضريبي'] == company_id].sort_values('السنة')
            
            if len(company_data) >= 2:
                for i in range(1, len(company_data)):
                    current_row = company_data.iloc[i]
                    previous_row = company_data.iloc[i-1]
                    
                    if pd.notna(current_row['نمو_المبيعات']):
                        calculated_growth = (current_row['المبيعات_جنيه'] - previous_row['المبيعات_جنيه']) / previous_row['المبيعات_جنيه']
                        reported_growth = current_row['نمو_المبيعات']
                        
                        if abs(calculated_growth - reported_growth) > 0.01:  # 1% tolerance
                            inconsistent_count += 1
                            if inconsistent_count <= 5:  # Show first 5 examples
                                print(f"    Inconsistency found: Company {company_id}, Year {current_row['السنة']}")
                                print(f"      Calculated: {calculated_growth:.3f}, Reported: {reported_growth:.3f}")
        
        consistency_issues += inconsistent_count
        print(f"  Sales growth inconsistencies found: {inconsistent_count}")
    
    # Range validation
    print("\nRANGE VALIDATION:")
    print("-" * 40)
    
    range_violations = 0
    
    # Define expected ranges
    expected_ranges = {
        'السنة': (2020, 2025),
        'الموظفون': (1, 1000),
        'score': (0, 10),
        'العائد_على_رأس_المال': (0, 1000),
        'نمو_المبيعات': (-1, 100),
        'نمو_الموظفين': (-1, 20)
    }
    
    for col, (min_val, max_val) in expected_ranges.items():
        if col in df.columns:
            violations = df[(df[col] < min_val) | (df[col] > max_val)][col].count()
            if violations > 0:
                range_violations += violations
                print(f"  {col}: {violations:,} values outside expected range [{min_val}, {max_val}]")
    
    print(f"\nTotal range violations: {range_violations:,}")
    
    # Sector distribution analysis
    if 'القطاع' in df.columns and 'فئة_SME' in df.columns:
        print("\nSECTOR-SIZE DISTRIBUTION ANALYSIS:")
        print("-" * 40)
        
        sector_size_crosstab = pd.crosstab(df['القطاع'], df['فئة_SME'], normalize='index') * 100
        
        print("SME size distribution by sector (percentage):")
        print(sector_size_crosstab.round(1).head(10).to_string())
        
        # Check for sectors with extreme size distributions
        extreme_sectors = []
        for sector in sector_size_crosstab.index:
            size_distribution = sector_size_crosstab.loc[sector]
            max_percentage = size_distribution.max()
            
            if max_percentage > 90:  # More than 90% in one size category
                extreme_sectors.append((sector, max_percentage))
        
        if extreme_sectors:
            print(f"\n⚠️  Sectors with extreme size concentration:")
            for sector, percentage in extreme_sectors[:5]:
                print(f"  {sector[:50]}: {percentage:.1f}% in one size category")
    
    # Summary
    print("\nDATA QUALITY SUMMARY:")
    print("-" * 40)
    print(f"Total records: {len(df):,}")
    print(f"Consistency issues: {consistency_issues:,}")
    print(f"Range violations: {range_violations:,}")
    
    if consistency_issues + range_violations == 0:
        print("✓ Data quality checks passed")
    else:
        print("⚠️  Data quality issues detected - review recommended")
    
    return {
        'consistency_issues': consistency_issues,
        'range_violations': range_violations,
        'missing_data_info': cols_with_missing if 'cols_with_missing' in locals() else []
    }

def visualizations_reporting():
    """10. Visualizations and Reporting"""
    global df_global
    df = df_global.copy()
    
    print("\n" + "="*80)
    print("10. VISUALIZATIONS AND REPORTING")
    print("="*80)
    
    # Create comprehensive dashboard
    print("Creating comprehensive dashboard visualizations...")
    
    # Dashboard Layout: 2x3 subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Sales distribution (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'المبيعات_جنيه' in df.columns:
        sales_data = df['المبيعات_جنيه'][df['المبيعات_جنيه'] > 0]
        log_sales = np.log10(sales_data)
        ax1.hist(log_sales, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(get_display(arabic_reshaper.reshape('Sales Distribution (Log Scale)')))
        ax1.set_xlabel(get_display(arabic_reshaper.reshape('Log10(Sales EGP)')))
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
    
    # 2. Score distribution by SME category
    ax2 = fig.add_subplot(gs[0, 1])
    if 'score' in df.columns and 'فئة_SME' in df.columns:
        sme_categories = df['فئة_SME'].unique()
        reshaped_labels = [get_display(arabic_reshaper.reshape(str(cat))) for cat in sme_categories]
        for i, category in enumerate(sme_categories):
            category_data = df[df['فئة_SME'] == category]['score']
            ax2.hist(category_data, bins=30, alpha=0.6, label=reshaped_labels[i], 
                     color=plt.cm.Set1(i))
        ax2.set_title(get_display(arabic_reshaper.reshape('Score Distribution by SME Category')))
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Yearly trends
    ax3 = fig.add_subplot(gs[0, 2])
    if 'السنة' in df.columns and 'score' in df.columns:
        yearly_avg_score = df.groupby('السنة')['score'].mean()
        ax3.plot(yearly_avg_score.index, yearly_avg_score.values, 
                 marker='o', linewidth=3, markersize=8, color='green')
        ax3.set_title('Average Score Trend by Year')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Average Score')
        ax3.grid(True, alpha=0.3)
    
    # 4. Top sectors by total sales
    ax4 = fig.add_subplot(gs[1, 0])
    if 'القطاع' in df.columns and 'المبيعات_جنيه' in df.columns:
        sector_sales = df.groupby('القطاع')['المبيعات_جنيه'].sum().nlargest(10)
        y_pos = range(len(sector_sales))
        ax4.barh(y_pos, sector_sales.values/1e9, color='coral', alpha=0.7)
        ax4.set_yticks(y_pos)
        reshaped_labels = [get_display(arabic_reshaper.reshape(s[:25] + '...' if len(s) > 25 else s)) for s in sector_sales.index]
        ax4.set_yticklabels(reshaped_labels, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_title('Top 10 Sectors by Total Sales')
        ax4.set_xlabel('Total Sales (Billions EGP)')
        ax4.grid(True, alpha=0.3)
    
    # 5. Sales vs Employees scatter
    ax5 = fig.add_subplot(gs[1, 1])
    if 'المبيعات_جنيه' in df.columns and 'الموظفون' in df.columns:
        # Sample data for better performance
        sample_size = min(5000, len(df))
        sample_df = df.sample(n=sample_size)
        
        scatter = ax5.scatter(sample_df['الموظفون'], sample_df['المبيعات_جنيه'], 
                              alpha=0.6, s=20, c=sample_df['score'], cmap='viridis')
        ax5.set_xlabel(get_display(arabic_reshaper.reshape('Employees (الموظفون)')))
        ax5.set_ylabel(get_display(arabic_reshaper.reshape('Sales (EGP)')))
        ax5.set_title('Sales vs Employees (colored by Score)')
        ax5.set_yscale('log')
        plt.colorbar(scatter, ax=ax5, label='Score')
        ax5.grid(True, alpha=0.3)
    
    # 6. Growth correlation
    ax6 = fig.add_subplot(gs[1, 2])
    if 'نمو_المبيعات' in df.columns and 'نمو_الموظفين' in df.columns:
        growth_data = df[['نمو_المبيعات', 'نمو_الموظفين']].dropna()
        # Filter extreme outliers
        growth_filtered = growth_data[(growth_data['نمو_المبيعات'].between(-0.5, 2)) & 
                                      (growth_data['نمو_الموظفين'].between(-0.5, 1))]
        
        if len(growth_filtered) > 100:
            ax6.scatter(growth_filtered['نمو_المبيعات'], growth_filtered['نمو_الموظفين'], 
                        alpha=0.5, s=15)
            ax6.set_xlabel(get_display(arabic_reshaper.reshape('Sales Growth (نمو_المبيعات)')))
            ax6.set_ylabel(get_display(arabic_reshaper.reshape('Employee Growth (نمو_الموظفين)')))
            ax6.set_title('Sales vs Employee Growth')
            ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax6.grid(True, alpha=0.3)
    
    # 7. Company age distribution
    ax7 = fig.add_subplot(gs[2, 0])
    if 'عمر_المنشأة' in df.columns:
        age_data = df['عمر_المنشأة']
        ax7.hist(age_data, bins=20, alpha=0.7, color='gold', edgecolor='black')
        ax7.set_title(get_display(arabic_reshaper.reshape('Company Age Distribution (عمر_المنشأة)')))
        ax7.set_xlabel('Company Age (Years)')
        ax7.set_ylabel('Frequency')
        ax7.grid(True, alpha=0.3)
    
    # 8. ROI vs Score relationship
    ax8 = fig.add_subplot(gs[2, 1])
    if 'العائد_على_رأس_المال' in df.columns and 'score' in df.columns:
        roi_filtered = df[df['العائد_على_رأس_المال'] < 200]  # Filter extreme values
        sample_roi = roi_filtered.sample(n=min(3000, len(roi_filtered)))
        
        ax8.scatter(sample_roi['score'], sample_roi['العائد_على_رأس_المال'], 
                    alpha=0.6, s=20, color='purple')
        ax8.set_xlabel('Score')
        ax8.set_ylabel(get_display(arabic_reshaper.reshape('Return on Capital (%) (العائد_على_رأس_المال)')))
        ax8.set_title('Score vs Return on Capital')
        ax8.grid(True, alpha=0.3)
    
    # 9. Missing data bar chart
    ax9 = fig.add_subplot(gs[2, 2])
    missing_data = df.isnull().sum().sort_values(ascending=False).head(10)
    if len(missing_data) > 0 and missing_data.max() > 0:
        ax9.bar(range(len(missing_data)), missing_data.values, color='red', alpha=0.7)
        ax9.set_xticks(range(len(missing_data)))
        reshaped_labels = [get_display(arabic_reshaper.reshape(label)) for label in missing_data.index]
        ax9.set_xticklabels(reshaped_labels, rotation=45, ha='right', fontsize=8)
        ax9.set_title('Missing Data by Column (Top 10)')
        ax9.set_ylabel('Missing Count')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                 transform=ax9.transAxes, fontsize=14)
        ax9.set_title('Data Completeness: 100%')
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
    
    plt.suptitle('Egyptian SME Dataset - Comprehensive EDA Dashboard', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('analysis/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create interactive visualizations with Plotly
    print("\nCreating interactive visualizations...")
    
    # Interactive sector analysis
    if 'القطاع' in df.columns and 'المبيعات_جنيه' in df.columns:
        sector_summary = df.groupby('القطاع').agg({
            'المبيعات_جنيه': ['sum', 'mean', 'count'],
            'score': ['mean', 'median'],
            'الموظفون': 'sum'
        })
        sector_summary.columns = ['_'.join(col).strip() for col in sector_summary.columns]
        sector_summary = sector_summary.sort_values('المبيعات_جنيه_sum', ascending=False).head(15)
        
        reshaped_index = [get_display(arabic_reshaper.reshape(idx)) for idx in sector_summary.index]
        
        # Interactive bar chart
        fig_interactive = go.Figure(data=[
            go.Bar(
                x=reshaped_index,
                y=sector_summary['المبيعات_جنيه_sum'] / 1e9,
                text=sector_summary['المبيعات_جنيه_count'],
                texttemplate='%{text} companies',
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                              'Total Sales: %{y:.2f}B EGP<br>' +
                              'Companies: %{text}<br>' +
                              '<extra></extra>',
                marker_color='steelblue'
            )
        ])
        
        fig_interactive.update_layout(
            title='Interactive Sector Analysis - Total Sales (Top 15)',
            xaxis_title='Sector',
            yaxis_title='Total Sales (Billions EGP)',
            height=600,
            xaxis_tickangle=-45
        )
        
        fig_interactive.write_html('analysis/interactive_sector_analysis.html')
        print("✓ Created: interactive_sector_analysis.html")
    
    # Summary statistics export
    print("\nGenerating summary report...")
    
    summary_stats = {
        'Total Records': len(df),
        'Total Companies': df['الرقم_الضريبي'].nunique() if 'الرقم_الضريبي' in df.columns else 'N/A',
        'Years Covered': f"{df['السنة'].min()}-{df['السنة'].max()}" if 'السنة' in df.columns else 'N/A',
        'Sectors': df['القطاع'].nunique() if 'القطاع' in df.columns else 'N/A',
        'Average Sales': f"{df['المبيعات_جنيه'].mean()/1e6:.1f}M EGP" if 'المبيعات_جنيه' in df.columns else 'N/A',
        'Average Score': f"{df['score'].mean():.2f}" if 'score' in df.columns else 'N/A',
        'Average Employees': f"{df['الموظفون'].mean():.0f}" if 'الموظفون' in df.columns else 'N/A',
    }
    
    print("FINAL SUMMARY STATISTICS:")
    print("-" * 40)
    for key, value in summary_stats.items():
        print(f"{key}: {value}")
    
    return summary_stats

def main():
    """Main function to run all EDA analyses"""
    print("COMPREHENSIVE EDA ANALYSIS - EGYPTIAN SME DATASET")
    print("=" * 80)
    
    df = load_data()
    
    data_overview_and_quality()
    
    # univariate_analysis()
    
    # bivariate_analysis()
    
    # multivariate_analysis()
    
    # timeseries_trend_analysis()
    
    # outlier_anomaly_detection()
    
    # grouping_aggregation_analysis()
    
    # derived_metrics_analysis()
    
    # data_quality_bias_checks()
    
    # visualizations_reporting()
    
    print("\n" + "=" * 80)
    print("EDA ANALYSIS COMPLETE")
    print("All visualizations and reports saved to analysis/")
    print("=" * 80)

if __name__ == "__main__":
    main()