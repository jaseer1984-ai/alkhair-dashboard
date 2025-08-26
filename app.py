import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List
import io

# Configuration
st.set_page_config(page_title="Al Khair Business Performance", layout="wide", page_icon="📊")

class Config:
    BRANCHES = {
        "Al Khair": {"color": "#3b82f6", "code": "102"},
        "Noora": {"color": "#10b981", "code": "104"}, 
        "Hamra": {"color": "#f59e0b", "code": "109"},
        "Magnus": {"color": "#8b5cf6", "code": "107"},
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}

config = Config()

# Enhanced CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"] { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
}

.main .block-container { 
    padding: 1.2rem 2rem; 
    max-width: 1500px; 
}

.title { 
    font-weight: 900; 
    font-size: 1.8rem; 
    margin-bottom: .25rem; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle { 
    color: #6b7280; 
    font-size: .95rem; 
    margin-bottom: 1rem; 
}

[data-testid="metric-container"] {
    background: #fff !important; 
    border-radius: 16px !important; 
    border: 1px solid rgba(0,0,0,.06) !important; 
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
}

[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
    transform: translateY(-1px) !important;
}

.card { 
    background: linear-gradient(135deg, #fcfcfc 0%, #f8fafc 100%); 
    border: 1px solid #e2e8f0; 
    border-radius: 16px; 
    padding: 20px; 
    height: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
}

.card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}

.pill { 
    display: inline-block; 
    padding: 6px 12px; 
    border-radius: 999px; 
    font-weight: 700; 
    font-size: .80rem;
    transition: all 0.2s ease;
}

.pill.excellent { 
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
    color: #059669; 
    border: 1px solid #a7f3d0;
}

.pill.good { 
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
    color: #2563eb; 
    border: 1px solid #93c5fd;
}

.pill.warn { 
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); 
    color: #d97706; 
    border: 1px solid #fcd34d;
}

.pill.danger { 
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%); 
    color: #dc2626; 
    border: 1px solid #fca5a5;
}

.alert-card {
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border-left: 4px solid;
}

.alert-critical {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-left-color: #dc2626;
    color: #7f1d1d;
}

.alert-warning {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-left-color: #f59e0b;
    color: #92400e;
}

@media (max-width: 768px) {
    .main .block-container { 
        padding: 0.8rem 1rem; 
    }
}
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data():
    """Generate realistic sample data for Al Khair branches."""
    np.random.seed(42)  # For consistent sample data
    
    branches = list(config.BRANCHES.keys())
    data = []
    
    # Generate 30 days of data
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        current_date = base_date + timedelta(days=i)
        
        for branch in branches:
            # Realistic business metrics with some variation
            base_sales = np.random.normal(25000, 5000)  # Average sales around 25K
            sales_target = base_sales * np.random.uniform(1.05, 1.15)  # Target 5-15% higher
            sales_actual = base_sales * np.random.uniform(0.85, 1.1)   # Actual varies
            
            nob_target = np.random.normal(180, 30)  # Number of baskets target
            nob_actual = nob_target * np.random.uniform(0.8, 1.15)
            
            abv_target = sales_target / nob_target if nob_target > 0 else 150
            abv_actual = sales_actual / nob_actual if nob_actual > 0 else 150
            
            data.append({
                'Date': current_date,
                'BranchName': branch,
                'Branch': branch,
                'BranchCode': config.BRANCHES[branch]['code'],
                'SalesTarget': max(0, sales_target),
                'SalesActual': max(0, sales_actual), 
                'SalesPercent': (sales_actual / sales_target * 100) if sales_target > 0 else 0,
                'NOBTarget': max(0, nob_target),
                'NOBActual': max(0, nob_actual),
                'NOBPercent': (nob_actual / nob_target * 100) if nob_target > 0 else 0,
                'ABVTarget': abv_target,
                'ABVActual': abv_actual,
                'ABVPercent': (abv_actual / abv_target * 100) if abv_target > 0 else 0,
            })
    
    return pd.DataFrame(data)

def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key performance indicators."""
    if df.empty:
        return {}
    
    kpis = {}
    
    # Sales metrics
    kpis['total_sales_target'] = df['SalesTarget'].sum()
    kpis['total_sales_actual'] = df['SalesActual'].sum()
    kpis['total_sales_variance'] = kpis['total_sales_actual'] - kpis['total_sales_target']
    kpis['overall_sales_percent'] = (kpis['total_sales_actual'] / kpis['total_sales_target'] * 100) if kpis['total_sales_target'] > 0 else 0
    
    # NOB metrics
    kpis['total_nob_target'] = df['NOBTarget'].sum()
    kpis['total_nob_actual'] = df['NOBActual'].sum()
    kpis['overall_nob_percent'] = (kpis['total_nob_actual'] / kpis['total_nob_target'] * 100) if kpis['total_nob_target'] > 0 else 0
    
    # ABV metrics
    kpis['avg_abv_target'] = df['ABVTarget'].mean()
    kpis['avg_abv_actual'] = df['ABVActual'].mean()
    kpis['overall_abv_percent'] = (kpis['avg_abv_actual'] / kpis['avg_abv_target'] * 100) if kpis['avg_abv_target'] > 0 else 0
    
    # Performance score (weighted average)
    weights = {'sales': 0.4, 'nob': 0.35, 'abv': 0.25}
    score = 0.0
    total_weight = 0.0
    
    if kpis['overall_sales_percent'] > 0:
        score += min(kpis['overall_sales_percent'] / 100, 1.2) * weights['sales'] * 100
        total_weight += weights['sales']
    
    if kpis['overall_nob_percent'] > 0:
        score += min(kpis['overall_nob_percent'] / 100, 1.2) * weights['nob'] * 100
        total_weight += weights['nob']
    
    if kpis['overall_abv_percent'] > 0:
        score += min(kpis['overall_abv_percent'] / 100, 1.2) * weights['abv'] * 100
        total_weight += weights['abv']
    
    kpis['performance_score'] = (score / total_weight) if total_weight > 0 else 0.0
    
    # Branch performance
    if 'BranchName' in df.columns:
        kpis['branch_performance'] = df.groupby('BranchName').agg({
            'SalesTarget': 'sum',
            'SalesActual': 'sum',
            'SalesPercent': 'mean',
            'NOBTarget': 'sum',
            'NOBActual': 'sum',
            'NOBPercent': 'mean',
            'ABVTarget': 'mean',
            'ABVActual': 'mean',
            'ABVPercent': 'mean',
        }).round(2)
    
    return kpis

def create_trend_chart(df: pd.DataFrame, y_col: str, title: str) -> go.Figure:
    """Create trend chart for metrics."""
    if df.empty or 'Date' not in df.columns:
        return go.Figure()
    
    fig = go.Figure()
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
    
    for i, branch in enumerate(sorted(df['BranchName'].unique())):
        branch_data = df[df['BranchName'] == branch].sort_values('Date')
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=branch_data['Date'],
            y=branch_data[y_col],
            name=branch,
            mode='lines+markers',
            line=dict(width=3, color=color),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{branch}</b><br>Date: %{{x|%b %d}}<br>Value: %{{y:,.0f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=title,
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        xaxis_title="Date",
        yaxis_title="Value"
    )
    
    return fig

def create_performance_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create performance heatmap."""
    if df.empty or 'BranchName' not in df.columns:
        return go.Figure()
    
    branch_performance = df.groupby('BranchName').agg({
        'SalesPercent': 'mean',
        'NOBPercent': 'mean',
        'ABVPercent': 'mean'
    }).round(1)
    
    if branch_performance.empty:
        return go.Figure()
    
    metrics = ['SalesPercent', 'NOBPercent', 'ABVPercent']
    metric_names = ['Sales %', 'NOB %', 'ABV %']
    
    z_data = []
    branches = list(branch_performance.index)
    
    for metric in metrics:
        z_data.append(branch_performance[metric].tolist())
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=branches,
        y=metric_names,
        colorscale='RdYlGn',
        text=z_data,
        texttemplate="%{text:.1f}%",
        textfont={"size": 14, "color": "black"},
        hovertemplate="<b>%{y}</b><br>Branch: %{x}<br>Performance: %{z:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Branch Performance Heatmap",
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def render_branch_cards(branch_performance: pd.DataFrame):
    """Render branch performance cards."""
    if branch_performance.empty:
        st.info("No branch data available.")
        return
    
    st.markdown("### Branch Performance Overview")
    
    # Sort by performance
    bp_sorted = branch_performance.copy()
    bp_sorted['avg_performance'] = bp_sorted[['SalesPercent', 'NOBPercent', 'ABVPercent']].mean(axis=1)
    bp_sorted = bp_sorted.sort_values('avg_performance', ascending=False)
    
    branches = list(bp_sorted.index)
    cols_per_row = min(2, len(branches))
    
    for i in range(0, len(branches), cols_per_row):
        row_branches = branches[i:i + cols_per_row]
        cols = st.columns(len(row_branches))
        
        for j, branch in enumerate(row_branches):
            with cols[j]:
                branch_data = bp_sorted.loc[branch]
                avg_performance = branch_data['avg_performance']
                
                # Status determination
                if avg_performance >= 95:
                    status_class = "excellent"
                    status_emoji = "🟢"
                elif avg_performance >= 85:
                    status_class = "good"
                    status_emoji = "🟡"
                elif avg_performance >= 75:
                    status_class = "warn"
                    status_emoji = "🟠"
                else:
                    status_class = "danger"
                    status_emoji = "🔴"
                
                branch_color = config.BRANCHES.get(branch, {}).get('color', '#6b7280')
                
                st.markdown(f"""
                    <div class="card" style="background: linear-gradient(135deg, {branch_color}15 0%, #ffffff 50%);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div style="font-weight: 800; font-size: 1.1em; color: #1e293b;">{branch}</div>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span>{status_emoji}</span>
                                <span class="pill {status_class}">{avg_performance:.1f}%</span>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px;">
                            <div style="text-align: center;">
                                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 4px;">Sales</div>
                                <div style="font-weight: 700; font-size: 1.1em; color: #1e293b;">{branch_data.get('SalesPercent', 0):.1f}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 4px;">NOB</div>
                                <div style="font-weight: 700; font-size: 1.1em; color: #1e293b;">{branch_data.get('NOBPercent', 0):.1f}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 4px;">ABV</div>
                                <div style="font-weight: 700; font-size: 1.1em; color: #1e293b;">{branch_data.get('ABVPercent', 0):.1f}%</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0; font-size: 0.8em; color: #64748b;">
                            Actual Sales: SAR {branch_data.get('SalesActual', 0):,.0f}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

def generate_alerts(kpis: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate performance alerts."""
    alerts = []
    
    sales_pct = kpis.get('overall_sales_percent', 0)
    if sales_pct < 75:
        alerts.append({
            'type': 'critical',
            'title': 'Critical Sales Performance',
            'message': f'Overall sales achievement at {sales_pct:.1f}% - immediate attention required!'
        })
    elif sales_pct < 85:
        alerts.append({
            'type': 'warning',
            'title': 'Sales Below Target',
            'message': f'Sales achievement at {sales_pct:.1f}% - below target threshold'
        })
    
    # NOB alerts
    nob_pct = kpis.get('overall_nob_percent', 0)
    if nob_pct < 75:
        alerts.append({
            'type': 'critical',
            'title': 'Critical Basket Count',
            'message': f'Number of baskets at {nob_pct:.1f}% - customer acquisition issue!'
        })
    
    return alerts

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown("<div class='title'>Al Khair Business Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Demo Dashboard with Sample Data</div>", unsafe_allow_html=True)
    
    # Load sample data
    df = generate_sample_data()
    
    # Date filter
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        start_date = st.date_input("Start Date", value=df['Date'].min().date(), 
                                  min_value=df['Date'].min().date(), 
                                  max_value=df['Date'].max().date())
    
    with col2:
        end_date = st.date_input("End Date", value=df['Date'].max().date(),
                                min_value=df['Date'].min().date(),
                                max_value=df['Date'].max().date())
    
    # Filter data by date
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    df_filtered = df.loc[mask]
    
    if df_filtered.empty:
        st.warning("No data in selected date range.")
        return
    
    # Calculate KPIs
    kpis = calculate_kpis(df_filtered)
    
    # Show alerts
    alerts = generate_alerts(kpis)
    if alerts:
        st.markdown("### Performance Alerts")
        for alert in alerts:
            alert_type = alert['type']
            icon = "🔴" if alert_type == 'critical' else "🟡"
            st.markdown(f"""
                <div class="alert-card alert-{alert_type}">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span>{icon}</span>
                        <div>
                            <div style="font-weight: 700; margin-bottom: 4px;">{alert['title']}</div>
                            <div style="font-size: 0.9em;">{alert['message']}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Main metrics
    st.markdown("### Overall Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Sales", f"SAR {kpis['total_sales_actual']:,.0f}", 
                 delta=f"{kpis['overall_sales_percent']:.1f}% of target")
    
    with col2:
        variance_color = "normal" if kpis['total_sales_variance'] >= 0 else "inverse"
        st.metric("Sales Variance", f"SAR {kpis['total_sales_variance']:,.0f}",
                 delta=f"{kpis['overall_sales_percent']-100:+.1f}%", 
                 delta_color=variance_color)
    
    with col3:
        nob_color = "normal" if kpis['overall_nob_percent'] >= 90 else "inverse"
        st.metric("Total Baskets", f"{kpis['total_nob_actual']:,.0f}",
                 delta=f"{kpis['overall_nob_percent']:.1f}% achievement",
                 delta_color=nob_color)
    
    with col4:
        abv_color = "normal" if kpis['overall_abv_percent'] >= 90 else "inverse"
        st.metric("Avg Basket Value", f"SAR {kpis['avg_abv_actual']:,.2f}",
                 delta=f"{kpis['overall_abv_percent']:.1f}% vs target",
                 delta_color=abv_color)
    
    with col5:
        score_color = "normal" if kpis['performance_score'] >= 80 else "off"
        st.metric("Performance Score", f"{kpis['performance_score']:.0f}/100",
                 delta="Weighted average", delta_color=score_color)
    
    # Branch cards
    if 'branch_performance' in kpis and not kpis['branch_performance'].empty:
        render_branch_cards(kpis['branch_performance'])
    
    # Tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["Daily Trends", "Performance Analysis", "Export Data"])
    
    with tab1:
        st.markdown("### Daily Performance Trends")
        
        trend_tabs = st.tabs(["Sales Trends", "Basket Trends", "Value Trends"])
        
        with trend_tabs[0]:
            sales_chart = create_trend_chart(df_filtered, 'SalesActual', 'Daily Sales Performance')
            st.plotly_chart(sales_chart, use_container_width=True, config={'displayModeBar': False})
        
        with trend_tabs[1]:
            nob_chart = create_trend_chart(df_filtered, 'NOBActual', 'Daily Number of Baskets')
            st.plotly_chart(nob_chart, use_container_width=True, config={'displayModeBar': False})
        
        with trend_tabs[2]:
            abv_chart = create_trend_chart(df_filtered, 'ABVActual', 'Daily Average Basket Value')
            st.plotly_chart(abv_chart, use_container_width=True, config={'displayModeBar': False})
    
    with tab2:
        st.markdown("### Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            heatmap = create_performance_heatmap(df_filtered)
            st.plotly_chart(heatmap, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            if 'branch_performance' in kpis and not kpis['branch_performance'].empty:
                st.markdown("#### Branch Performance Table")
                performance_df = kpis['branch_performance'][['SalesPercent', 'NOBPercent', 'ABVPercent', 'SalesActual']].round(2)
                st.dataframe(performance_df, use_container_width=True)
    
    with tab3:
        st.markdown("### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_filtered.to_excel(writer, sheet_name='Raw Data', index=False)
                if 'branch_performance' in kpis:
                    kpis['branch_performance'].to_excel(writer, sheet_name='Branch Summary')
            
            st.download_button(
                "Download Excel Report",
                buffer.getvalue(),
                f"Al_Khair_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # CSV export
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV Data",
                csv_data,
                f"Al_Khair_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Export Summary")
            st.info(f"""
            **Date Range:** {start_date} to {end_date}  
            **Branches:** {len(df_filtered['BranchName'].unique())}  
            **Records:** {len(df_filtered):,}  
            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)
    
    # Instructions for connecting real data
    with st.expander("Connect Your Google Sheets Data", expanded=False):
        st.markdown("""
        ### To connect your real Google Sheets data:
        
        1. **Check your Google Sheet URL** - Make sure it's properly published
        2. **Verify permissions** - Set to "Anyone with the link can view"  
        3. **Test the URL manually** - Open your CSV link in a browser
        4. **Check data format** - Ensure columns match expected names
        
        **Common Google Sheets Issues:**
        - 400 errors often indicate the sheet isn't properly published
        - Make sure you're using the CSV publish URL, not the edit URL
        - Check that your sheet contains actual data, not just headers
        - Try republishing the sheet and getting a fresh URL
        
        **Expected Column Names:**
        - Date, Sales Target, Sales Achievement, Sales %
        - NOB Target, NOB Achievement, NOB %  
        - ABV Target, ABV Achievement, ABV %
        
        This demo shows the dashboard working with sample data. Once your Google Sheets connection is fixed, 
        replace the sample data generation with the Google Sheets loader.
        """)

if __name__ == "__main__":
    main()
