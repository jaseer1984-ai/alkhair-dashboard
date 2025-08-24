)
        
        fig.update_layout(
            title="Branch Performance Comparison - Sales Achievement",
            xaxis_title="Branch",
            yaxis_title="Achievement (%)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=12),
            height=450,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_multi_metric_radar(branch_data: pd.DataFrame) -> go.Figure:
        """Create radar chart for multi-metric comparison"""
        if branch_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        metrics = ['SalesPercent', 'NOBPercent', 'ABVPercent']
        metric_labels = ['Sales Achievement', 'NOB Achievement', 'ABV Achievement']
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
        
        for i, branch in enumerate(branch_data.index):
            values = [branch_data.loc[branch, metric] for metric in metrics]
            values.append(values[0])  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels + [metric_labels[0]],
                fill='toself',
                name=branch,
                line=dict(color=colors[i % len(colors)], width=3),
                fillcolor=f"rgba{tuple(list(bytes.fromhex(colors[i % len(colors)][1:])[:3]) + [0.1])}"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 120],
                    tickmode='linear',
                    tick0=0,
                    dtick=20
                )
            ),
            title="Branch Performance Radar - All Metrics",
            showlegend=True,
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter")
        )
        
        return fig
    
    @staticmethod
    def create_daily_trend_chart(df: pd.DataFrame) -> go.Figure:
        """Create beautiful daily trend analysis"""
        if df.empty or df['Date'].isna().all():
            return go.Figure()
        
        daily_data = df.groupby(['Date', 'BranchName']).agg({
            'SalesActual': 'sum',
            'NOBActual': 'sum',
            'ABVActual': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Daily Sales Performance', 'Number of Baskets', 'Average Basket Value'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
        
        for i, branch in enumerate(daily_data['BranchName'].unique()):
            branch_data = daily_data[daily_data['BranchName'] == branch]
            color = colors[i % len(colors)]
            
            # Sales
            fig.add_trace(
                go.Scatter(
                    x=branch_data['Date'],
                    y=branch_data['SalesActual'],
                    name=f"{branch} - Sales",
                    line=dict(color=color, width=3),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            # NOB
            fig.add_trace(
                go.Scatter(
                    x=branch_data['Date'],
                    y=branch_data['NOBActual'],
                    name=f"{branch} - NOB",
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # ABV
            fig.add_trace(
                go.Scatter(
                    x=branch_data['Date'],
                    y=branch_data['ABVActual'],
                    name=f"{branch} - ABV",
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=700,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")
        )
        
        return fig

# =========================
# TAB FUNCTIONS
# =========================
def render_branch_overview(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render beautiful branch overview"""
    
    if df.empty:
        st.warning("No data available. Please upload your Excel file.")
        return
    
    # Overall performance metrics
    st.markdown("### 🏆 Overall Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Total Sales",
            f"SAR {kpis['total_sales_actual']:,.0f}",
            delta=f"vs Target: {kpis['overall_sales_percent']:.1f}%"
        )
    
    with col2:
        variance_color = "normal" if kpis['total_sales_variance'] >= 0 else "inverse"
        st.metric(
            "📊 Sales Variance",
            f"SAR {kpis['total_sales_variance']:,.0f}",
            delta=f"{kpis['overall_sales_percent']-100:+.1f}%",
            delta_color=variance_color
        )
    
    with col3:
        nob_color = "normal" if kpis['overall_nob_percent'] >= 90 else "inverse"
        st.metric(
            "🛍️ Total Baskets",
            f"{kpis['total_nob_actual']:,.0f}",
            delta=f"Achievement: {kpis['overall_nob_percent']:.1f}%",
            delta_color=nob_color
        )
    
    with col4:
        abv_color = "normal" if kpis['overall_abv_percent'] >= 90 else "inverse"
        st.metric(
            "💎 Avg Basket Value",
            f"SAR {kpis['avg_abv_actual']:.2f}",
            delta=f"vs Target: {kpis['overall_abv_percent']:.1f}%",
            delta_color=abv_color
        )
    
    with col5:
        score_color = "normal" if kpis['performance_score'] >= 80 else "off"
        st.metric(
            "⭐ Performance Score",
            f"{kpis['performance_score']:.0f}/100",
            delta="Weighted Score",
            delta_color=score_color
        )
    
    # Branch comparison visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'branch_performance' in kpis:
            comparison_chart = BranchVisualization.create_branch_comparison_chart(kpis['branch_performance'])
            st.plotly_chart(comparison_chart, use_container_width=True)
    
    with col2:
        if 'branch_performance' in kpis:
            radar_chart = BranchVisualization.create_multi_metric_radar(kpis['branch_performance'])
            st.plotly_chart(radar_chart, use_container_width=True)
    
    # Individual branch cards
    st.markdown("### 🏪 Branch Performance Dashboard")
    
    if 'branch_performance' in kpis:
        branch_data = kpis['branch_performance']
        
        for i, (branch_name, metrics) in enumerate(branch_data.iterrows()):
            branch_info = None
            for key, info in config.BRANCHES.items():
                if info['name'] == branch_name:
                    branch_info = info
                    break
            
            if not branch_info:
                branch_info = {"color": "#6b7280", "code": "000"}
            
            # Performance status
            avg_performance = (metrics['SalesPercent'] + metrics['NOBPercent'] + metrics['ABVPercent']) / 3
            
            if avg_performance >= 95:
                status_class = "status-excellent"
                status_text = "🏆 EXCELLENT"
            elif avg_performance >= 85:
                status_class = "status-good"
                status_text = "✅ GOOD"
            elif avg_performance >= 75:
                status_class = "status-warning"
                status_text = "⚠️ NEEDS FOCUS"
            else:
                status_class = "status-danger"
                status_text = "🔴 ACTION REQUIRED"
            
            st.markdown(f"""
            <div class="branch-comparison-card" style="--branch-color: {branch_info['color']};">
                <div class="branch-header">
                    <span style="color: {branch_info['color']};">🏪</span>
                    {branch_name} (Code: {branch_info['code']})
                    <span class="branch-badge {status_class}">{status_text}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Branch metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Sales Achievement",
                    f"SAR {metrics['SalesActual']:,.0f}",
                    delta=f"{metrics['SalesPercent']:.1f}% of target"
                )
            
            with col2:
                st.metric(
                    "Baskets (NOB)",
                    f"{metrics['NOBActual']:,.0f}",
                    delta=f"{metrics['NOBPercent']:.1f}% of target"
                )
            
            with col3:
                st.metric(
                    "Avg Basket Value",
                    f"SAR {metrics['ABVActual']:.2f}",
                    delta=f"{metrics['ABVPercent']:.1f}% of target"
                )
            
            with col4:
                st.metric(
                    "Overall Score",
                    f"{avg_performance:.1f}%",
                    delta="Average Performance"
                )
            
            # Progress bars
            st.markdown(f"""
                <div style="margin-top: 16px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: var(--text-secondary);">Sales</span>
                        <span style="font-weight: 700; color: {branch_info['color']};">{metrics['SalesPercent']:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(metrics['SalesPercent'], 100)}%; background: linear-gradient(90deg, {branch_info['color']}, #22d3ee);"></div>
                    </div>
                </div>
                
                <div style="margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: var(--text-secondary);">Baskets (NOB)</span>
                        <span style="font-weight: 700; color: {branch_info['color']};">{metrics['NOBPercent']:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(metrics['NOBPercent'], 100)}%; background: linear-gradient(90deg, {branch_info['color']}, #10b981);"></div>
                    </div>
                </div>
                
                <div style="margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: var(--text-secondary);">Basket Value (ABV)</span>
                        <span style="font-weight: 700; color: {branch_info['color']};">{metrics['ABVPercent']:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(metrics['ABVPercent'], 100)}%; background: linear-gradient(90deg, {branch_info['color']}, #f59e0b);"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_daily_trends(df: pd.DataFrame):
    """Render daily trends analysis"""
    st.markdown("## 📈 Daily Performance Trends")
    
    if df.empty:
        st.info("No data available for trend analysis.")
        return
    
    # Time period selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if df['Date'].notna().any():
            date_range = st.selectbox(
                "📅 Time Period",
                ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"]
            )
            
            today = datetime.now().date()
            if date_range == "Last 7 Days":
                start_date = today - timedelta(days=7)
                filtered_df = df[df['Date'].dt.date >= start_date]
            elif date_range == "Last 30 Days":
                start_date = today - timedelta(days=30)
                filtered_df = df[df['Date'].dt.date >= start_date]
            elif date_range == "Last 3 Months":
                start_date = today - timedelta(days=90)
                filtered_df = df[df['Date'].dt.date >= start_date]
            else:
                filtered_df = df.copy()
        else:
            filtered_df = df.copy()
    
    # Daily trends chart
    if not filtered_df.empty:
        trend_chart = BranchVisualization.create_daily_trend_chart(filtered_df)
        st.plotly_chart(trend_chart, use_container_width=True)
        
        # Weekly performance summary
        if filtered_df['Date'].notna().any():
            st.markdown("### 📊 Weekly Performance Summary")
            
            weekly_data = filtered_df.copy()
            weekly_data['Week'] = weekly_data['Date'].dt.isocalendar().week
            weekly_data['Year'] = weekly_data['Date'].dt.year
            
            weekly_summary = weekly_data.groupby(['Year', 'Week', 'BranchName']).agg({
                'SalesActual': 'sum',
                'SalesTarget': 'sum',
                'NOBActual': 'sum',
                'ABVActual': 'mean'
            }).reset_index()
            
            weekly_summary['SalesAchievement'] = (weekly_summary['SalesActual'] / weekly_summary['SalesTarget'] * 100).round(1)
            weekly_summary['WeekLabel'] = weekly_summary['Year'].astype(str) + '-W' + weekly_summary['Week'].astype(str)
            
            # Display weekly data
            pivot_data = weekly_summary.pivot(index='WeekLabel', columns='BranchName', values='SalesAchievement').fillna(0)
            
            # Format for display
            display_data = pivot_data.copy()
            for col in display_data.columns:
                display_data[col] = display_data[col].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
            
            st.dataframe(display_data, use_container_width=True)

def render_branch_comparison(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render detailed branch comparison"""
    st.markdown("## 🏆 Branch Competition Analysis")
    
    if df.empty or 'branch_performance' not in kpis:
        st.info("No branch data available for comparison.")
        return
    
    branch_data = kpis['branch_performance']
    
    # Top performers
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Best sales performer
        best_sales_branch = branch_data['SalesPercent'].idxmax()
        best_sales_score = branch_data.loc[best_sales_branch, 'SalesPercent']
        
        st.markdown(f"""
        <div class="branch-comparison-card">
            <div class="branch-header">
                🥇 Sales Champion
            </div>
            <div style="text-align: center;">
                <h2 style="color: var(--success); margin: 0;">{best_sales_branch}</h2>
                <h1 style="color: var(--success); margin: 8px 0;">{best_sales_score:.1f}%</h1>
                <p style="color: var(--text-secondary); margin: 0;">Target Achievement</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Best NOB performer
        best_nob_branch = branch_data['NOBPercent'].idxmax()
        best_nob_score = branch_data.loc[best_nob_branch, 'NOBPercent']
        
        st.markdown(f"""
        <div class="branch-comparison-card">
            <div class="branch-header">
                🥇 Basket Champion
            </div>
            <div style="text-align: center;">
                <h2 style="color: var(--primary); margin: 0;">{best_nob_branch}</h2>
                <h1 style="color: var(--primary); margin: 8px 0;">{best_nob_score:.1f}%</h1>
                <p style="color: var(--text-secondary); margin: 0;">NOB Achievement</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Best ABV performer
        best_abv_branch = branch_data['ABVPercent'].idxmax()
        best_abv_score = branch_data.loc[best_abv_branch, 'ABVPercent']
        
        st.markdown(f"""
        <div class="branch-comparison-card">
            <div class="branch-header">
                🥇 Value Champion
            </div>
            <div style="text-align: center;">
                <h2 style="color: var(--warning); margin: 0;">{best_abv_branch}</h2>
                <h1 style="color: var(--warning); margin: 8px 0;">{best_abv_score:.1f}%</h1>
                <p style="color: var(--text-secondary); margin: 0;">ABV Achievement</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed comparison table
    st.markdown("### 📋 Detailed Branch Metrics")
    
    # Prepare display data
    display_branch_data = branch_data.copy()
    display_branch_data['SalesTarget'] = display_branch_data['SalesTarget'].apply(lambda x: f"SAR {x:,.0f}")
    display_branch_data['SalesActual'] = display_branch_data['SalesActual'].apply(lambda x: f"SAR {x:,.0f}")
    display_branch_data['SalesPercent'] = display_branch_data['SalesPercent'].apply(lambda x: f"{x:.1f}%")
    display_branch_data['NOBTarget'] = display_branch_data['NOBTarget'].apply(lambda x: f"{x:,.0f}")
    display_branch_data['NOBActual'] = display_branch_data['NOBActual'].apply(lambda x: f"{x:,.0f}")
    display_branch_data['NOBPercent'] = display_branch_data['NOBPercent'].apply(lambda x: f"{x:.1f}%")
    display_branch_data['ABVTarget'] = display_branch_data['ABVTarget'].apply(lambda x: f"SAR {x:.2f}")
    display_branch_data['ABVActual'] = display_branch_data['ABVActual'].apply(lambda x: f"SAR {x:.2f}")
    display_branch_data['ABVPercent'] = display_branch_data['ABVPercent'].apply(lambda x: f"{x:.1f}%")
    
    # Rename columns for display
    display_columns = {
        'SalesTarget': 'Sales Target',
        'SalesActual': 'Sales Actual', 
        'SalesPercent': 'Sales %',
        'NOBTarget': 'NOB Target',
        'NOBActual': 'NOB Actual',
        'NOBPercent': 'NOB %',
        'ABVTarget': 'ABV Target',
        'ABVActual': 'ABV Actual',
        'ABVPercent': 'ABV %'
    }
    
    display_branch_data = display_branch_data.rename(columns=display_columns)
    st.dataframe(display_branch_data, use_container_width=True)

def render_file_upload_section():
    """Render beautiful file upload section"""
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h2 style="color: white; margin-bottom: 20px;">📁 Upload Your Branch Data</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">
            Upload your Excel file with branch sheets to view comprehensive analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose Excel file",
        type=['xlsx', 'xls'],
        help="Upload Excel file with branch data sheets",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        return uploaded_file
    
    # Sample data structure info
    st.markdown("""
    <div class="branch-comparison-card" style="margin-top: 40px;">
        <h3 style="color: var(--text-primary); margin-bottom: 16px;">📊 Expected Data Structure</h3>
        <p style="color: var(--text-secondary);">Your Excel file should contain:</p>
        <ul style="color: var(--text-secondary); padding-left: 20px;">
            <li><strong>Multiple sheets</strong> - One for each branch</li>
            <li><strong>Sheet names</strong> - Branch names (e.g., "Al khair - 102", "Magnus - 107")</li>
            <li><strong>Columns</strong>: Date, Sales Target, Sales Achievement, NOB Target, NOB Achievement, ABV Target, ABV Achievement</li>
            <li><strong>Daily data</strong> - One row per day per branch</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    return None

# =========================
# MAIN APPLICATION
# =========================
def main():
    """Main branch dashboard application"""
    
    # Apply beautiful styling
    apply_branch_dashboard_css()
    
    # Beautiful Hero Header
    st.markdown(f"""
    <div class="hero-header floating">
        <div class="hero-content">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="margin: 0; font-size: 3rem; font-weight: 900; text-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        🏪 Alkhair Family Market
                    </h1>
                    <p style="margin: 12px 0 0 0; font-size: 1.3rem; opacity: 0.9; font-weight: 600;">
                        Multi-Branch Performance Dashboard • Real-time Analytics
                    </p>
                    <div style="margin-top: 16px; display: flex; gap: 12px;">
                        <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">
                            📊 {len(config.BRANCHES)} Branches
                        </span>
                        <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">
                            ⚡ Live Data
                        </span>
                    </div>
                </div>
                <div style="text-align: right; opacity: 0.9;">
                    <div style="font-size: 1.1rem; font-weight: 700;">📈 Executive Analytics</div>
                    <div style="font-size: 0.95rem; margin-top: 4px;">Updated: {datetime.now().strftime('%H:%M:%S')}</div>
                    <div style="font-size: 0.85rem; margin-top: 2px;">Real-time Dashboard</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = render_file_upload_section()
    
    if uploaded_file is None:
        st.stop()
    
    # Process uploaded file
    @st.cache_data
    def load_branch_data(file_content):
        """Load and process branch data from Excel file"""
        try:
            # Read all sheets
            excel_data = pd.read_excel(file_content, sheet_name=None, engine='openpyxl')
            
            # Process each sheet
            processed_data = BranchDataProcessor.process_branch_data(excel_data)
            
            return processed_data
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return pd.DataFrame()
    
    with st.spinner("🔄 Processing branch data..."):
        df = load_branch_data(uploaded_file)
    
    if df.empty:
        st.error("❌ Could not process the uploaded file. Please check the format.")
        st.stop()
    
    # Success message
    st.success(f"✅ Successfully loaded data for {df['BranchName'].nunique()} branches with {len(df):,} records!")
    
    # Calculate KPIs
    kpis = BranchAnalytics.calculate_branch_kpis(df)
    
    # Date filter
    if df['Date'].notna().any():
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            start_date = st.date_input(
                "📅 Start Date", 
                value=df['Date'].min().date(),
                key="start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "📅 End Date", 
                value=df['Date'].max().date(),
                key="end_date"
            )
        
        # Apply filter
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df[mask].copy()
        
        # Recalculate KPIs for filtered data
        kpis = BranchAnalytics.calculate_branch_kpis(filtered_df)
    else:
        filtered_df = df.copy()
    
    # Beautiful Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Branch Overview",
        "📈 Daily Trends", 
        "🏆 Competition Analysis",
        "📊 Data Export"
    ])
    
    with tab1:
        render_branch_overview(filtered_df, kpis)
    
    with tab2:
        render_daily_trends(filtered_df)
    
    with tab3:
        render_branch_comparison(filtered_df, kpis)
    
    with tab4:
        render_export_section(filtered_df, kpis)
    
    # Beautiful footer
    st.markdown("""
    <div style="text-align: center; padding: 60px 0 30px 0; color: rgba(255,255,255,0.8);">
        <p style="margin: 0; font-size: 1rem; font-weight: 500;">
            🏪 Alkhair Family Market • Multi-Branch Excellence Dashboard
        </p>
        <p style="margin: 8px 0 0 0; font-size: 0.9rem; opacity: 0.7;">
            Built with ❤️ for Data-Driven Decisions
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_export_section(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render data export section"""
    st.markdown("## 📥 Data Export & Analysis")
    
    if df.empty:
        st.info("No data to export.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
    with col3:
        st.markdown("""
        <div class="branch-comparison-card">
            <h4 style="margin: 0 0 12px 0; color: var(--text-primary);">💰 Financial</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Total Sales", f"SAR {kpis.get('total_sales_actual', 0):,.0f}")
        st.metric("Target Sales", f"SAR {kpis.get('total_sales_target', 0):,.0f}")
        st.metric("Variance", f"SAR {kpis.get('total_sales_variance', 0):,.0f}")
    
    with col4:
        st.markdown("""
        <div class="branch-comparison-card">
            <h4 style="margin: 0 0 12px 0; color: var(--text-primary);">📥 Export Data</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create comprehensive Excel export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Complete data
            df.to_excel(writer, sheet_name='All Branches Data', index=False)
            
            # Branch summary
            if 'branch_performance' in kpis:
                kpis['branch_performance'].to_excel(writer, sheet_name='Branch Summary')
            
            # Daily summary
            if df['Date'].notna().any():
                daily_summary = df.groupby(['Date', 'BranchName']).agg({
                    'SalesActual': 'sum',
                    'SalesTarget': 'sum',
                    'NOBActual': 'sum',
                    'ABVActual': 'mean'
                }).reset_index()
                daily_summary.to_excel(writer, sheet_name='Daily Summary', index=False)
        
        st.download_button(
            "📊 Download Excel Report",
            buffer.getvalue(),
            f"Alkhair_Branch_Analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # CSV export
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📄 Download CSV",
            csv_buffer.getvalue(),
            f"Alkhair_Branch_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Raw data preview
    st.markdown("### 📋 Data Preview")
    
    # Branch selector for preview
    selected_branch = st.selectbox(
        "Select Branch for Preview",
        options=["All Branches"] + list(df['BranchName'].unique())
    )
    
    if selected_branch == "All Branches":
        preview_data = df.head(100)
    else:
        preview_data = df[df['BranchName'] == selected_branch].head(50)
    
    st.dataframe(preview_data, use_container_width=True)

# =========================
# MISSING FUNCTION IMPLEMENTATIONS
# =========================
class BranchDataProcessor:
    """Process Excel data with multiple branch sheets"""
    
    @staticmethod
    def process_branch_data(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process and combine branch data"""
        combined_data = []
        
        for sheet_name, df in excel_data.items():
            if df.empty:
                continue
            
            # Clean column names
            df_clean = df.copy()
            df_clean.columns = df_clean.columns.str.strip()
            
            # Standard column mapping
            column_mapping = {
                "Date": "Date",
                "Day": "Day",
                "Sales Target": "SalesTarget", 
                "Sales Achivement": "SalesActual",
                "Sales Achievement": "SalesActual",
                "Sales %": "SalesPercent",
                " Sales %": "SalesPercent",
                "NOB Target": "NOBTarget",
                "NOB Achievemnet": "NOBActual", 
                "NOB Achievement": "NOBActual",
                "NOB %": "NOBPercent",
                " NOB %": "NOBPercent",
                "ABV Target": "ABVTarget",
                "ABV Achievement": "ABVActual",
                " ABV Achievement": "ABVActual",
                "ABV %": "ABVPercent"
            }
            
            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in df_clean.columns:
                    df_clean.rename(columns={old_name: new_name}, inplace=True)
            
            # Add branch info
            df_clean['Branch'] = sheet_name
            df_clean['BranchName'] = config.BRANCHES.get(sheet_name, {}).get('name', sheet_name)
            df_clean['BranchCode'] = config.BRANCHES.get(sheet_name, {}).get('code', '000')
            
            # Convert data types
            if 'Date' in df_clean.columns:
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            
            # Clean numeric columns
            numeric_cols = ['SalesTarget', 'SalesActual', 'SalesPercent', 'NOBTarget', 
                          'NOBActual', 'NOBPercent', 'ABVTarget', 'ABVActual', 'ABVPercent']
            
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            combined_data.append(df_clean)
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            return result
        else:
            return pd.DataFrame()

class BranchAnalytics:
    """Advanced branch analytics"""
    
    @staticmethod
    def calculate_branch_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive branch KPIs"""
        if df.empty:
            return {}
        
        kpis = {}
        
        # Overall totals
        kpis['total_sales_target'] = df['SalesTarget'].sum() if 'SalesTarget' in df.columns else 0
        kpis['total_sales_actual'] = df['SalesActual'].sum() if 'SalesActual' in df.columns else 0
        kpis['total_sales_variance'] = kpis['total_sales_actual'] - kpis['total_sales_target']
        kpis['overall_sales_percent'] = (kpis['total_sales_actual'] / kpis['total_sales_target'] * 100) if kpis['total_sales_target'] > 0 else 0
        
        kpis['total_nob_target'] = df['NOBTarget'].sum() if 'NOBTarget' in df.columns else 0
        kpis['total_nob_actual'] = df['NOBActual'].sum() if 'NOBActual' in df.columns else 0
        kpis['overall_nob_percent'] = (kpis['total_nob_actual'] / kpis['total_nob_target'] * 100) if kpis['total_nob_target'] > 0 else 0
        
        kpis['avg_abv_target'] = df['ABVTarget'].mean() if 'ABVTarget' in df.columns else 0
        kpis['avg_abv_actual'] = df['ABVActual'].mean() if 'ABVActual' in df.columns else 0
        kpis['overall_abv_percent'] = (kpis['avg_abv_actual'] / kpis['avg_abv_target'] * 100) if kpis['avg_abv_target'] > 0 else 0
        
        # Branch performance summary
        if 'BranchName' in df.columns:
            branch_performance = df.groupby('BranchName').agg({
                'SalesTarget': 'sum',
                'SalesActual': 'sum', 
                'SalesPercent': 'mean',
                'NOBTarget': 'sum',
                'NOBActual': 'sum',
                'NOBPercent': 'mean',
                'ABVTarget': 'mean',
                'ABVActual': 'mean',
                'ABVPercent': 'mean'
            }).round(2)
            
            kpis['branch_performance'] = branch_performance
        
        # Time analysis
        if df['Date'].notna().any():
            kpis['date_range'] = {
                'start': df['Date'].min(),
                'end': df['Date'].max(),
                'days': df['Date'].dt.date.nunique()
            }
            
            kpis['avg_daily_sales'] = kpis['total_sales_actual'] / kpis['date_range']['days'] if kpis['date_range']['days'] > 0 else 0
            kpis['avg_daily_nob'] = kpis['total_nob_actual'] / kpis['date_range']['days'] if kpis['date_range']['days'] > 0 else 0
        
        # Performance scoring
        kpis['performance_score'] = BranchAnalytics.calculate_overall_score(kpis)
        
        return kpis
    
    @staticmethod
    def calculate_overall_score(kpis: Dict[str, Any]) -> float:
        """Calculate weighted performance score"""
        score = 0
        weights = 0
        
        # Sales achievement (40% weight)
        if kpis.get('overall_sales_percent', 0) > 0:
            sales_score = min(kpis['overall_sales_percent'] / 100, 1.2)
            score += sales_score * 40
            weights += 40
        
        # NOB achievement (35% weight)  
        if kpis.get('overall_nob_percent', 0) > 0:
            nob_score = min(kpis['overall_nob_percent'] / 100, 1.2)
            score += nob_score * 35
            weights += 35
        
        # ABV achievement (25% weight)
        if kpis.get('overall_abv_percent', 0) > 0:
            abv_score = min(kpis['overall_abv_percent'] / 100, 1.2)
            score += abv_score * 25
            weights += 25
        
        return (score / weights * 100) if weights > 0 else 0

class BranchVisualization:
    """Create stunning branch visualizations"""
    
    @staticmethod
    def create_branch_comparison_chart(branch_data: pd.DataFrame) -> go.Figure:
        """Beautiful branch comparison chart"""
        if branch_data.empty:
            return go.Figure()
        
        branches = branch_data.index
        
        fig = go.Figure()
        
        # Sales Achievement bars
        fig.add_trace(go.Bar(
            x=branches,
            y=branch_data['SalesPercent'],
            name='Sales Achievement %',
            marker=dict(
                color=branch_data['SalesPercent'],
                colorscale='RdYlGn',
                cmin=0,
                cmax=120,
                showscale=True,
                colorbar=dict(title="Achievement %", x=1.02)
            ),
            text=[f"{val:.1f}%" for val in branch_data['SalesPercent']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Achievement: %{y:.1f}%<br>Target: SAR %{customdata:,.0f}<extra></extra>',
            customdata=branch_data['SalesTarget']
        ))
        
        # Target line at 100%
        fig.add_hline(
            y=100, 
            line_dash="dash", 
            line_color="#ef4444", 
            line_width=3,
            annotation_text="💯 Target Line",
            annotation_position="top right"
        )
        
        # Excellence line at 95%
        fig.add_hline(
            y=95,
            line_dash="dot",
            line_color="#10b981",
            line_width=2,
            annotation_text="🌟 Excellence",
            annotation_position="bottom right"
        )
        )
        
        # Excellence line at 95%
        fig.add_hline(
            y=95,
            line_dash="dot",
            line_color="#10b981",
            line_width=2,
            annotation_text="🌟 Excellence",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title="Branch Sales Performance Comparison",
            xaxis_title="Branch",
            yaxis_title="Achievement (%)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=12),
            height=450,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_multi_metric_radar(branch_data: pd.DataFrame) -> go.Figure:
        """Create radar chart for multi-metric comparison"""
        if branch_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        metrics = ['SalesPercent', 'NOBPercent', 'ABVPercent']
        metric_labels = ['Sales Achievement', 'NOB Achievement', 'ABV Achievement']
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
        
        for i, branch in enumerate(branch_data.index):
            values = [branch_data.loc[branch, metric] for metric in metrics]
            values.append(values[0])  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels + [metric_labels[0]],
                fill='toself',
                name=branch,
                line=dict(color=colors[i % len(colors)], width=3),
                fillcolor=f"rgba{tuple(list(bytes.fromhex(colors[i % len(colors)][1:])[:3]) + [0.15])}"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 120],
                    tickmode='linear',
                    tick0=0,
                    dtick=20,
                    gridcolor="rgba(255,255,255,0.3)"
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.3)"
                )
            ),
            title="Branch Performance Radar",
            showlegend=True,
            legend=dict(x=1.1, y=0.5),
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter")
        )
        
        return fig
    
    @staticmethod
    def create_daily_trend_chart(df: pd.DataFrame) -> go.Figure:
        """Create daily trend analysis"""
        if df.empty or df['Date'].isna().all():
            return go.Figure()
        
        daily_data = df.groupby(['Date', 'BranchName']).agg({
            'SalesActual': 'sum',
            'NOBActual': 'sum', 
            'ABVActual': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('📊 Daily Sales Performance', '🛍️ Number of Baskets', '💰 Average Basket Value'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
        
        for i, branch in enumerate(daily_data['BranchName'].unique()):
            branch_data = daily_data[daily_data['BranchName'] == branch]
            color = colors[i % len(colors)]
            
            # Sales trend
            fig.add_trace(
                go.Scatter(
                    x=branch_data['Date'],
                    y=branch_data['SalesActual'],
                    name=f"{branch}",
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # NOB trend
            fig.add_trace(
                go.Scatter(
                    x=branch_data['Date'],
                    y=branch_data['NOBActual'],
                    name=f"{branch} NOB",
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # ABV trend
            fig.add_trace(
                go.Scatter(
                    x=branch_data['Date'],
                    y=branch_data['ABVActual'],
                    name=f"{branch} ABV",
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=800,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)")
        
        return fig

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        st.stop() class="branch-comparison-card">
            <h4 style="margin: 0 0 12px 0; color: var(--text-primary);">📊 Data Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Branches", f"{df['BranchName'].nunique()}")
        st.metric("Date Range", f"{df['Date'].dt.date.nunique()} days")
    
    with col2:
        st.markdown("""
        <div class="branch-comparison-card">
            <h4 style="margin: 0 0 12px 0; color: var(--text-primary);">🎯 Performance</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Overall Score", f"{kpis.get('performance_score', 0):.0f}/100")
        st.metric("Sales Achievement", f"{kpis.get('overall_sales_percent', 0):.1f}%")
        st.metric("NOB Achievement", f"{kpis.get('overall_nob_percent', 0):.1f}%")
    
    with col3:
        st.markdown("""
        <div# Alkhair Family Market - Multi-Branch Executive Dashboard
# Beautiful branch-wise analytics with Excel data integration

import io
import re
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Tuple, Any
import base64

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# =========================
# CONFIGURATION
# =========================
class Config:
    """Dashboard configuration"""
    PAGE_TITLE = "Alkhair Family Market — Branch Analytics Dashboard"
    LAYOUT = "wide"
    
    # Branch information
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6"}
    }
    
    # Performance thresholds
    TARGETS = {
        "sales_achievement": 95.0,  # Target: 95%+ of sales target
        "nob_achievement": 90.0,    # Target: 90%+ of NOB target
        "abv_achievement": 90.0     # Target: 90%+ of ABV target
    }

config = Config()

# Page setup
st.set_page_config(
    page_title=config.PAGE_TITLE,
    layout=config.LAYOUT,
    initial_sidebar_state="collapsed"
)

# =========================
# STUNNING VISUAL DESIGN
# =========================
def apply_branch_dashboard_css():
    """Apply beautiful CSS for branch dashboard"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --primary: #3b82f6;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #06b6d4;
        --dark: #1e293b;
        --light: #f8fafc;
        --white: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --glass: rgba(255, 255, 255, 0.25);
        --glass-border: rgba(255, 255, 255, 0.18);
    }
    
    /* Animated Background */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        padding: 2rem;
        max-width: 1600px;
    }
    
    /* Glassmorphism Hero Header */
    .hero-header {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(30px);
        border: 1px solid var(--glass-border);
        border-radius: 28px;
        padding: 40px;
        margin: -1rem -1rem 3rem -1rem;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    /* Branch Cards */
    .branch-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 24px;
        padding: 32px;
        margin: 20px 0;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .branch-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.15);
    }
    
    .branch-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: var(--branch-color, linear-gradient(90deg, var(--primary), var(--secondary)));
        border-radius: 24px 24px 0 0;
    }
    
    /* Enhanced Metrics */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
        padding: 28px 24px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        margin: 16px 0 !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-6px) !important;
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.12) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 50%, var(--success) 100%);
        background-size: 200% 100%;
        animation: shimmer 4s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: -200% 0; }
        50% { background-position: 200% 0; }
    }
    
    /* Metric Values */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 2.75rem !important;
        font-weight: 900 !important;
        line-height: 1 !important;
        margin: 12px 0 8px 0 !important;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 8px !important;
        text-transform: none !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        padding: 8px 14px !important;
        border-radius: 14px !important;
        margin-top: 8px !important;
        backdrop-filter: blur(10px);
    }
    
    /* Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        padding: 10px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 2rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 16px 32px;
        background: transparent;
        border-radius: 14px;
        color: var(--text-secondary);
        font-weight: 700;
        font-size: 1rem;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        color: var(--primary);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--white) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        transform: translateY(-3px);
    }
    
    .stTabs [aria-selected="true"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    /* Charts */
    .stPlotlyChart > div {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(25px) !important;
        border-radius: 24px !important;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1) !important;
        padding: 28px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        transition: all 0.3s ease !important;
        margin: 20px 0 !important;
    }
    
    .stPlotlyChart > div:hover {
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.15) !important;
        transform: translateY(-4px) !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        overflow: hidden !important;
        backdrop-filter: blur(20px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 16px 32px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 45px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(15px) !important;
    }
    
    /* Performance Indicators */
    .performance-ring {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        margin: 0 auto 24px auto;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    .performance-ring::before {
        content: '';
        width: 120px;
        height: 120px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 50%;
        position: absolute;
        box-shadow: inset 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .performance-score {
        position: relative;
        z-index: 1;
        font-size: 2rem;
        font-weight: 900;
        color: var(--text-primary);
        text-align: center;
    }
    
    /* Branch Comparison Cards */
    .branch-comparison-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .branch-comparison-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    }
    
    .branch-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .branch-badge {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status Indicators */
    .status-excellent { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-good { background: rgba(59, 130, 246, 0.15); color: var(--primary); border: 1px solid rgba(59, 130, 246, 0.3); }
    .status-warning { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-danger { background: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.3); }
    
    /* Hide Sidebar */
    [data-testid='stSidebar'] { display: none !important; }
    
    /* File Upload Styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 2px dashed rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(59, 130, 246, 0.6);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Success/Error Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Floating Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    
    .floating {
        animation: float 8s ease-in-out infinite;
    }
    
    /* Progress Bars */
    .progress-bar {
        width: 100%;
        height: 12px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 6px;
        overflow: hidden;
        position: relative;
        margin: 8px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 1s ease;
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: progress-shine 2s ease-in-out infinite;
    }
    
    @keyframes progress-shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# DATA PROCESSING
# =========================
class BranchDataProcessor:
    """Process Excel data with multiple branch sheets"""
    
    @staticmethod
    def load_excel_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load data from uploaded Excel file"""
        try:
            # This would be called when file is uploaded
            # For now, we'll simulate the data structure
            return pd.DataFrame(), {}
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return pd.DataFrame(), {}
    
    @staticmethod
    def process_branch_data(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process and combine branch data"""
        combined_data = []
        
        for sheet_name, df in excel_data.items():
            if df.empty:
                continue
            
            # Standardize column names
            df_clean = df.copy()
            df_clean.columns = df_clean.columns.str.strip()
            
            # Map columns to standard names
            column_mapping = {
                "Date": "Date",
                "Day": "Day",
                "Sales Target": "SalesTarget", 
                "Sales Achivement": "SalesActual",
                "Sales Achievement": "SalesActual",  # Alternative spelling
                "Sales %": "SalesPercent",
                "NOB Target": "NOBTarget",
                "NOB Achievemnet": "NOBActual", 
                "NOB Achievement": "NOBActual",  # Alternative spelling
                "NOB %": "NOBPercent",
                " NOB %": "NOBPercent",  # Handle space
                "ABV Target": "ABVTarget",
                "ABV Achievement": "ABVActual",
                " ABV Achievement": "ABVActual",  # Handle space
                "ABV %": "ABVPercent"
            }
            
            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in df_clean.columns:
                    df_clean.rename(columns={old_name: new_name}, inplace=True)
            
            # Add branch information
            df_clean['Branch'] = sheet_name
            df_clean['BranchName'] = config.BRANCHES.get(sheet_name, {}).get('name', sheet_name)
            df_clean['BranchCode'] = config.BRANCHES.get(sheet_name, {}).get('code', '000')
            
            # Convert data types
            if 'Date' in df_clean.columns:
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            
            # Clean numeric columns
            numeric_cols = ['SalesTarget', 'SalesActual', 'SalesPercent', 'NOBTarget', 
                          'NOBActual', 'NOBPercent', 'ABVTarget', 'ABVActual', 'ABVPercent']
            
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            combined_data.append(df_clean)
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            return result
        else:
            return pd.DataFrame()

# =========================
# ANALYTICS ENGINE
# =========================
class BranchAnalytics:
    """Advanced branch analytics"""
    
    @staticmethod
    def calculate_branch_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive branch KPIs"""
        if df.empty:
            return {}
        
        kpis = {}
        
        # Overall totals
        kpis['total_sales_target'] = df['SalesTarget'].sum()
        kpis['total_sales_actual'] = df['SalesActual'].sum()
        kpis['total_sales_variance'] = kpis['total_sales_actual'] - kpis['total_sales_target']
        kpis['overall_sales_percent'] = (kpis['total_sales_actual'] / kpis['total_sales_target'] * 100) if kpis['total_sales_target'] > 0 else 0
        
        kpis['total_nob_target'] = df['NOBTarget'].sum()
        kpis['total_nob_actual'] = df['NOBActual'].sum()
        kpis['overall_nob_percent'] = (kpis['total_nob_actual'] / kpis['total_nob_target'] * 100) if kpis['total_nob_target'] > 0 else 0
        
        kpis['avg_abv_target'] = df['ABVTarget'].mean()
        kpis['avg_abv_actual'] = df['ABVActual'].mean()
        kpis['overall_abv_percent'] = (kpis['avg_abv_actual'] / kpis['avg_abv_target'] * 100) if kpis['avg_abv_target'] > 0 else 0
        
        # Branch performance
        branch_performance = df.groupby('BranchName').agg({
            'SalesTarget': 'sum',
            'SalesActual': 'sum',
            'SalesPercent': 'mean',
            'NOBTarget': 'sum',
            'NOBActual': 'sum',
            'NOBPercent': 'mean',
            'ABVTarget': 'mean',
            'ABVActual': 'mean',
            'ABVPercent': 'mean'
        }).round(2)
        
        kpis['branch_performance'] = branch_performance
        
        # Time analysis
        if df['Date'].notna().any():
            kpis['date_range'] = {
                'start': df['Date'].min(),
                'end': df['Date'].max(),
                'days': df['Date'].dt.date.nunique()
            }
            
            # Daily averages
            kpis['avg_daily_sales'] = kpis['total_sales_actual'] / kpis['date_range']['days']
            kpis['avg_daily_nob'] = kpis['total_nob_actual'] / kpis['date_range']['days']
        
        # Performance scoring
        kpis['performance_score'] = BranchAnalytics.calculate_overall_score(kpis)
        
        return kpis
    
    @staticmethod
    def calculate_overall_score(kpis: Dict[str, Any]) -> float:
        """Calculate weighted performance score"""
        score = 0
        weights = 0
        
        # Sales achievement (40% weight)
        if kpis.get('overall_sales_percent', 0) > 0:
            sales_score = min(kpis['overall_sales_percent'] / 100, 1.2)  # Cap at 120%
            score += sales_score * 40
            weights += 40
        
        # NOB achievement (35% weight)
        if kpis.get('overall_nob_percent', 0) > 0:
            nob_score = min(kpis['overall_nob_percent'] / 100, 1.2)
            score += nob_score * 35
            weights += 35
        
        # ABV achievement (25% weight)
        if kpis.get('overall_abv_percent', 0) > 0:
            abv_score = min(kpis['overall_abv_percent'] / 100, 1.2)
            score += abv_score * 25
            weights += 25
        
        return (score / weights * 100) if weights > 0 else 0

# =========================
# VISUALIZATION ENGINE
# =========================
class BranchVisualization:
    """Create stunning branch visualizations"""
    
    @staticmethod
    def create_branch_comparison_chart(branch_data: pd.DataFrame) -> go.Figure:
        """Beautiful branch comparison chart"""
        if branch_data.empty:
            return go.Figure()
        
        branches = branch_data.index
        
        fig = go.Figure()
        
        # Sales Achievement bars
        fig.add_trace(go.Bar(
            x=branches,
            y=branch_data['SalesPercent'],
            name='Sales Achievement %',
            marker=dict(
                color=branch_data['SalesPercent'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Performance %", x=1.02)
            ),
            text=[f"{val:.1f}%" for val in branch_data['SalesPercent']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Sales: %{y:.1f}%<br>Target: SAR %{customdata:,.0f}<extra></extra>',
            customdata=branch_data['SalesTarget']
        ))
        
        # Target line
        fig.add_hline(
            y=100, 
            line_dash="dash", 
            line_color="#ef4444", 
            line_width=3,
            annotation_text="100% Target",
            annotation_position="top right"
