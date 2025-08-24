# Sales achievement bars
    fig.add_trace(go.Bar(
        x=branch_data.index,
        y=branch_data['SalesPercent'],
        name='Sales Achievement %',
        marker=dict(
            color=branch_data['SalesPercent'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=120,
            showscale=True,
            colorbar=dict(title="Achievement %", x=1.02, thickness=15)
        ),
        text=[f"{val:.1f}%" for val in branch_data['SalesPercent']],
        textposition='outside',
        textfont=dict(size=12, color='#0f172a', family='Inter'),
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
        annotation_position="top right",
        annotation=dict(font=dict(color="#ef4444", size=12, family="Inter"))
    )
    
    # Excellence line at 95%
    fig.add_hline(
        y=95,
        line_dash="dot", 
        line_color="#10b981", 
        line_width=2,
        annotation_text="🌟 Excellence",
        annotation_position="bottom right",
        annotation=dict(font=dict(color="#10b981", size=11, family="Inter"))
    )
    
    fig.update_layout(
        title=dict(
            text="Branch Sales Performance Comparison",
            font=dict(size=18, family="Inter", color="#0f172a"),
            x=0.5
        ),
        xaxis=dict(
            title="Branch",
            titlefont=dict(size=14, family="Inter"),
            tickfont=dict(size=12, family="Inter")
        ),
        yaxis=dict(
            title="Achievement (%)",
            titlefont=dict(size=14, family="Inter"),
            tickfont=dict(size=12, family="Inter"),
            gridcolor="rgba(0,0,0,0.05)"
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=460,
        showlegend=False,
        font=dict(family="Inter")
    )
    
    return fig

def create_radar_chart(branch_data: pd.DataFrame) -> go.Figure:
    """Create multi-metric radar chart"""
    if branch_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    metrics = ['SalesPercent', 'NOBPercent', 'ABVPercent']
    metric_labels = ['Sales Achievement', 'NOB Achievement', 'ABV Achievement']
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    
    for i, branch in enumerate(branch_data.index):
        values = [safe_numeric(branch_data.loc[branch, metric]) for metric in metrics]
        values.append(values[0])  # Close the radar
        
        color = colors[i % len(colors)]
        fill_color = hex_to_rgba(color, 0.15)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill='toself',
            name=branch,
            line=dict(color=color, width=4),
            fillcolor=fill_color,
            marker=dict(size=8, color=color)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 120],
                tickmode='linear',
                tick0=0,
                dtick=20,
                gridcolor="rgba(0,0,0,0.1)",
                linecolor="rgba(0,0,0,0.2)"
            ),
            angularaxis=dict(
                gridcolor="rgba(0,0,0,0.1)",
                linecolor="rgba(0,0,0,0.2)"
            )
        ),
        title=dict(
            text="Branch Performance Radar - All Metrics",
            font=dict(size=18, family="Inter", color="#0f172a"),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.1,
            y=0.5,
            font=dict(family="Inter", size=12)
        ),
        height=480,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter")
    )
    
    return fig

def create_daily_trends_chart(df: pd.DataFrame) -> go.Figure:
    """Create comprehensive daily trends chart"""
    if df.empty or 'Date' not in df.columns or df['Date'].isna().all():
        return go.Figure()
    
    # Group by date and branch
    daily_data = df.groupby(['Date', 'BranchName']).agg({
        'SalesActual': 'sum',
        'NOBActual': 'sum',
        'ABVActual': 'mean'
    }).reset_index()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            '📊 Daily Sales Performance (SAR)',
            '🛍️ Number of Baskets (NOB)', 
            '💰 Average Basket Value (ABV)'
        ),
        vertical_spacing=0.08
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
                marker=dict(size=6),
                hovertemplate=f'<b>{branch}</b><br>Date: %{{x}}<br>Sales: SAR %{{y:,.0f}}<extra></extra>'
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
                showlegend=False,
                hovertemplate=f'<b>{branch}</b><br>Date: %{{x}}<br>Baskets: %{{y:,.0f}}<extra></extra>'
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
                showlegend=False,
                hovertemplate=f'<b>{branch}</b><br>Date: %{{x}}<br>ABV: SAR %{{y:.2f}}<extra></extra>'
            ),
            row=3, col=1
        )
    
    fig.update_layout(
        height=800,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#0f172a"),
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center",
            font=dict(size=12)
        )
    )
    
    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)",
        linecolor="rgba(0,0,0,0.1)"
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)",
        linecolor="rgba(0,0,0,0.1)"
    )
    
    return fig

# =========================
# TAB RENDER FUNCTIONS
# =========================
def render_overview_tab(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render beautiful overview tab"""
    
    # Overall performance metrics
    st.markdown("### 🏆 Overall Network Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Total Sales",
            format_currency(kpis.get('total_sales_actual', 0)),
            delta=f"vs Target: {kpis.get('overall_sales_percent', 0):.1f}%"
        )
    
    with col2:
        variance_color = "normal" if kpis.get('total_sales_variance', 0) >= 0 else "inverse"
        st.metric(
            "📊 Sales Variance", 
            format_currency(kpis.get('total_sales_variance', 0)),
            delta=f"{kpis.get('overall_sales_percent', 0) - 100:+.1f}%",
            delta_color=variance_color
        )
    
    with col3:
        nob_color = "normal" if kpis.get('overall_nob_percent', 0) >= 90 else "inverse"
        st.metric(
            "🛍️ Total Baskets",
            f"{kpis.get('total_nob_actual', 0):,.0f}",
            delta=f"Achievement: {kpis.get('overall_nob_percent', 0):.1f}%",
            delta_color=nob_color
        )
    
    with col4:
        abv_color = "normal" if kpis.get('overall_abv_percent', 0) >= 90 else "inverse"
        st.metric(
            "💎 Avg Basket Value",
            f"SAR {kpis.get('avg_abv_actual', 0):.2f}",
            delta=f"vs Target: {kpis.get('overall_abv_percent', 0):.1f}%",
            delta_color=abv_color
        )
    
    with col5:
        score_color = "normal" if kpis.get('performance_score', 0) >= 80 else "off"
        st.metric(
            "⭐ Performance Score",
            f"{kpis.get('performance_score', 0):.0f}/100",
            delta="Weighted Average",
            delta_color=score_color
        )
    
    # Branch performance cards
    st.markdown("### 🏪 Individual Branch Performance")
    
    if 'branch_performance' in kpis and not kpis['branch_performance'].empty:
        branch_data = kpis['branch_performance']
        
        for branch_name, metrics in branch_data.iterrows():
            # Get branch info
            branch_info = None
            for key, info in config.BRANCHES.items():
                if info['name'] == branch_name:
                    branch_info = info
                    break
            
            if branch_info is None:
                branch_info = {"color": "#6b7280", "code": "000", "icon": "🏪"}
            
            # Calculate average performance
            avg_performance = np.mean([
                safe_numeric(metrics.get('SalesPercent', 0)),
                safe_numeric(metrics.get('NOBPercent', 0)),
                safe_numeric(metrics.get('ABVPercent', 0))
            ])
            
            # Determine status
            if avg_performance >= 95:
                status_class = "excellent"
                status_text = "🏆 EXCELLENT"
            elif avg_performance >= 85:
                status_class = "good"
                status_text = "✅ GOOD"
            elif avg_performance >= 75:
                status_class = "warn"
                status_text = "⚠️ NEEDS FOCUS"
            else:
                status_class = "danger"
                status_text = "🔴 ACTION REQUIRED"
            
            # Branch card header
            st.markdown(f"""
            <div class='branch-card' style='--branch-gradient: linear-gradient(90deg, {branch_info['color']}, #22d3ee);'>
                <div class='branch-header'>
                    <span style='color: {branch_info['color']}; font-size: 1.5rem;'>{branch_info.get('icon', '🏪')}</span>
                    {branch_name} 
                    <span style='color: #64748b; font-weight: 600;'>(Code: {branch_info['code']})</span>
                    <span class='badge {status_class}' style='margin-left: auto;'>{status_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Branch metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Sales Achievement",
                    format_currency(safe_numeric(metrics.get('SalesActual', 0))),
                    delta=f"{safe_numeric(metrics.get('SalesPercent', 0)):.1f}% of target"
                )
            
            with col2:
                st.metric(
                    "Baskets (NOB)",
                    f"{safe_numeric(metrics.get('NOBActual', 0)):,.0f}",
                    delta=f"{safe_numeric(metrics.get('NOBPercent', 0)):.1f}% of target"
                )
            
            with col3:
                st.metric(
                    "Avg Basket Value",
                    f"SAR {safe_numeric(metrics.get('ABVActual', 0)):.2f}",
                    delta=f"{safe_numeric(metrics.get('ABVPercent', 0)):.1f}% of target"
                )
            
            with col4:
                st.metric(
                    "Overall Score",
                    f"{avg_performance:.1f}%",
                    delta="Average Performance"
                )
            
            # Beautiful progress bars
            progress_metrics = [
                ("Sales", safe_numeric(metrics.get('SalesPercent', 0)), '#22d3ee'),
                ("Baskets (NOB)", safe_numeric(metrics.get('NOBPercent', 0)), '#10b981'),
                ("Basket Value (ABV)", safe_numeric(metrics.get('ABVPercent', 0)), '#f59e0b')
            ]
            
            for label, percentage, gradient_color in progress_metrics:
                percentage_capped = min(percentage, 120.0)
                
                st.markdown(f"""
                <div style='margin: 16px 0;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                        <span style='font-weight: 600; color: #64748b; font-size: 0.95rem;'>{label}</span>
                        <span style='font-weight: 800; color: {branch_info['color']}; font-size: 0.95rem;'>{percentage_capped:.1f}%</span>
                    </div>
                    <div class='progress'>
                        <div style='width: {min(percentage_capped, 100)}%; background: linear-gradient(90deg, {branch_info['color']}, {gradient_color}); border-radius: 999px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Comparison charts
    st.markdown("### 📊 Branch Comparison Analytics")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if 'branch_performance' in kpis:
            comparison_chart = create_branch_comparison_chart(kpis['branch_performance'])
            st.plotly_chart(comparison_chart, use_container_width=True, config={"displayModeBar": False})
    
    with col2:
        if 'branch_performance' in kpis:
            radar_chart = create_radar_chart(kpis['branch_performance'])
            st.plotly_chart(radar_chart, use_container_width=True, config={"displayModeBar": False})

def render_daily_trends_tab(df: pd.DataFrame):
    """Render daily trends analysis"""
    st.markdown("## 📈 Daily Performance Trends")
    
    if df.empty:
        st.info("No data available for trend analysis.")
        return
    
    # Time period selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if 'Date' in df.columns and df['Date'].notna().any():
            period_options = ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"]
            selected_period = st.selectbox("📅 Time Period", period_options, index=1)
            
            # Calculate date range
            latest_date = df['Date'].max().date() if df['Date'].notna().any() else datetime.now().date()
            
            if selected_period == "Last 7 Days":
                start_date = latest_date - timedelta(days=7)
            elif selected_period == "Last 30 Days":
                start_date = latest_date - timedelta(days=30)
            elif selected_period == "Last 3 Months":
                start_date = latest_date - timedelta(days=90)
            else:
                start_date = df['Date'].min().date()
            
            # Filter data
            mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= latest_date)
            filtered_df = df.loc[mask].copy()
        else:
            filtered_df = df.copy()
            selected_period = "All Time"
    
    with col2:
        if not filtered_df.empty:
            trends_chart = create_daily_trends_chart(filtered_df)
            st.plotly_chart(trends_chart, use_container_width=True, config={"displayModeBar": False})
    
    # Weekly performance summary
    if 'Date' in filtered_df.columns and filtered_df['Date'].notna().any():
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
        
        weekly_summary['SalesAchievement'] = np.where(
            weekly_summary['SalesTarget'] > 0,
            (weekly_summary['SalesActual'] / weekly_summary['SalesTarget'] * 100).round(1),
            0
        )
        
        weekly_summary['WeekLabel'] = (weekly_summary['Year'].astype(str) + 
                                     '-W' + weekly_summary['Week'].astype(str))
        
        # Create pivot table
        pivot_data = weekly_summary.pivot(
            index='WeekLabel', 
            columns='BranchName', 
            values='SalesAchievement'
        ).fillna(0)
        
        # Format for display
        display_data = pivot_data.copy()
        for col in display_data.columns:
            display_data[col] = display_data[col].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
        
        st.dataframe(display_data, use_container_width=True)

def render_competition_tab(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render competition analysis tab"""
    st.markdown("## 🏆 Branch Competition Analysis")
    
    if 'branch_performance' not in kpis or kpis['branch_performance'].empty:
        st.info("No branch performance data available.")
        return
    
    branch_data = kpis['branch_performance']
    
    # Champion cards
    st.markdown("### 🥇 Performance Champions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sales champion
        sales_champion = branch_data['SalesPercent'].idxmax()
        sales_score = branch_data.loc[sales_champion, 'SalesPercent']
        
        st.markdown(f"""
        <div class='champion-card' style='--champion-color: linear-gradient(90deg, #10b981, #22d3ee);'>
            <div style='font-size: 3rem; margin-bottom: 16px;'>🥇</div>
            <h3 style='color: #059669; margin: 0; font-size: 1.3rem;'>Sales Champion</h3>
            <h1 style='color: #059669; margin: 12px 0; font-size: 2.2rem; font-weight: 900;'>{sales_champion}</h1>
            <h2 style='margin: 8px 0; font-size: 1.8rem; color: #0f172a;'>{sales_score:.1f}%</h2>
            <p style='color: #64748b; margin: 0; font-weight: 600;'>Target Achievement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # NOB champion
        nob_champion = branch_data['NOBPercent'].idxmax()
        nob_score = branch_data.loc[nob_champion, 'NOBPercent']
        
        st.markdown(f"""
        <div class='champion-card' style='--champion-color: linear-gradient(90deg, #3b82f6, #8b5cf6);'>
            <div style='font-size: 3rem; margin-bottom: 16px;'>🥇</div>
            <h3 style='color: #2563eb; margin: 0; font-size: 1.3rem;'>Basket Champion</h3>
            <h1 style='color: #2563eb; margin: 12px 0; font-size: 2.2rem; font-weight: 900;'>{nob_champion}</h1>
            <h2 style='margin: 8px 0; font-size: 1.8rem; color: #0f172a;'>{nob_score:.1f}%</h2>
            <p style='color: #64748b; margin: 0; font-weight: 600;'>NOB Achievement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # ABV champion
        abv_champion = branch_data['ABVPercent'].idxmax()
        abv_score = branch_data.loc[abv_champion, 'ABVPercent']
        
        st.markdown(f"""
        <div class='champion-card' style='--champion-color: linear-gradient(90deg, #f59e0b, #f97316);'>
            <div style='font-size: 3rem; margin-bottom: 16px;'>🥇</div>
            <h3 style='color: #d97706; margin: 0; font-size: 1.3rem;'>Value Champion</h3>
            <h1 style='color: #d97706; margin: 12px 0; font-size: 2.2rem; font-weight: 900;'>{abv_champion}</h1>
            <h2 style='margin: 8px 0; font-size: 1.8rem; color: #0f172a;'>{abv_score:.1f}%</h2>
            <p style='color: #64748b; margin: 0; font-weight: 600;'>ABV Achievement</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed metrics table
    st.markdown("### 📋 Comprehensive Branch Metrics")
    
    # Format the branch data for display
    display_data = branch_data.copy()
    
    # Format columns
    if 'SalesTarget' in display_data.columns:
        display_data['SalesTarget'] = display_data['SalesTarget'].apply(lambda x: format_currency(safe_numeric(x)))
    if 'SalesActual' in display_data.columns:
        display_data['SalesActual'] = display_data['SalesActual'].apply(lambda x: format_currency(safe_numeric(x)))
    if 'NOBTarget' in display_data.columns:
        display_data['NOBTarget'] = display_data['NOBTarget'].apply(lambda x: f"{safe_numeric(x):,.0f}")
    if 'NOBActual' in display_data.columns:
        display_data['NOBActual'] = display_data['NOBActual'].apply(lambda x: f"{safe_numeric(x):,.0f}")
    if 'ABVTarget' in display_data.columns:
        display_data['ABVTarget'] = display_data['ABVTarget'].apply(lambda x: f"SAR {safe_numeric(x):.2f}")
    if 'ABVActual' in display_data.columns:
        display_data['ABVActual'] = display_data['ABVActual'].apply(lambda x: f"SAR {safe_numeric(x):.2f}")
    
    # Format percentage columns
    for col in ['SalesPercent', 'NOBPercent', 'ABVPercent']:
        if col in display_data.columns:
            display_data[col] = display_data[col].apply(lambda x: format_percentage(safe_numeric(x)))
    
    # Rename columns for display
    column_renames = {
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
    
    display_data = display_data.rename(columns=column_renames)
    st.dataframe(display_data, use_container_width=True)

def render_export_tab(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render export and data management tab"""
    st.markdown("## 📥 Data Export & Analysis")
    
    if df.empty:
        st.info("No data available to export.")
        return
    
    # Data summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='branch-card'>
            <h4 style='margin: 0 0 16px 0; color: #0f172a;'>📊 Data Overview</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Active Branches", f"{df['BranchName'].nunique() if 'BranchName' in df.columns else 0}")
        if 'Date' in df.columns:
            st.metric("Date Range", f"{df['Date'].dt.date.nunique()} days")
    
    with col2:
        st.markdown("""
        <div class='branch-card'>
            <h4 style='margin: 0 0 16px 0; color: #0f172a;'>💰 Financial Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Total Sales", format_currency(kpis.get('total_sales_actual', 0)))
        st.metric("Target Sales", format_currency(kpis.get('total_sales_target', 0)))
        st.metric("Variance", format_currency(kpis.get('total_sales_variance', 0)))
    
    with col3:
        st.markdown("""
        <div class='branch-card'>
            <h4 style='margin: 0 0 16px 0; color: #0f172a;'>🎯 Performance</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Overall Score", f"{kpis.get('performance_score', 0):.0f}/100")
        st.metric("Sales Achievement", format_percentage(kpis.get('overall_sales_percent', 0)))
        st.metric("NOB Achievement", format_percentage(kpis.get('overall_nob_percent', 0)))
    
    with col4:
        st.markdown("""
        <div class='branch-card'>
            <h4 style='margin: 0 0 16px 0; color: #0f172a;'>📥 Export Options</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Excel export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # All data
            df.to_excel(writer, sheet_name='All Branches Data', index=False)
            
            # Branch summary
            if 'branch_performance' in kpis:
                kpis['branch_performance'].to_excel(writer, sheet_name='Branch Summary')
            
            # Daily summary
            if 'Date' in df.columns and df['Date'].notna().any():
                daily_summary = df.groupby(['Date', 'BranchName']).agg({
                    'SalesActual': 'sum',
                    'SalesTarget': 'sum',
                    'NOBActual': 'sum',
                    'ABVActual': 'mean'
                }).reset_index()
                daily_summary.to_excel(writer, sheet_name='Daily Summary', index=False)
        
            f"Alkhair_Branch_Analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # CSV export
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📄 Download CSV",
            csv_buffer.getvalue(),
            f"Alkhair_Branch_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Data preview section
    st.markdown("### 📋 Data Preview")
    
    # Branch selector for preview
    if 'BranchName' in df.columns:
        preview_options = ["All Branches"] + list(df['BranchName'].unique())
        selected_branch = st.selectbox(
            "Select Branch for Preview",
            options=preview_options,
            key="preview_branch"
        )
        
        if selected_branch == "All Branches":
            preview_data = df.head(50)
        else:
            preview_data = df[df['BranchName'] == selected_branch].head(50)
    else:
        preview_data = df.head(50)
    
    st.dataframe(preview_data, use_container_width=True)

# =========================
# MAIN APPLICATION
# =========================
def main():
    """Main application entry point"""
    
    # Apply beautiful styling
    apply_beautiful_css()
    
    # Stunning Hero Header
    st.markdown(f"""
    <div class='hero-section floating'>
        <div class='hero-content'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h1 style='margin: 0; font-size: 3.5rem; font-weight: 900; text-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
                        🏪 Alkhair Family Market
                    </h1>
                    <p style='margin: 16px 0 0 0; font-size: 1.4rem; opacity: 0.9; font-weight: 600;'>
                        Multi-Branch Executive Dashboard • Real-time Performance Analytics
                    </p>
                    <div style='margin-top: 24px; display: flex; gap: 16px; flex-wrap: wrap;'>
                        <span style='background: rgba(255,255,255,0.25); padding: 10px 20px; border-radius: 25px; font-size: 1rem; font-weight: 700; backdrop-filter: blur(10px);'>
                            📊 4 Branches
                        </span>
                        <span style='background: rgba(255,255,255,0.25); padding: 10px 20px; border-radius: 25px; font-size: 1rem; font-weight: 700; backdrop-filter: blur(10px);'>
                            ⚡ Live Analytics
                        </span>
                        <span style='background: rgba(255,255,255,0.25); padding: 10px 20px; border-radius: 25px; font-size: 1rem; font-weight: 700; backdrop-filter: blur(10px);'>
                            💎 Executive Intelligence
                        </span>
                    </div>
                </div>
                <div style='text-align: right; opacity: 0.9;'>
                    <div style='font-size: 1.2rem; font-weight: 800;'>📈 Executive Analytics</div>
                    <div style='font-size: 1rem; margin-top: 6px;'>Updated: {datetime.now().strftime('%H:%M:%S')}</div>
                    <div style='font-size: 0.9rem; margin-top: 4px; opacity: 0.8;'>Auto-refresh Dashboard</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Upload Your Branch Performance Data")
    
    uploaded_file = st.file_uploader(
        "Choose Excel file with branch sheets",
        type=['xlsx', 'xls'],
        help="Upload Excel file where each sheet represents a branch (Al Khair, Noora, Hamra, Magnus)",
        accept_multiple_files=False
    )
    
    if uploaded_file is None:
        # Show expected format
        st.markdown("""
        <div class='branch-card' style='text-align: center; margin-top: 40px;'>
            <h3 style='color: #0f172a; margin-bottom: 20px;'>📊 Expected Data Structure</h3>
            <p style='color: #64748b; margin-bottom: 20px;'>Your Excel file should contain multiple sheets, each representing a branch:</p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0;'>
                <div style='background: rgba(59, 130, 246, 0.1); padding: 16px; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2);'>
                    <div style='font-size: 1.5rem; margin-bottom: 8px;'>🏪</div>
                    <div style='font-weight: 700; color: #2563eb;'>Al khair - 102</div>
                </div>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 16px; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2);'>
                    <div style='font-size: 1.5rem; margin-bottom: 8px;'>🏬</div>
                    <div style='font-weight: 700; color: #059669;'>Noora - 104</div>
                </div>
                <div style='background: rgba(245, 158, 11, 0.1); padding: 16px; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.2);'>
                    <div style='font-size: 1.5rem; margin-bottom: 8px;'>🏢</div>
                    <div style='font-weight: 700; color: #d97706;'>Hamra - 109</div>
                </div>
                <div style='background: rgba(139, 92, 246, 0.1); padding: 16px; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.2);'>
                    <div style='font-size: 1.5rem; margin-bottom: 8px;'>🏭</div>
                    <div style='font-weight: 700; color: #7c3aed;'>Magnus - 107</div>
                </div>
            </div>
            <p style='color: #64748b; font-size: 0.95rem;'>
                <strong>Required columns:</strong> Date, Sales Target, Sales Achievement, NOB Target, NOB Achievement, ABV Target, ABV Achievement
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Process uploaded file
    @st.cache_data
    def load_branch_data(file):
        """Load and process branch data from Excel file"""
        try:
            excel_data = pd.read_excel(file, sheet_name=None, engine='openpyxl')
            return process_excel_sheets(excel_data)
        except Exception as e:
            st.error(f"Error processing Excel file: {str(e)}")
            return pd.DataFrame()
    
    with st.spinner("🔄 Processing branch data..."):
        df = load_branch_data(uploaded_file)
    
    if df.empty:
        st.error("❌ Could not process the uploaded file. Please check the sheet structure and column names.")
        st.stop()
    
    # Success message with data summary
    st.success(
        f"✅ Successfully loaded data for **{df['BranchName'].nunique()}** branches "
        f"with **{len(df):,}** records!"
    )
    
    # Optional date filtering
    if 'Date' in df.columns and df['Date'].notna().any():
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            start_date = st.date_input(
                "📅 Start Date",
                value=df['Date'].min().date(),
                key="global_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "📅 End Date", 
                value=df['Date'].max().date(),
                key="global_end_date"
            )
        
        # Apply date filter
        date_mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df.loc[date_mask].copy()
        
        # Show filter results
        with col3:
            if len(filtered_df) != len(df):
                st.info(f"📅 Filtered to {len(filtered_df):,} records ({len(filtered_df)/len(df)*100:.1f}% of total)")
    else:
        filtered_df = df.copy()
    
    # Calculate KPIs for filtered data
    kpis = calculate_comprehensive_kpis(filtered_df)
    
    # Beautiful Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Branch Overview",
        "📈 Daily Trends",
        "🏆 Competition Analysis", 
        "📊 Data Export"
    ])
    
    with tab1:
        render_overview_tab(filtered_df, kpis)
    
    with tab2:
        render_daily_trends_tab(filtered_df)
    
    with tab3:
        render_competition_tab(filtered_df, kpis)
    
    with tab4:
        render_export_tab(filtered_df, kpis)
    
    # Beautiful footer
    st.markdown("""
    <div style='text-align: center; padding: 80px 0 40px 0; color: rgba(255,255,255,0.8);'>
        <p style='margin: 0; font-size: 1.1rem; font-weight: 600;'>
            🏪 Alkhair Family Market • Multi-Branch Excellence Dashboard
        </p>
        <p style='margin: 12px 0 0 0; font-size: 0.95rem; opacity: 0.8;'>
            Built with ❤️ for Data-Driven Business Excellence
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        st.stop()# Alkhair Family Market — Multi-Branch Executive Dashboard (Enhanced Beautiful Version)
# Upload Excel (multiple sheets = branches) for stunning branch analytics
# Beautiful glassmorphism UI with advanced visualizations

import io
import re
import math
from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# =========================
# CONFIGURATION
# =========================
class Config:
    PAGE_TITLE = "Alkhair Family Market — Executive Branch Dashboard"
    LAYOUT = "wide"
    
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6", "icon": "🏪"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981", "icon": "🏬"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b", "icon": "🏢"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6", "icon": "🏭"},
    }
    
    TARGETS = {
        "sales_achievement": 95.0,
        "nob_achievement": 90.0,
        "abv_achievement": 90.0,
    }

config = Config()

st.set_page_config(
    page_title=config.PAGE_TITLE, 
    layout=config.LAYOUT, 
    initial_sidebar_state="collapsed",
    page_icon="🏪"
)

# =========================
# STUNNING VISUAL DESIGN
# =========================
def apply_beautiful_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Animated Gradient Background */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 600% 600%;
        animation: gradientDance 20s ease infinite;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
        min-height: 100vh;
    }
    
    @keyframes gradientDance {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        padding: 2.5rem 2rem;
        max-width: 1600px;
    }
    
    /* Glassmorphism Hero Section */
    .hero-section {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 32px;
        padding: 48px 40px;
        margin: -2rem -1rem 3rem -1rem;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.2);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced Metric Cards */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.98) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 24px !important;
        padding: 32px 28px !important;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        margin: 20px 0 !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 0 40px 80px rgba(0, 0, 0, 0.12) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #10b981 100%);
        background-size: 300% 100%;
        animation: shimmerEffect 4s ease-in-out infinite;
    }
    
    @keyframes shimmerEffect {
        0%, 100% { background-position: -200% 0; }
        50% { background-position: 200% 0; }
    }
    
    /* Metric Typography */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 3rem !important;
        font-weight: 900 !important;
        line-height: 1 !important;
        margin: 16px 0 12px 0 !important;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #374151 !important;
        margin-bottom: 12px !important;
        text-transform: none !important;
        letter-spacing: 0.02em !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border-radius: 16px !important;
        margin-top: 12px !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Beautiful Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        padding: 12px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 3rem 0 2rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 64px;
        padding: 18px 36px;
        background: transparent;
        border-radius: 16px;
        color: #64748b;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.08);
        color: #3b82f6;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #0f172a !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15) !important;
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
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    /* Chart Containers */
    .stPlotlyChart > div {
        background: rgba(255, 255, 255, 0.98) !important;
        backdrop-filter: blur(25px) !important;
        border-radius: 24px !important;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.08) !important;
        padding: 32px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        transition: all 0.3s ease !important;
        margin: 24px 0 !important;
    }
    
    .stPlotlyChart > div:hover {
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.12) !important;
        transform: translateY(-4px) !important;
    }
    
    /* Beautiful Branch Cards */
    .branch-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 24px;
        padding: 28px;
        margin: 20px 0;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.08);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .branch-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 40px 80px rgba(0, 0, 0, 0.12);
    }
    
    .branch-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: var(--branch-gradient, linear-gradient(90deg, #3b82f6, #8b5cf6));
    }
    
    .branch-header {
        font-weight: 800;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #0f172a;
        margin-bottom: 20px;
    }
    
    /* Status Badges */
    .badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        backdrop-filter: blur(10px);
    }
    
    .excellent { 
        background: rgba(16, 185, 129, 0.15); 
        color: #059669; 
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    .good { 
        background: rgba(59, 130, 246, 0.15); 
        color: #2563eb; 
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .warn { 
        background: rgba(245, 158, 11, 0.15); 
        color: #d97706; 
        border: 1px solid rgba(245, 158, 11, 0.3);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }
    
    .danger { 
        background: rgba(239, 68, 68, 0.15); 
        color: #dc2626; 
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
    }
    
    /* Enhanced Progress Bars */
    .progress {
        height: 14px;
        background: rgba(241, 245, 249, 0.8);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .progress > div {
        height: 100%;
        border-radius: 999px;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress > div::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: progressShine 2.5s ease-in-out infinite;
    }
    
    @keyframes progressShine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* File Upload Styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(25px);
        border: 3px dashed rgba(59, 130, 246, 0.4);
        border-radius: 24px;
        padding: 48px;
        text-align: center;
        transition: all 0.3s ease;
        margin: 20px 0;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(59, 130, 246, 0.7);
        background: rgba(255, 255, 255, 0.95);
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced DataFrames */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.98) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        overflow: hidden !important;
        backdrop-filter: blur(20px);
    }
    
    /* Beautiful Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 16px 32px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.25) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 45px rgba(59, 130, 246, 0.35) !important;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #22d3ee 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 16px 28px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.25) !important;
        width: 100% !important;
        margin: 8px 0 !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 45px rgba(16, 185, 129, 0.35) !important;
    }
    
    /* Form Elements */
    .stSelectbox > div > div,
    .stFileUploader,
    .stDateInput > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08) !important;
        backdrop-filter: blur(15px) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Champion Cards */
    .champion-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .champion-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.12);
    }
    
    .champion-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--champion-color);
    }
    
    /* Sidebar Hidden */
    [data-testid='stSidebar'] { display: none !important; }
    
    /* Section Headers */
    .main h1, .main h2, .main h3 {
        color: white;
        font-weight: 800;
        margin: 2rem 0 1.5rem 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main h3 {
        font-size: 1.5rem;
        position: relative;
        padding-bottom: 12px;
    }
    
    .main h3::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.3));
        border-radius: 2px;
    }
    
    /* Floating Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    
    .floating {
        animation: float 8s ease-in-out infinite;
    }
    </style>
    """,
        unsafe_allow_html=True
    )

# =========================
# UTILITY FUNCTIONS
# =========================
def safe_numeric(val: Any) -> float:
    """Safely convert to numeric"""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return 0.0
    try:
        return float(val)
    except:
        return 0.0

def format_currency(amount: float) -> str:
    """Format currency with SAR prefix"""
    return f"SAR {amount:,.0f}"

def format_percentage(pct: float) -> str:
    """Format percentage"""
    return f"{pct:.1f}%"

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba with alpha"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16) 
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# =========================
# DATA PROCESSING
# =========================
def process_excel_sheets(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Process Excel sheets into standardized format"""
    combined_data = []
    
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
    
    for sheet_name, df in excel_data.items():
        if df.empty:
            continue
        
        # Clean the dataframe
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.astype(str).str.strip()
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in df_clean.columns:
                df_clean.rename(columns={old_name: new_name}, inplace=True)
        
        # Add branch metadata
        df_clean['Branch'] = sheet_name
        branch_info = config.BRANCHES.get(sheet_name, {"name": sheet_name, "code": "000", "color": "#6b7280"})
        df_clean['BranchName'] = branch_info['name']
        df_clean['BranchCode'] = branch_info['code']
        df_clean['BranchColor'] = branch_info['color']
        
        # Convert data types
        if 'Date' in df_clean.columns:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['SalesTarget', 'SalesActual', 'SalesPercent', 'NOBTarget', 
                          'NOBActual', 'NOBPercent', 'ABVTarget', 'ABVActual', 'ABVPercent']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        combined_data.append(df_clean)
    
    if combined_data:
        result = pd.concat(combined_data, ignore_index=True)
        return result
    else:
        return pd.DataFrame()

def calculate_comprehensive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive KPIs for all branches"""
    if df.empty:
        return {}
    
    kpis = {}
    
    # Overall aggregated metrics
    kpis['total_sales_target'] = safe_numeric(df['SalesTarget'].sum())
    kpis['total_sales_actual'] = safe_numeric(df['SalesActual'].sum())
    kpis['total_sales_variance'] = kpis['total_sales_actual'] - kpis['total_sales_target']
    kpis['overall_sales_percent'] = (kpis['total_sales_actual'] / kpis['total_sales_target'] * 100) if kpis['total_sales_target'] > 0 else 0
    
    kpis['total_nob_target'] = safe_numeric(df['NOBTarget'].sum())
    kpis['total_nob_actual'] = safe_numeric(df['NOBActual'].sum())
    kpis['overall_nob_percent'] = (kpis['total_nob_actual'] / kpis['total_nob_target'] * 100) if kpis['total_nob_target'] > 0 else 0
    
    kpis['avg_abv_target'] = safe_numeric(df['ABVTarget'].mean())
    kpis['avg_abv_actual'] = safe_numeric(df['ABVActual'].mean())
    kpis['overall_abv_percent'] = (kpis['avg_abv_actual'] / kpis['avg_abv_target'] * 100) if kpis['avg_abv_target'] > 0 else 0
    
    # Branch-level performance summary
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
    if 'Date' in df.columns and df['Date'].notna().any():
        kpis['date_range'] = {
            'start': df['Date'].min(),
            'end': df['Date'].max(),
            'days': int(df['Date'].dt.date.nunique())
        }
        
        kpis['avg_daily_sales'] = kpis['total_sales_actual'] / kpis['date_range']['days'] if kpis['date_range']['days'] > 0 else 0
        kpis['avg_daily_nob'] = kpis['total_nob_actual'] / kpis['date_range']['days'] if kpis['date_range']['days'] > 0 else 0
    
    # Weighted performance score calculation
    score = 0
    weights = 0
    
    if kpis['overall_sales_percent'] > 0:
        sales_score = min(kpis['overall_sales_percent'] / 100, 1.2)
        score += sales_score * 40
        weights += 40
    
    if kpis['overall_nob_percent'] > 0:
        nob_score = min(kpis['overall_nob_percent'] / 100, 1.2)
        score += nob_score * 35
        weights += 35
    
    if kpis['overall_abv_percent'] > 0:
        abv_score = min(kpis['overall_abv_percent'] / 100, 1.2)
        score += abv_score * 25
        weights += 25
    
    kpis['performance_score'] = (score / weights * 100) if weights > 0 else 0
    
    return kpis

# =========================
# VISUALIZATION ENGINE
# =========================
def create_branch_comparison_chart(branch_data: pd.DataFrame) -> go.Figure:
    """Create beautiful branch comparison chart"""
    if branch_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Sales achievement bars
    fig.add_
