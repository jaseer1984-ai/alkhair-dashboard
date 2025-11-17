import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load Excel file
df = pd.read_excel("Daily_MTD_Dashboard.xlsx", sheet_name="Daily Input", engine="openpyxl")

# Clean data
df = df.dropna(subset=["Brand"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Summary metrics
summary = {
    "Total Sales Target": df["Sales Target"].sum(),
    "Total Sales Achieved": df["Sales Achieved"].sum(),
    "Total ABV Target": df["ABV Target"].sum(),
    "Total ABV Achieved": df["ABV Achieved"].sum(),
    "Total NOB Target": df["NOB Target"].sum(),
    "Total NOB Achieved": df["NOB Achieved"].sum(),
}

# Charts
fig_sales = px.bar(df, x="Brand", y=["Sales Target", "Sales Achieved"], barmode="group", title="Sales Target vs Achieved")
fig_abv = px.bar(df, x="Brand", y=["ABV Target", "ABV Achieved"], barmode="group", title="ABV Target vs Achieved")
fig_nob = px.bar(df, x="Brand", y=["NOB Target", "NOB Achieved"], barmode="group", title="NOB Target vs Achieved")

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Daily MTD Dashboard"),
    html.Div([
        html.H3("Summary Metrics"),
        html.Ul([html.Li(f"{k}: {v}") for k, v in summary.items()])
    ]),
    dcc.Graph(figure=fig_sales),
    dcc.Graph(figure=fig_abv),
    dcc.Graph(figure=fig_nob)
])

if __name__ == "__main__":
    app.run_server(debug=True)
