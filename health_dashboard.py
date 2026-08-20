import os
import os.path
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import re
from plotly.subplots import make_subplots

import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Define the directory and file path for the dataset
datadir = "data"
# Load the dataset into a DataFrame
original_df = pd.read_csv(os.path.join(datadir, "nsduh_subset.csv"))

df = original_df[[
    "IRSEX", "CATAG6", "NEWRACE2", "EDUHIGHCAT", "INCOME", "COUTYP4",  # Demographics
    "MHTRTPY", "MHTSEEKTX", "MHTSKTHPY", "MHTNSEEKPY",  # Access
    "MHTUNCOST", "MHTUNNOHLP", "MHTUNCARE", "MHTUNWHER", "MHTUNTIME", "MHTUNPRIV", "MHTUNPTHNK", "MHTUNHNDL", # Barriers
    "SMIPY", "AMIPY", "LMIPY", "MMIPY", "AMDEYR", "YMDEYR", "SPDPSTMON"  # Mental health need
]].copy()

# Rename columns for easier handling
df.rename(columns={
    "IRSEX": "Gender",
    "CATAG6": "Age",
    "NEWRACE2": "Race",
    "EDUHIGHCAT": "Education",
    "INCOME": "Income",
    "COUTYP4": "MetroStatus",
    "MHTRTPY": "ReceivedTreatment",
    "MHTSEEKTX": "SoughtNoTreatment",
    "MHTSKTHPY": "ThoughtNoTreatment",
    "MHTNSEEKPY": "UnmetNeed",
    "MHTUNCOST": "BarrierCost",
    "MHTUNNOHLP": "BarrierNoHelp",
    "MHTUNCARE": "BarrierNoCare",
    "MHTUNWHER": "BarrierDontKnowWhere",
    "MHTUNTIME": "BarrierTime",
    "MHTUNPRIV": "BarrierPrivacy",
    "MHTUNPTHNK": "BarrierStigma",
    "MHTUNHNDL": "BarrierSelf",
    "SMIPY": "SeriousMI",
    "AMIPY": "AnyMI",
    "LMIPY": "MildMI",
    "MMIPY": "ModerateMI",
    "AMDEYR": "AdultMajorDepressiveEpisode",
    "YMDEYR": "YouthMajorDepressiveEpisode",
    "SPDPSTMON": "SPDPastMonth" 
}, inplace=True)

def recode_binary(col):
    return col.apply(lambda x: 
        1 if x == 1 
        else 0 if x in (0, 2) 
        else np.nan
    )

binary_vars = ["ReceivedTreatment", "SoughtNoTreatment", "ThoughtNoTreatment",
               "UnmetNeed", "BarrierCost", "BarrierNoHelp", "BarrierNoCare",
               "BarrierDontKnowWhere", "BarrierTime", "BarrierPrivacy",
               "BarrierStigma", "BarrierSelf",
               "SeriousMI", "AnyMI", "MildMI", "ModerateMI", "AdultMajorDepressiveEpisode",
               "YouthMajorDepressiveEpisode", "SPDPastMonth"]

for var in binary_vars:
    df[var] = recode_binary(df[var])

race_map = {
    1: "White",
    2: "Black",
    3: "Native American/Alaska Native",
    4: "Native Hawaiian/Pacific Islander",
    5: "Asian",
    6: "Multiracial",
    7: "Hispanic"
}
df["Race"] = df["Race"].replace(race_map)


income_map = {
    1: "Less than $20,000",
    2: "$20,000 - $49,999",
    3: "$50,000 - $74,999",
    4: "$75,000 or More"
}
df["Income"] = df["Income"].replace(income_map)
edu_map = {
    1: "Less than High School",
    2: "High School Grad",
    3: "Some College",
    4: "College Graduate"
}
df["Education"] = df["Education"].replace(edu_map)

metro_map = {
    1: "Large Metro",
    2: "Small Metro",
    3: "Nonmetro",
    4: "Unknown"
}
df["MetroStatus"] = df["MetroStatus"].replace(metro_map)

gender_map = {
    1: "Male",
    2: "Female"
}
df["Gender"] = df["Gender"].replace(gender_map)

age_map = {
    1: "12-17",
    2: "18-25",
    3: "26-34",
    4: "35-49",
    5: "50-64",
    6: "65+"
}
df["Age"] = df["Age"].replace(age_map)

df.dropna(subset=["Age", "Gender", "Race", "Income", "MetroStatus"], inplace=True)

def age_sort_key(age_str):
    # grab everything before the dash, strip any '+' and cast to int
    start = age_str.split('-')[0].replace('+', '')
    return int(start)

# compute your sorted list once
age_groups = df['Age'].dropna().unique()
age_groups = sorted(age_groups, key=age_sort_key)

race_groups = sorted(df['Race'].dropna().unique())

def income_sort_key(s):
    # find the first number (like "20,000" or "75,000")
    m = re.search(r'(\d{1,3}(?:,\d{3})*)', s)
    if m:
        return int(m.group(1).replace(',', ''))
    # anything else (e.g. “Less than…”) falls to zero
    return 0

income_groups = sorted(df['Income'].dropna().unique(), key=income_sort_key)

area_order = ['Large Metro', 'Small Metro', 'Nonmetro']
area_groups = sorted(
    df['MetroStatus'].dropna().unique(),
    key=lambda x: area_order.index(x)
)

sorted_genders = sorted(df['Gender'].dropna().unique())

# Change the scale of the graphs
scale = 95

about_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("About this Dashboard")),
        dbc.ModalBody(
            dcc.Markdown(
                """
                **Integrated Mental Health Dashboard**  
                
                This dashboard allows users to explore disparities in mental health in the United States by demographics based on assigned gender at birth, age, race, income, and living area.  
                
                **Data Source:** 2023 National Survey on Drug Use and Health (NSDUH) from the Substance Abuse and Mental Health Services Administration (SAMHSA) \n
                **Programming Languages and Libraries Used:** Python, HTML, CSS, Plotly, Dash  \n
                **Author:** Bach Nguyen \n
                **Contact:** nguyen_b3@denison.edu
                """
            )
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="btn-close-abt", className="ms-auto", n_clicks=0)
        ),
    ],
    id="modal-abt",
    is_open=False,
    size="lg",
)

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Integrated Mental Health Dashboard"

# Layout
app.layout = dbc.Container([
    dcc.Store(id='tab_selector', data='tab_home'),
    html.Div([
        html.Div([
            html.H1([
                html.Span("Welcome"),
                html.Br(),
                html.Span("to the Mental Health Disparities Dashboard")
            ]),
            html.P("This interactive dashboard allows you to explore mental health disparities in the United States")
        ], style={"vertical-alignment": "top", "height": 260}),
        about_modal,
        dbc.Button(
            "Home",
            id="btn-home",
            color="info",
            outline=True,
            n_clicks=0,
            className="w-100 mb-2",   # full width, small bottom margin
        ),
        dbc.Button(
            "Summary Statistics",
            id="btn-summary",
            color="info",
            outline=True,
            n_clicks=0,
            className="w-100 mb-2",   # full width, a bit more space before row 2
        ),
        dbc.Button(
            "About",
            id="btn-abt",
            color="info",
            outline=True,
            n_clicks=0,
            className="w-100 mb-2",   
        ),
        html.Div([
            html.Div(
                dbc.RadioItems(
                    className='btn-group',
                    inputClassName='btn-check',
                    labelClassName="btn btn-outline-info",
                    labelCheckedClassName="btn btn-info",
                    options=[
                        {"label": "Access", "value": "tab_access"},
                        {"label": "Barriers", "value": "tab_barriers"},
                        {"label": "Gap", "value": "tab_gap"},
                        {"label": "Region", "value": "tab_geo"}
                    ],
                    id='tab_secondary',
                    value=None,
                    inline=True,
                    style={'width': '100%'}
                ), style={'width': 100}
            )
        ], style={'margin-left': 10, 'margin-right': 15, 'display': 'flex',}),

        html.Div([
            html.Div([
                html.H2('Select Gender:'),
                dbc.InputGroup([
                dbc.InputGroupText("Gender"),
                dbc.Select(
                    id="gender_filter",
                    options=[{"label": g, "value": g} for g in sorted_genders],
                    placeholder="Select…",
                    value=None,
                    ),
                ], className="mb-3", style={"width": "325px"}),
            ]),

            html.Div([
                html.H2('Select Age Group:'),
                dbc.InputGroup([
                    dbc.InputGroupText("Age Group"),
                    dbc.Select(
                        id="age_filter",
                        options=[{"label": a, "value": a} for a in age_groups],
                        placeholder="Select…",
                        value=None,
                    ),
                ], className="mb-3", style={"width": "325px"}),
            ]),

            html.Div([
                html.H2('Select Race:'),
                dbc.InputGroup([
                    dbc.InputGroupText("Race Group"),
                    dbc.Select(
                        id="race_filter",
                        options=[{"label": a, "value": a} for a in race_groups],
                        placeholder="Select…",
                        value=None,
                    ),
                ], className="mb-3", style={"width": "325px"}),
            ]),

            html.Div([
                html.H2('Select Income Level:'),
                dbc.InputGroup([
                    dbc.InputGroupText("Income Group"),
                    dbc.Select(
                        id="income_filter",
                        options=[{"label": a, "value": a} for a in income_groups],
                        placeholder="Select…",
                        value=None,
                    ),
                ], className="mb-3", style={"width": "325px"}),

            ]),
            html.Div([
                html.H2('Select Region Type:'),
                dbc.InputGroup([
                    dbc.InputGroupText("Region Group"),
                    dbc.Select(
                        id="region_filter",
                        options=[{"label": a, "value": a} for a in area_groups],
                        placeholder="Select…",
                        value=None,
                    ),
                ], className="mb-3", style={"width": "325px"}),
            ]),

            dbc.Button(
                "Clear Filters",
                id="btn-clear-filters",
                color="secondary",
                outline=True,
                className="mt-3"
            )
        ], style={'margin-left': 15, 'margin-right': 15, 'margin-top': 30})

    ], style={
        'width': 340,
        'margin-left': 35,
        'margin-top': 35,
        'margin-bottom': 35
    }, className='dashboard-sidebar'),

    html.Div([
        html.Div(
            dcc.Graph(id='main_graph'),
            style={'width': 2000,
                   'height': 4000}
        )
    ], style={
        'width': 990,
        'margin-top': 35,
        'margin-right': 35,
        'margin-bottom': 35,
        'display': 'flex'
    })
], fluid=True, style={'display': 'flex'}, className='dashboard-container')

@app.callback(
    Output('tab_selector', 'data'),
    [
        Input('btn-home',    'n_clicks'),
        Input('btn-summary', 'n_clicks'),
        Input('tab_secondary', 'value'),
    ],
    prevent_initial_call=True
)
def merge_tab_inputs(n_home, n_summary, secondary_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    btn = ctx.triggered[0]['prop_id'].split('.')[0]

    if btn == 'btn-home':
        return 'tab_home'
    if btn == 'btn-summary':
        return 'tab_summary'
    if btn == 'tab_secondary':
        return secondary_value

    raise PreventUpdate

@app.callback(
    Output('tab_secondary', 'value'),
    [ Input('btn-home',    'n_clicks'),
      Input('btn-summary', 'n_clicks') ],
    prevent_initial_call=True
)
def clear_secondary(home_clicks, summary_clicks):
    # whenever you hit Home or Summary, un‐select the radios
    ctx = callback_context.triggered[0]['prop_id'].split('.')[0]
    if ctx in ('btn-home','btn-summary'):
        return None
    # shouldn't get here
    raise PreventUpdate

@app.callback(
    Output("modal-abt", "is_open"),
    [ Input("btn-abt",       "n_clicks"),
      Input("btn-close-abt", "n_clicks") ],
    [ State("modal-abt", "is_open") ]
)
def toggle_about_modal(open_clicks, close_clicks, is_open):
    # if either button was clicked, flip `is_open`
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    [
        Output('gender_filter',   'value'),
        Output('age_filter', 'value'),
        Output('race_filter',   'value'),
        Output('income_filter', 'value'),
        Output('region_filter', 'value'),
    ],
    Input('btn-clear-filters', 'n_clicks'),
    prevent_initial_call=True
)
def clear_all_filters(n_clicks):
    # Whenever the button is clicked, set all five dropdowns back to None
    return None, None, None, None, None

@app.callback(
    Output("main_graph", "figure"),
    Input("tab_selector", "data"),
    Input("gender_filter",   "value"),
    Input("age_filter",   "value"),
    Input("race_filter",   "value"),
    Input("income_filter", "value"),
    Input("region_filter", "value"),
)

def update_graph(tab, gender_filter, age_filter, race_filter, income_filter, region_filter):
    filtered_df = df.copy()
    if gender_filter:
        filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
    if age_filter:
        filtered_df = filtered_df[filtered_df['Age'] == age_filter]
    if race_filter:
        filtered_df = filtered_df[filtered_df['Race'] == race_filter]
    if income_filter:
        filtered_df = filtered_df[filtered_df['Income'] == income_filter]
    if region_filter:
        filtered_df = filtered_df[filtered_df['MetroStatus'] == region_filter]

    if tab == "tab_home":
        df_treemap = filtered_df.groupby(["Race", "Income"]).agg({
            "ReceivedTreatment": ["mean", "count"]
        }).reset_index()
        df_treemap.columns = ["Race", "Income", "TreatmentRate", "Count"]

        fig = px.treemap(
            df_treemap,
            path=["Race", "Income"],
            values="Count",
            custom_data = ["Race", "Income", "TreatmentRate", "Count"],
            color="TreatmentRate",
            color_continuous_scale="Blues"
        )

        fig.update_traces(hovertemplate =
                                        "<b>Race: </b><br>" +
                                        "<b>%{customdata[0]}</b><br><br>" +
                                        "Income Group: %{customdata[1]}<br>" +
                                        "Treatment Rate: %{customdata[2]:,.2f}<br>" +
                                        "Count: %{customdata[3]:,.0f}" +
                                        "<extra></extra>",
                                        texttemplate = "%{customdata[1]}<br>" +
                                                    "<b>%{customdata[2]:,.2f}</b><br>",
                                        textfont=dict(
                                            size=14,  
                                            family="Poppins"
                                    )
        )
        fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        },
                        font_family = "Poppins",
                        width = 16 * scale,
                        height = 9 * scale,
                        title=dict(
                        text = "<b>Treatment Rate by Race and Income</b>",
                        font = dict(
                            size=28,
                            family = "Poppins"),
                        x = 0.5))
        
        return fig
    elif tab == "tab_summary":
        # Summary stats
        total = len(filtered_df)
        if total == 0:
            return go.Figure()

        pct_treated = filtered_df["ReceivedTreatment"].mean() * 100
        pct_serious_mi = filtered_df["SeriousMI"].mean() * 100
        pct_unmet = filtered_df["UnmetNeed"].mean() * 100
        pct_ami = filtered_df["AnyMI"].mean() * 100            # AMIPY
        pct_lmi = filtered_df["MildMI"].mean() * 100            # LMIPY
        pct_mmi = filtered_df["ModerateMI"].mean() * 100            # MMIPY
        pct_amde = filtered_df["AdultMajorDepressiveEpisode"].mean() * 100            # Adult MDE
        pct_ymde = filtered_df["YouthMajorDepressiveEpisode"].mean() * 100            # Youth MDE
        pct_spd = filtered_df["SPDPastMonth"].mean() * 100

        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_treated,
            number={"suffix": "%"},
            title={"text": "Received Treatment"},
            domain={"x": [0, 0.33], "y": [0.44, 0.77]}
        ))
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_serious_mi,
            number={"suffix": "%"},
            title={"text": "Serious Mental Illness"},
            domain={"x": [0.33, 0.66], "y": [0.44, 0.77]}
        ))
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_unmet,
            number={"suffix": "%"},
            title={"text": "Unmet Need"},
            domain={"x": [0.66, 1], "y": [0.44, 0.77]}
        ))

        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_ami,
            number={"suffix": "%"},
            title={"text": "Any Mental Illness"},
            domain={"x": [0, 0.33], "y": [0.31, 0.42]}
        ))
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_lmi,
            number={"suffix": "%"},
            title={"text": "Mild Mental Illness"},
            domain={"x": [0.33, 0.66], "y": [0.31, 0.42]}
        ))
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_mmi,
            number={"suffix": "%"},
            title={"text": "Moderate Mental Illness"},
            domain={"x": [.66, 1], "y": [0.31, 0.42]}
        ))
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_amde,
            number={"suffix": "%"},
            title={"text": "Adult MDE"},
            domain={"x": [0, 0.33], "y": [0.11, 0.22]}
        ))
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_ymde,
            number={"suffix": "%"},
            title={"text": "Youth MDE"},
            domain={"x": [0.33, 0.66], "y": [0.11, 0.22]}
        ))
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=pct_spd,
            number={"suffix": "%"},
            title={"text": "Adult SPD"},
            domain={"x": [.66, 1], "y": [0.11, 0.22]}
        ))

        fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                },
                font_family = "Poppins",
                width = 16 * scale,
                height = 9 * scale,
                grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
                title=dict(
                        text = "<b>Summary Statistics for Selected Population</b>",
                        font = dict(
                            size=28,
                            family = "Poppins"),
                        x = 0.5,
                        y = 0.8),
                annotations=[
                    dict(
                        text="<i>Notes. MDE = Major Depressive Episode, SPD = Serious Psychological Distress, Youth are people from 12-17, Adult are people from 18 and above.</i>",  
                        x=0.5,
                        y=0.02,                     
                        font=dict(size=16, family="Poppins"),
                        showarrow = False
                    )]
        )

        return fig
    
    elif tab == "tab_access":
        access_counts = filtered_df[["ReceivedTreatment", "SoughtNoTreatment", "ThoughtNoTreatment"]].apply(
            pd.Series.value_counts, normalize=True).T
        barrier_cols = ["ReceivedTreatment", "SoughtNoTreatment", "ThoughtNoTreatment"]
        access_long = (
        access_counts
        .reset_index()
        .melt(
            id_vars="index",
            value_vars=[0, 1],
            var_name="Response",
            value_name="Proportion"
        )
        .rename(columns={"index": "Group"})
    )

        # 3. map 0/1 → No/Yes
        access_long["Response"] = access_long["Response"].map({0: "No", 1: "Yes"})

        # 4. make the bar chart
        fig = px.bar(
            access_long,
            x="Group",
            y="Proportion",
            color="Response",
            barmode="group",
            color_discrete_sequence=["#ae2012", "#005f73"],   # custom palette: [No, Yes]
            labels={
                "Group": "",
                "Proportion": "Proportion",
            }
        )
        fig.update_xaxes(
                        tickvals   = barrier_cols,          # the _original_ categories
                        ticktext  = [
                            "Received treatment",
                            "Didn't seek treatment",
                            "Didn't think of treatment"
                        ],
                        tickfont = dict(
                            family = "Poppins",
                            size   = 16
                        )
                    )
    
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)

        # 6. custom hovertemplate
        fig.update_traces(
            hovertemplate=(
                "%{x}<br>" +                          # the Group label
                "Proportion: %{y:.2f}<extra></extra>",
            ),
            # for safety, ensure textfont stays consistent:
            textfont=dict(family="Poppins", size=14),
            texttemplate = "<b>%{y:.2f}</b><br>"
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Poppins",
            title=dict(
                text="<b>Access to Mental Health Services (Proportion)</b>",
                font=dict(size=28, family="Poppins"),
                x=0.5
            ),
            showlegend=True,
            width=16*scale,
            height=9*scale
        )

        return fig

    elif tab == "tab_barriers":
        barrier_cols = ["BarrierNoHelp", "BarrierNoCare",
                        "BarrierDontKnowWhere", "BarrierCost", "BarrierTime", "BarrierPrivacy",
                        "BarrierStigma", "BarrierSelf"]
        custom_colors = ["#ae2012", "#bb3e03", "#ca6702", "#ee9b00",
                         "#94d2bd", "#0a9396", "#005f73", "#001219"]
        barrier_data = filtered_df[barrier_cols].apply(pd.Series.value_counts).T[1]
        fig = px.bar(barrier_data, color = barrier_cols,
                     color_discrete_sequence=custom_colors,
                     labels={"value": "Number of People Saying Yes",
                             "index": ""})
        fig.update_xaxes(
                        tickvals   = barrier_cols,          # the _original_ categories
                        ticktext  = [
                            "Doesn't think <br>it helps",
                            "Doesn't care",
                            "Doesn't know <br>where to go to",
                            "Cost",
                            "Time",
                            "Privacy",
                            "Stigma <br>(Worried what <br>people would <br>think/say)",
                            "Thought could <br>handle on <br>their own"
                        ],
                        tickfont = dict(
                            family = "Poppins",
                            size   = 16
                        )
                    )
        fig.update_traces(hovertemplate =
                                        "%{x}<br>" +       
                                        "Count: %{y}<extra></extra>",
                                        texttemplate = "<b>%{y}</b><br>",
                                        textfont=dict(
                                            size=14,  
                                            family="Poppins"
                                    )
        )
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)
        fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        },
                        font_family = "Poppins",
                        title=dict(
                        text = "<b>Reported Barriers to Mental Health Access</b>",
                        font = dict(
                            size=28,
                            family = "Poppins"),
                        x = 0.5),
                        yaxis=dict(tickfont=dict(size=12)),
                        showlegend = False)
        return fig

    elif tab == "tab_gap":
        gap_data = filtered_df.groupby("Race")[["SeriousMI", "ReceivedTreatment"]].mean().reset_index()
        fig = px.scatter(gap_data, x="SeriousMI", y="ReceivedTreatment", color="Race", size="SeriousMI",
                         labels={"SeriousMI": "Serious Mental Illness Rate",
                                 "ReceivedTreatment": "Treatment Rate"},
                        custom_data= ["Race", "SeriousMI", "ReceivedTreatment"])
        fig.update_traces(hovertemplate =
                                        "<b>Race: </b><br>" +
                                        "<b>%{customdata[0]}</b><br><br>" +
                                        "Mentall Illness Rate: %{customdata[1]:,.2f}<br>" +
                                        "Treatment Rate: %{customdata[2]:,.2f}<br>" +
                                        "<extra></extra>",
                                        textfont=dict(
                                            size=14,  
                                            family="Poppins"
                                    ))
        fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)
        fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        },
                        font_family = "Poppins",
                        title=dict(
                        text = "<b>Gap Between Mental Health Need and Treatment By Race</b>",
                        font = dict(
                            size=28,
                            family = "Poppins"),
                        x = 0.5))
        return fig

    elif tab == "tab_geo":
        region_df = filtered_df.groupby("MetroStatus").agg({
            "ReceivedTreatment": "mean",
            "UnmetNeed": "mean"
        }).reset_index()
        custom_colors = ["#ae2012", "#ee9b00", "#0a9396"]
        fig = px.bar(region_df, x="MetroStatus", y="ReceivedTreatment", 
                     category_orders={
                        "MetroStatus": [
                            "Nonmetro",
                            "Small Metro",
                            "Large Metro"
                        ]
                    },
                     color="MetroStatus", color_discrete_sequence=custom_colors,
                     labels={"MetroStatus": "Region Group",
                            "ReceivedTreatment": "Treatment Rate"})
        fig.update_xaxes(tickfont = dict(
                            family = "Poppins",
                            size   = 16
                        )
                    )
        fig.update_traces(hovertemplate =
                                        "Region Group: %{x}<br>" +       
                                        "Treatment Rate: %{y:,.2f}<extra></extra>",
                                        texttemplate = "<b>%{y:,.2f}</b><br>",
                                        textfont=dict(
                                            size=14,  
                                            family="Poppins"
                                    ))
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)
        fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        },
                        font_family = "Poppins",
                        title=dict(
                        text = "<b>Mental Health Treatment Rates by Region Group</b>",
                        font = dict(
                            size=28,
                            family = "Poppins"),
                        x = 0.5),
                        showlegend = False)
        return fig

    return go.Figure()

if __name__ == "__main__":
    app.run(debug=True, port=8050)