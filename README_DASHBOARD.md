# SpaceX Launch Records Dashboard

An interactive Plotly Dash dashboard for analyzing SpaceX launch data in real-time.

## Features

This dashboard application includes:

1. **Launch Site Dropdown** - Select specific launch sites or view all sites
2. **Success Pie Chart** - Visualize success rates by site or overall success/failure rates
3. **Payload Range Slider** - Filter data by payload mass (0-10,000 kg)
4. **Success-Payload Scatter Chart** - Analyze correlation between payload and launch success, color-coded by Booster Version

## Setup and Installation

### 1. Install Required Packages

Run the following command in your terminal:

```bash
python3.11 -m pip install pandas dash plotly
```

### 2. Verify Dataset

Make sure the `spacex_launch_dash.csv` file is in the same directory as the app.

### 3. Run the Application

```bash
python3.11 spacex_dash_app.py
```

The application will start on port 8050.

### 4. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8050
```

Or use the "Launch Application" feature in VS Code:
- Click on "Others" in the left navigation pane
- Click "Launch Application"
- Enter port number: 8050
- Click "Your Application"

## Using the Dashboard

### Task 1: Launch Site Selection
- Use the dropdown menu to select a specific launch site or "All Sites"
- The dropdown is searchable for easy navigation

### Task 2: Success Pie Chart
- When "All Sites" is selected: Shows total successful launches by each site
- When a specific site is selected: Shows success vs. failure count for that site

### Task 3: Payload Range Selection
- Use the range slider to select a payload mass range
- Values range from 0 to 10,000 kg in 1,000 kg increments

### Task 4: Success-Payload Scatter Chart
- X-axis: Payload Mass (kg)
- Y-axis: Launch Outcome (0 = Failure, 1 = Success)
- Color: Booster Version Category
- Filters by selected site and payload range

## Analysis Questions

Use the dashboard to answer these questions:

1. Which site has the largest successful launches?
2. Which site has the highest launch success rate?
3. Which payload range(s) has the highest launch success rate?
4. Which payload range(s) has the lowest launch success rate?
5. Which F9 Booster version (v1.0, v1.1, FT, B4, B5, etc.) has the highest launch success rate?

## Files

- `spacex_dash_app.py` - Main dashboard application
- `spacex_launch_dash.csv` - SpaceX launch data
- `README_DASHBOARD.md` - This file

## Technologies Used

- **Python 3.11**
- **Pandas** - Data manipulation and analysis
- **Plotly Dash** - Interactive web application framework
- **Plotly Express** - High-level plotting library

## Author

Dashboard implementation based on IBM Data Science Professional Certificate curriculum.
