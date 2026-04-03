# range_finder — Weekly SPX Range Prediction & Credit Spread Placement
#
# Modules:
#   data_collector  — fetch/store SPX, VIX, FRED macro data (SQLite)
#   feature_builder — compute HAR + VIX + HV + macro features
#   har_model       — fit HAR regression, generate weekly range forecast
#   spread_levels   — convert forecast → actionable credit spread parameters
#   gex_bridge      — feed live GEX dashboard data into the range model
