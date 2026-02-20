import pandas as pd
import numpy as np


def create_features(df):
    df = df.copy()
    
    df["stock_turnover_rate"] = np.where(
        df["current_stock"] > 0,
        df["monthly_sales"] / df["current_stock"],
        0
    )
    
    df["safety_stock_ratio"] = np.where(
        df["current_stock"] > 0,
        df["minimum_stock"] / df["current_stock"],
        0
    )
    
    df["stock_coverage_days"] = np.where(
        df["monthly_sales"] > 0,
        df["current_stock"] / (df["monthly_sales"] / 30),
        0
    )
    
    df["stock_pressure_index"] = np.where(
        df["current_stock"] > 0,
        df["sales_last_7_days"] / df["current_stock"],
        0
    )
    
    df["inventory_value"] = df["current_stock"] * df["unit_cost"]
    
    df["excess_stock"] = np.where(
        df["current_stock"] > df["maximum_stock"],
        df["current_stock"] - df["maximum_stock"],
        0
    )
    
    df["stock_deficit"] = np.where(
        df["current_stock"] < df["minimum_stock"],
        df["minimum_stock"] - df["current_stock"],
        0
    )
    
    df["sales_velocity"] = np.where(
        df["sales_last_30_days"] > 0,
        df["sales_last_7_days"] / (df["sales_last_30_days"] / 4),
        0
    )
    
    df["reorder_urgency"] = np.where(
        df["lead_time_days"] > 0,
        (df["minimum_stock"] - df["current_stock"]) / df["lead_time_days"],
        0
    )
    
    return df
