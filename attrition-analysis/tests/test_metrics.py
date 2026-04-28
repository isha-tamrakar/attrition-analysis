import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


# --- Fixtures ---

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "No", "No", "Yes"],
        "overtime": ["Yes", "No", "Yes", "No"],
        "monthly_income": [3000, 5000, 4000, 6000],
        "job_satisfaction": [1, 3, 2, 4],
    })


# --- attrition_rate ---

def test_attrition_rate_half():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["Yes", "Yes"],
    })
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["No", "No"],
    })
    assert attrition_rate(df) == 0.0


def test_attrition_rate_rounds_to_two_decimals():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3],
        "attrition": ["Yes", "No", "No"],
    })
    assert attrition_rate(df) == 33.33


# --- attrition_by_department ---

def test_attrition_by_department_columns(simple_df):
    result = attrition_by_department(simple_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(simple_df):
    result = attrition_by_department(simple_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    hr = result[result["department"] == "HR"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0
    assert hr["employees"] == 2
    assert hr["leavers"] == 1
    assert hr["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "department": ["Sales", "Sales", "HR", "HR", "HR"],
        "attrition": ["Yes", "Yes", "Yes", "No", "No"],
    })
    result = attrition_by_department(df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(simple_df):
    result = attrition_by_overtime(simple_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "overtime": ["Yes", "Yes", "No", "No"],
        "attrition": ["Yes", "Yes", "No", "No"],
    })
    result = attrition_by_overtime(df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 0.0


def test_attrition_by_overtime_higher_for_overtime():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "overtime": ["Yes", "Yes", "Yes", "No", "No", "No"],
        "attrition": ["Yes", "Yes", "No", "Yes", "No", "No"],
    })
    result = attrition_by_overtime(df)
    yes_rate = result[result["overtime"] == "Yes"]["attrition_rate"].iloc[0]
    no_rate = result[result["overtime"] == "No"]["attrition_rate"].iloc[0]
    assert yes_rate > no_rate


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns():
    df = pd.DataFrame({
        "attrition": ["Yes", "No"],
        "monthly_income": [3000, 6000],
    })
    result = average_income_by_attrition(df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values():
    df = pd.DataFrame({
        "attrition": ["Yes", "Yes", "No", "No"],
        "monthly_income": [3000, 5000, 7000, 9000],
    })
    result = average_income_by_attrition(df)
    yes_income = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_income = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert yes_income == 4000.0
    assert no_income == 8000.0


def test_average_income_leavers_earn_less():
    df = pd.DataFrame({
        "attrition": ["Yes", "Yes", "No", "No"],
        "monthly_income": [3000, 4000, 7000, 8000],
    })
    result = average_income_by_attrition(df)
    yes_income = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_income = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert yes_income < no_income


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(simple_df):
    result = satisfaction_summary(simple_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_is_per_group_not_share_of_leavers():
    # Group 1: 2 employees, both left  → should be 100%, not 50%
    # Group 2: 2 employees, none left  → should be 0%
    # Total leavers = 2; wrong denominator would give 2/2*100=100% and 0/2*100=0%
    # which accidentally passes — use 4 total leavers to expose the bug
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "job_satisfaction": [1, 1, 2, 2, 3, 3],
        "attrition": ["Yes", "Yes", "No", "No", "Yes", "Yes"],
    })
    result = satisfaction_summary(df)
    group1 = result[result["job_satisfaction"] == 1].iloc[0]
    group2 = result[result["job_satisfaction"] == 2].iloc[0]
    group3 = result[result["job_satisfaction"] == 3].iloc[0]
    assert group1["attrition_rate"] == 100.0  # 2/2
    assert group2["attrition_rate"] == 0.0    # 0/2
    assert group3["attrition_rate"] == 100.0  # 2/2


def test_satisfaction_summary_sorted_by_satisfaction():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "job_satisfaction": [3, 1, 4, 2],
        "attrition": ["Yes", "No", "Yes", "No"],
    })
    result = satisfaction_summary(df)
    levels = result["job_satisfaction"].tolist()
    assert levels == sorted(levels)
