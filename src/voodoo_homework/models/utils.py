def dataframe_to_dict(df, numeric_cols, categorical_cols):
    return {col: df[col].values for col in numeric_cols + categorical_cols}
